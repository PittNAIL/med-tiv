import time
import uuid
import aiohttp
import requests
import regex as re
import openai
import os
import torch
from vllm import SamplingParams
from typing import Dict, Any, List, Tuple, Optional
from config import ModelConfig, ToolConfig
from transformers import AutoTokenizer
import asyncio
import random
import subprocess

# 1) A sanitizer that strips all embedded NULs (and, optionally, any
#    other C0 control characters except common whitespace).
CONTROL_CHAR_RE = re.compile(
    # this matches U+0000 through U+001F, excluding tab(09), LF(0A), CR(0D)
    r'[\x00-\x08\x0B\x0C\x0E-\x1F]'
)

def sanitize_request(obj: Any) -> Any:
    """
    Recursively walk through obj and:
      - For dicts: sanitize each value
      - For lists/tuples: sanitize each element
      - For strings: remove embedded nulls (and other control chars)
      - Leave other types untouched
    """
    if isinstance(obj, dict):
        return {sanitize_request(key): sanitize_request(val) for key, val in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(sanitize_request(item) for item in obj)
    elif isinstance(obj, str):
        return CONTROL_CHAR_RE.sub('', obj)
    else:
        return obj
    
class ModelService:
    """verl-tool model inference service"""
    
    def __init__(self, model_config: ModelConfig, tool_config: ToolConfig):
        """initialize model service"""
        self.model_config = model_config
        self.tool_config = tool_config
        self.model = None
        self.session = None
        self.vllm_processes = []
        self.clients = []
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model)
        self.encode_lock = asyncio.Lock()
        if self.tool_config.mtrl_sep is None:
            messages = [{"role": "system", "content": "{obs}"}]
            self.tool_config.mtrl_sep = "\n" + self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    def call_tool_server(self, trajectory_ids: List[str], actions: List[str], finish: List[bool], **kwargs: Dict[str, List[Any]]) -> Dict[str, Any]:
        """querying the tool server for the observation and done flag"""
        server_url = self.tool_config.tool_server_url
        data = {
            "trajectory_ids": trajectory_ids,
            "actions": actions,
            "finish": finish,
            **kwargs
        }
        try:
            data = sanitize_request(data)
            response = requests.post(server_url, json=data)
            response.raise_for_status()
            result = response.json()
            return result   
        except Exception as e:
            print(f"Error calling tool server: {str(e)}")
            return {
                "observations": [f"Error calling tool server: {str(e)}" for _ in range(len(trajectory_ids))],
                "dones": [True for _ in range(len(trajectory_ids))],
                "valids": [False for _ in range(len(trajectory_ids))]
            }
    
    async def call_tool_server_async(self, trajectory_ids: List[str], actions: List[str], finish: List[bool], **kwargs: Dict[str, List[Any]]) -> Dict[str, Any]:
        """querying the tool server for the observation and done flag using aiohttp"""
        server_url = self.tool_config.tool_server_url
        data = {
            "trajectory_ids": trajectory_ids,
            "actions": actions,
            "finish": finish,
            **kwargs
        }
        
        if self.session is None:
            self.session = aiohttp.ClientSession()
            
        try:
            data = sanitize_request(data)
            async with self.session.post(server_url, json=data) as response:
                response.raise_for_status()
                result = await response.json()
                return result
        except Exception as e:
            print(f"Error calling tool server: {str(e)}")
            return {
                "observations": [f"Error calling tool server: {str(e)}" for _ in range(len(trajectory_ids))],
                "dones": [True for _ in range(len(trajectory_ids))],
                "valids": [False for _ in range(len(trajectory_ids))]
            }
    
    async def post_process_observations(self, next_obs: List[str], dones: List[bool], valid_action: List[bool], finishs: List[bool]):
        """Process observations using the tokenizer with proper async locks"""
        next_obs = [obs if not done else "" for obs, done in zip(next_obs, dones)]
        async with self.encode_lock:
            mtrl_sep = self.tool_config.mtrl_sep
            if self.tool_config.truncate_obs_side == 'left':
                next_obs_ids = self.tokenizer(
                    next_obs,
                    padding='longest',
                    return_tensors='pt',
                    add_special_tokens=False,
                    padding_side='left',
                )['input_ids'].to(torch.int64)
                if next_obs_ids.shape[1] > self.tool_config.max_obs_length:
                    print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.tool_config.max_obs_length}")
                    next_obs_ids = next_obs_ids[:, -self.tool_config.max_obs_length:]
            elif self.tool_config.truncate_obs_side == 'right':
                next_obs_ids = self.tokenizer(
                    next_obs,
                    padding='longest',
                    return_tensors='pt',
                    add_special_tokens=False,
                    padding_side='right',
                )['input_ids'].to(torch.int64)
                if next_obs_ids.shape[1] > self.tool_config.max_obs_length:
                    print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.tool_config.max_obs_length}")
                    next_obs_ids = next_obs_ids[:, :self.tool_config.max_obs_length]
            else:
                raise ValueError(f"Invalid truncate_obs_side: {self.tool_config.truncate_obs_side}")
            
            if self.tool_config.enable_mtrl:
                next_obs = self.tokenizer.batch_decode(next_obs_ids, skip_special_tokens=True)
                processed_next_obs = []
                for i in range(len(next_obs)):
                    if finishs[i] or dones[i]:
                        assert next_obs[i] == "", f"next_obs should be empty when finishs is True, but got {next_obs[i]}"
                        processed_next_obs.append("")
                    elif valid_action[i]:
                        processed_next_obs.append(mtrl_sep.format(obs=next_obs[i]))
                    else:
                        processed_next_obs.append(mtrl_sep.format(obs="Your action is not valid, please check the format and try again." + next_obs[i]))
                next_obs = processed_next_obs
                next_obs_ids = self.tokenizer(
                    next_obs,
                    padding='longest',
                    return_tensors='pt',
                    add_special_tokens=False,
                )['input_ids'].to(torch.int64)
            
            next_obs = self.tokenizer.batch_decode(next_obs_ids, skip_special_tokens=True)
            return next_obs
    
    def _extract_logprobs_from_choice(self, choice) -> Optional[Dict]:
        """Extract logprobs from a vLLM completion choice into OpenAI-compatible format."""
        if not hasattr(choice, 'logprobs') or choice.logprobs is None:
            return None
        
        lp = choice.logprobs
        
        # Handle case where logprobs attributes might be None
        if not hasattr(lp, 'tokens') or lp.tokens is None:
            return None
        
        content = []
        num_tokens = len(lp.tokens)
        
        for j in range(num_tokens):
            token_entry = {
                "token": lp.tokens[j] if lp.tokens else "",
                "logprob": lp.token_logprobs[j] if lp.token_logprobs and j < len(lp.token_logprobs) else 0.0,
                "top_logprobs": []
            }
            
            # Extract top_logprobs if available
            if lp.top_logprobs and j < len(lp.top_logprobs) and lp.top_logprobs[j]:
                token_entry["top_logprobs"] = [
                    {"token": tok, "logprob": logprob}
                    for tok, logprob in lp.top_logprobs[j].items()
                ]
            
            content.append(token_entry)
        
        return {"content": content}
    
    async def _postprocess_responses(self, outputs, action_step: int) -> Tuple[List[str], List[bool], List[str], List[Optional[Dict]]]:
        """Process responses to stop at python operation or answer operation."""
        active_responses = []
        active_finish_reasons = []
        active_logprobs = []
        finishes = []
        
        for i in range(len(outputs.choices)):
            choice = outputs.choices[i]
            response_text = choice.text
            finish_reason = choice.finish_reason
            
            # Extract logprobs using helper method
            logprobs_data = self._extract_logprobs_from_choice(choice)
            active_logprobs.append(logprobs_data)
            
            # Determine if this is a finishing turn
            finish = True
            if finish_reason == "stop" and choice.stop_reason is not None:
                response_text = response_text + choice.stop_reason
                if self.tool_config.enable_mtrl:
                    response_text += self.tool_config.turn_end_token
                finish = False
            
            if finish and self.tool_config.min_turns > action_step:
                finish = False
                if self.tool_config.enable_mtrl:
                    if self.tool_config.action_stop_tokens:
                        response_text += self.tool_config.action_stop_tokens[0]
                    response_text += self.tool_config.turn_end_token
            
            active_responses.append(response_text)
            active_finish_reasons.append(finish_reason)
            finishes.append(finish)
        
        return active_responses, finishes, active_finish_reasons, active_logprobs
        
    def load_model(self):
        """load the model using VLLM backend"""
        print(f"Loading Model using VLLM: {self.model_config.model}...")
        
        vllm_args = []
        for k, v in self.model_config.__dict__.items():
            if k not in ["model", "api_key", "num_models", "host", "port"]:
                vllm_args.append(f"--{k.replace('_', '-')}")
                if not isinstance(v, bool):
                    vllm_args.append(str(v))
        
        host = "0.0.0.0"
        num_models = self.model_config.num_models
        ports = random.sample(range(8000, 9000), num_models)
        self.vllm_processes = []
        gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join([str(i) for i in range(torch.cuda.device_count())])).split(",")
        tensor_parallel_size = self.model_config.tensor_parallel_size
        gpu_ids_per_model = [gpu_ids[i:i+tensor_parallel_size] for i in range(0, len(gpu_ids), tensor_parallel_size)]
        assert len(gpu_ids) >= num_models * tensor_parallel_size, f"Not enough GPUs available: {len(gpu_ids)} < {num_models * tensor_parallel_size}"
        
        for i in range(num_models):
            cmd = [
                "vllm", "serve", self.model_config.model, "--api-key", "token-abc123",
                "--host", host, "--port", str(ports[i]), 
                "--disable-uvicorn-access-log", "--disable-log-stats", "--disable-log-requests"
            ] + vllm_args
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids_per_model[i])
            env["VLLM_LOGGING_LEVEL"] = "ERROR"
            vllm_process = subprocess.Popen(cmd, env=env)
            self.vllm_processes.append(vllm_process)
        
        self.clients = [
            openai.Client(api_key="token-abc123", base_url=f"http://{host}:{ports[i]}/v1") for i in range(num_models)
        ]
        
        # Wait for the service to start
        max_retries = 60
        retry_interval = 10
        vllm_model_status = [False for _ in range(num_models)]
        
        for i in range(max_retries):
            for j in range(num_models):
                if vllm_model_status[j]:
                    continue
                try:
                    response = self.clients[j].models.list()
                    vllm_model_status[j] = True
                    print(f"vLLM instance model-{j} status: {response}")
                except Exception as e:
                    continue
            if all(vllm_model_status):
                print(f"✅ vLLM service started successfully with model: {self.model_config.model}")
                return     
            else:
                time.sleep(retry_interval)
        
        print("Failed to start one or more vLLM services. Check vLLM logs.")
        for process in self.vllm_processes:
            stderr = process.stderr.read() if process.stderr else "No stderr"
            print(f"vLLM stderr: {stderr}")
            process.terminate()
        
        raise RuntimeError("Failed to start vLLM services")
    
    async def send_request(self, client, prompts: List[str], model: str, sampling_params: dict):
        """Send request to vLLM server"""
        sampling_params = sampling_params.copy()
        
        async with self.encode_lock:
            prompt_lens = [len(self.tokenizer.encode(prompt)) for prompt in prompts]
            max_prompt_tokens = max(prompt_lens)
        
        sampling_params['max_tokens'] = min(
            max(self.model_config.max_model_len - max_prompt_tokens, 0), 
            sampling_params['max_tokens']
        )
        
        # Extract seed if present (not a standard OpenAI completions param, pass via extra_body)
        seed = sampling_params.pop('seed', None)
        
        extra_body = {}
        if seed is not None:
            extra_body['seed'] = seed
        
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.completions.create(
                model=model,
                prompt=prompts,
                echo=False,
                stream=False,
                extra_body=extra_body if extra_body else None,
                **sampling_params
            )
        )
        return response
    
    async def generate_with_tools(self, prompts: List[str], sampling_params: dict) -> Tuple[List[str], List[str], List[Optional[Dict]]]:
        """
        Generate text with tool calls in a multi-turn loop.
        
        Args:
            prompts: Initial prompts for generation
            sampling_params: Sampling parameters for the model
            
        Returns:
            Tuple of (full_responses, finish_reasons, final_logprobs)
        """
        client = random.choice(self.clients)
        assert sampling_params.get("n", 1) <= 1, "n > 1 is not supported yet for tool generation"
        
        contexts = list(prompts)  # Make a copy
        final_responses = ["" for _ in range(len(prompts))]
        final_logprobs: List[Optional[Dict]] = [None for _ in range(len(prompts))]
        traj_ids = [str(uuid.uuid4()) for _ in range(len(prompts))]
        active_masks = [True for _ in range(len(prompts))]
        finish_reasons = [None for _ in range(len(prompts))]
        model = self.model_config.model
        
        for action_step in range(self.tool_config.max_turns + 1):
            if action_step == self.tool_config.max_turns:
                # Last turn: don't stop by action stop tokens
                if "stop" in sampling_params and sampling_params["stop"] is not None:
                    sampling_params = sampling_params.copy()  # Don't mutate original
                    for action_stop_token in self.tool_config.action_stop_tokens:
                        if action_stop_token in sampling_params["stop"]:
                            sampling_params["stop"].remove(action_stop_token)
                
            active_traj_ids = [traj_ids[i] for i in range(len(traj_ids)) if active_masks[i]]
            active_contexts = [contexts[i] for i in range(len(contexts)) if active_masks[i]]
            
            if len(active_contexts) == 0:
                break
            
            # Send request
            outputs = await self.send_request(client, active_contexts, model, sampling_params)
            active_responses, finishes, active_finish_reasons, active_logprobs = await self._postprocess_responses(outputs, action_step)
            
            # Call tool server
            tool_responses = await self.call_tool_server_async(
                active_traj_ids,
                active_responses,
                finishes
            )
                
            observations = await self.post_process_observations(
                tool_responses["observations"], 
                tool_responses["dones"], 
                tool_responses["valids"], 
                finishes
            )
            dones = tool_responses["dones"]
            
            # Update state for each trajectory
            active_idx = 0
            for i in range(len(contexts)):
                if active_masks[i]:
                    contexts[i] += active_responses[active_idx] + observations[active_idx]
                    final_responses[i] += active_responses[active_idx] + observations[active_idx]
                    finish_reasons[i] = active_finish_reasons[active_idx]
                    
                    # Accumulate logprobs across turns
                    if active_logprobs[active_idx] is not None:
                        if final_logprobs[i] is None:
                            final_logprobs[i] = {"content": []}
                        final_logprobs[i]["content"].extend(active_logprobs[active_idx]["content"])
                    
                    active_masks[i] = not dones[active_idx]
                    active_idx += 1
            
        return final_responses, finish_reasons, final_logprobs
    
    async def chat_completions_async(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Process API request and generate response"""
        if "messages" not in body or not body["messages"]:
            raise ValueError("No messages found in the request.")
        if 'user' not in [message["role"] for message in body["messages"]]:
            raise ValueError("No user message found in the request.")
        
        assert body["model"] == self.model_config.model, f"model mismatch: {body['model']} != {self.model_config.model}"
        
        async with self.encode_lock:
            prompt = self.tokenizer.apply_chat_template(
                body['messages'],
                add_generation_prompt=True,
                tokenize=False
            )
        
        if body.get('n', 1) > 1:
            prompts = [prompt for _ in range(body["n"])]
        else:
            prompts = [prompt]

        # Build sampling params
        sampling_params = {
            "temperature": body.get("temperature", 1.0),
            "max_tokens": body.get("max_tokens", body.get("max_completion_tokens", 512)),
            "top_p": body.get("top_p", 1.0),
            "stop": list(set(body.get("stop", []) + self.tool_config.action_stop_tokens)),
        }
        
        # Handle seed parameter for reproducibility
        if body.get("seed") is not None:
            sampling_params["seed"] = body.get("seed")
        
        # Handle logprobs request
        if body.get("top_logprobs"):
            sampling_params["logprobs"] = body.get("top_logprobs")
        elif body.get("logprobs"):
            sampling_params["logprobs"] = 10  # Default number of top logprobs

        all_responses, finish_reasons, all_logprobs = await self.generate_with_tools(prompts, sampling_params)
        
        async with self.encode_lock:
            prompt_tokens = len(self.tokenizer.encode(prompt))
            completion_tokens = sum(len(self.tokenizer.encode(r)) for r in all_responses)
            total_tokens = prompt_tokens + completion_tokens
        
        # Build response
        choices = []
        for i in range(len(all_responses)):
            choice = {
                "index": i,
                "message": {
                    "role": "assistant",
                    "content": all_responses[i],
                },
                "finish_reason": finish_reasons[i]
            }
            # Only include logprobs if requested and available
            if (body.get("logprobs") or body.get("top_logprobs")) and all_logprobs[i] is not None:
                choice["logprobs"] = all_logprobs[i]
            choices.append(choice)
        
        return {
            "id": f"chatcmpl-{str(uuid.uuid4())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_config.model,
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            } 
        }
    
    def chat_completions(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous wrapper for chat_completions"""
        return asyncio.run(self.chat_completions_async(body))
        
    async def completions_async(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Process API request and generate response async"""
        if 'prompt' not in body:
            raise ValueError("No prompt found in the request.")
        assert body["model"] == self.model_config.model, f"model mismatch: {body['model']} != {self.model_config.model}"
        
        prompt = body['prompt']

        if body.get('n', 1) > 1:
            prompts = [prompt for _ in range(body["n"])]
        else:
            prompts = [prompt]

        sampling_params = {
            "temperature": body.get("temperature", 1.0),
            "max_tokens": body.get("max_tokens", body.get("max_completion_tokens", 512)),
            "top_p": body.get("top_p", 1.0),
            "stop": list(set(body.get("stop", []) + self.tool_config.action_stop_tokens)),
        }
        
        # Handle seed parameter for reproducibility
        if body.get("seed") is not None:
            sampling_params["seed"] = body.get("seed")
        
        # Handle logprobs request
        if body.get("top_logprobs"):
            sampling_params["logprobs"] = body.get("top_logprobs")
        elif body.get("logprobs"):
            sampling_params["logprobs"] = 5

        all_responses, finish_reasons, all_logprobs = await self.generate_with_tools(prompts, sampling_params)
        
        async with self.encode_lock:
            prompt_tokens = len(self.tokenizer.encode(prompt))
            completion_tokens = sum(len(self.tokenizer.encode(r)) for r in all_responses)
            total_tokens = prompt_tokens + completion_tokens
        
        # Build response
        choices = []
        for i in range(len(all_responses)):
            choice = {
                "index": i,
                "text": all_responses[i],
                "finish_reason": finish_reasons[i]
            }
            if (body.get("logprobs") or body.get("top_logprobs")) and all_logprobs[i] is not None:
                choice["logprobs"] = all_logprobs[i]
            choices.append(choice)
        
        return {
            "id": f"cmpl-{str(uuid.uuid4())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": self.model_config.model,
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            } 
        }
    
    def completions(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous wrapper for completions_async"""
        return asyncio.run(self.completions_async(body))
        
    async def close(self):
        """Close any resources (like HTTP sessions and processes) when shutting down"""
        if self.session:
            await self.session.close()
            self.session = None
            
        for process in self.vllm_processes:
            if process:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    
        self.vllm_processes = []
        self.clients = []
        
    def __del__(self):
        """Destructor to ensure resources are cleaned up"""
        try:
            asyncio.run(self.close())
        except RuntimeError:
            pass