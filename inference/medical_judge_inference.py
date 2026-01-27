import json
import argparse
from typing import List, Dict, Any, Optional
import asyncio
import aiohttp
from tqdm import tqdm
import re
import os
import math

class MedicalJudgeInference:
    """
    Inference service for medical judge agent with search tool support
    Uses the vLLM-based API service with logprob extraction
    """
    
    def __init__(self, api_base_url: str, model_name: str, debug: bool = False):
        self.api_base_url = api_base_url.rstrip('/')
        self.model_name = model_name
        self.session = None
        self.debug = debug
        
    async def _ensure_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    def _debug_print(self, msg: str):
        if self.debug:
            print(msg)
    
    def create_messages(self, instruction: str) -> List[Dict[str, str]]:
        return [{"role": "user", "content": instruction}]
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """Extract answer from <answer> tags"""
        answer_match = re.search(r'<answer>\s*(\d+)\s*</answer>', response, re.IGNORECASE)
        
        if answer_match:
            predicted_label = int(answer_match.group(1))
        else:
            # Fallback: look for last digit 0 or 1 in response
            digits = re.findall(r'\b[01]\b', response)
            predicted_label = int(digits[-1]) if digits else 0
        
        return {
            "prm_evaluation": response,
            "predicted_label": predicted_label
        }

    def _extract_token_logprobs(self, logprobs_data: Optional[Dict]) -> List[Dict]:
        """
        Extract the token logprobs list from the API response.
        Handles both formats:
        - OpenAI format: {"content": [...]}
        - Direct list: [...]
        """
        if logprobs_data is None:
            return []
        
        if isinstance(logprobs_data, list):
            return logprobs_data
        
        if isinstance(logprobs_data, dict):
            # Our model_service.py format: {"content": [...]}
            if "content" in logprobs_data:
                return logprobs_data["content"]
            # Some APIs return {"tokens": [...], "token_logprobs": [...], ...}
            if "tokens" in logprobs_data:
                tokens = logprobs_data.get("tokens", [])
                token_logprobs = logprobs_data.get("token_logprobs", [])
                top_logprobs = logprobs_data.get("top_logprobs", [])
                return [
                    {
                        "token": tokens[i] if i < len(tokens) else "",
                        "logprob": token_logprobs[i] if i < len(token_logprobs) else 0.0,
                        "top_logprobs": [
                            {"token": k, "logprob": v} 
                            for k, v in (top_logprobs[i].items() if i < len(top_logprobs) and top_logprobs[i] else {}.items())
                        ]
                    }
                    for i in range(len(tokens))
                ]
        
        return []

    def _extract_answer_probs(self, logprobs_data: Optional[Dict]) -> Dict[str, float]:
        """Extract probabilities with full debug output."""
        DEFAULT_LOGPROB = -99.0
        
        token_logprobs = self._extract_token_logprobs(logprobs_data)
        if not token_logprobs:
            print("[DEBUG] No token_logprobs found")
            return {"prob_0": 0.5, "prob_1": 0.5}

        # Reconstruct text to find <answer>
        full_text = ""
        answer_start_char = -1
        answer_end_char = -1

        for entry in token_logprobs:
            tok = entry.get("token", "")
            full_text += tok
            lower = full_text.lower()
            if answer_start_char == -1 and "<answer>" in lower:
                answer_start_char = lower.index("<answer>") + len("<answer>")
            if answer_start_char != -1 and "</answer>" in lower:
                answer_end_char = lower.index("</answer>")
                break

        if answer_start_char == -1:
            print(f"[DEBUG] No <answer> tag found in: ...{full_text[-200:]}")
            return {"prob_0": 0.5, "prob_1": 0.5}

        # Find token at answer position and dump ALL top logprobs
        char_cursor = 0
        
        for entry in token_logprobs:
            tok = entry.get("token", "")
            tok_len = len(tok)
            token_start = char_cursor
            token_end = char_cursor + tok_len
            char_cursor += tok_len

            if token_end <= answer_start_char:
                continue
            if answer_end_char != -1 and token_start >= answer_end_char:
                break

            top_lps = entry.get("top_logprobs", [])
            
            # Continue with extraction...
            lp_0, lp_1 = DEFAULT_LOGPROB, DEFAULT_LOGPROB
            for lp_obj in top_lps:
                raw_tok = lp_obj.get("token", "")
                clean = raw_tok.replace("▁", "").replace("Ġ", "").strip()
                logprob = lp_obj.get("logprob", DEFAULT_LOGPROB)
                if clean == "0" or clean.startswith("0"):
                    lp_0 = max(lp_0, logprob)
                elif clean == "1" or clean.startswith("1"):
                    lp_1 = max(lp_1, logprob)

            if lp_0 > DEFAULT_LOGPROB or lp_1 > DEFAULT_LOGPROB:
                v_0, v_1 = math.exp(lp_0), math.exp(lp_1)
                total = v_0 + v_1
                return {"prob_0": v_0/total, "prob_1": v_1/total}
            
            break  # Only check first token in answer region

        return {"prob_0": 0.5, "prob_1": 0.5}

    async def inference_single_async(
        self, 
        data_item: Dict[str, Any], 
        max_tokens: int = 4096, 
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_logprobs: int = 20
    ) -> Dict[str, Any]:
        """Run inference on a single data item"""
        await self._ensure_session()
        
        messages = self.create_messages(data_item["instruction"])
        
        request_body = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "logprobs": True,
            "top_logprobs": top_logprobs
        }
        
        try:
            async with self.session.post(
                f"{self.api_base_url}/chat/completions",
                json=request_body,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                response.raise_for_status()
                result = await response.json()
            
            if "choices" not in result or len(result["choices"]) == 0:
                raise ValueError("Invalid response format from API")
            
            choice = result["choices"][0]
            generated_text = choice["message"]["content"]
            
            self._debug_print(f"[DEBUG] Generated text:\n{generated_text[:500]}...")
            
            # Get logprobs - handle both possible locations
            logprobs_data = choice.get("logprobs")
            
            self._debug_print(f"[DEBUG] Logprobs structure: {type(logprobs_data)}")
            if logprobs_data and isinstance(logprobs_data, dict):
                self._debug_print(f"[DEBUG] Logprobs keys: {logprobs_data.keys()}")
                if "content" in logprobs_data:
                    self._debug_print(f"[DEBUG] Number of tokens: {len(logprobs_data['content'])}")
            
            # Parse response for label
            parsed = self.parse_response(generated_text)
            
            # Extract probabilities
            answer_probs = self._extract_answer_probs(logprobs_data)
            
            # Update data item
            data_item["prm_evaluation"] = parsed["prm_evaluation"]
            data_item["predicted_label"] = parsed["predicted_label"]
            data_item["prob_0"] = answer_probs["prob_0"]
            data_item["prob_1"] = answer_probs["prob_1"]
            
            # Use prob_1 as the "correctness" score if no hard label
            data_item["prm_score"] = answer_probs["prob_1"]
            
            if "label" in data_item:
                data_item["prediction_correct"] = (
                    data_item["predicted_label"] == data_item["label"]
                )
                
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            import traceback
            traceback.print_exc()
            
            data_item["prm_evaluation"] = f"Error: {str(e)}"
            data_item["predicted_label"] = 0
            data_item["prob_0"] = 0.5
            data_item["prob_1"] = 0.5
            data_item["prm_score"] = 0.5
            if "label" in data_item:
                data_item["prediction_correct"] = False
        
        return data_item

    async def inference_batch_async(
        self, 
        data_items: List[Dict], 
        max_tokens: int,
        temperature: float, 
        top_p: float, 
        max_concurrent: int,
        top_logprobs: int = 20
    ) -> List[Dict]:
        """Run inference on a batch with concurrency control"""
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(item, index):
            async with semaphore:
                result = await self.inference_single_async(
                    item, max_tokens, temperature, top_p, top_logprobs
                )
                return (index, result)
        
        tasks = [process_with_semaphore(item, i) for i, item in enumerate(data_items)]
        
        with tqdm(total=len(tasks), desc="Running inference") as pbar:
            for coro in asyncio.as_completed(tasks):
                index, result = await coro
                results.append((index, result))
                pbar.update(1)
        
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None


async def main_async(args):
    print(f"Loading data from {args.input_file}...")
    
    # Load input data
    data_items = []
    with open(args.input_file, 'r') as f:
        content = f.read().strip()
        if content.startswith('['):
            data_items = json.loads(content)
        else:
            for line in content.split('\n'):
                if line.strip():
                    data_items.append(json.loads(line))
    
    print(f"Loaded {len(data_items)} items")
    
    if args.limit and args.limit > 0:
        data_items = data_items[:args.limit]
        print(f"Limited to {len(data_items)} items")
    
    # Handle checkpointing
    checkpoint_file = args.output_file + ".checkpoint"
    start_idx = 0
    completed_results = []
    
    if args.resume and os.path.exists(checkpoint_file):
        print(f"Resuming from checkpoint: {checkpoint_file}")
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
            completed_results = checkpoint_data.get('results', [])
            start_idx = len(completed_results)
        print(f"Resuming from index {start_idx}")
    
    remaining_items = data_items[start_idx:]
    
    if not remaining_items:
        print("No remaining items to process")
        results = completed_results
    else:
        inference_service = MedicalJudgeInference(
            args.api_base_url, 
            args.model_name,
            debug=args.debug
        )
        
        batch_size = args.checkpoint_every
        all_new_results = []
        
        for batch_start in range(0, len(remaining_items), batch_size):
            batch_end = min(batch_start + batch_size, len(remaining_items))
            batch_items = remaining_items[batch_start:batch_end]
            
            print(f"Processing batch {batch_start//batch_size + 1}: items {start_idx + batch_start} to {start_idx + batch_end}")
            
            batch_results = await inference_service.inference_batch_async(
                batch_items, 
                args.max_tokens, 
                args.temperature, 
                args.top_p, 
                args.max_concurrent,
                args.top_logprobs
            )
            all_new_results.extend(batch_results)
            
            # Save checkpoint
            if args.resume:
                current_results = completed_results + all_new_results
                with open(checkpoint_file, 'w') as f:
                    json.dump({'results': current_results}, f)
                print(f"Checkpoint saved: {len(current_results)} results")
        
        results = completed_results + all_new_results
        await inference_service.close()
    
    # Calculate and print statistics
    if results:
        correct = sum(1 for r in results if r.get("prediction_correct", False))
        total_with_labels = sum(1 for r in results if "label" in r)
        avg_prob_1 = sum(r.get("prob_1", 0) for r in results) / len(results)
        
        print(f"\n=== Results Summary ===")
        print(f"Total processed: {len(results)}")
        if total_with_labels > 0:
            print(f"Accuracy: {correct}/{total_with_labels} = {correct/total_with_labels:.4f}")
        print(f"Average prob_1 (PRM score): {avg_prob_1:.4f}")
    
    # Save final results
    print(f"Saving results to {args.output_file}")
    with open(args.output_file, 'w') as f:
        if args.output_file.endswith('.jsonl'):
            for item in results:
                f.write(json.dumps(item) + '\n')
        else:
            json.dump(results, f, indent=2)
    
    # Clean up checkpoint
    if args.resume and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("Checkpoint file removed")
    
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="PRM Inference with logprob extraction")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON/JSONL file")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON/JSONL file")
    parser.add_argument("--api_base_url", type=str, default="http://localhost:5000", help="API base URL")
    parser.add_argument("--model_name", type=str, required=True, help="Model name for API")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--top_logprobs", type=int, default=20, help="Number of top logprobs to return")
    parser.add_argument("--max_concurrent", type=int, default=8, help="Max concurrent requests")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--checkpoint_every", type=int, default=5000, help="Checkpoint frequency")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of items (0=no limit)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()