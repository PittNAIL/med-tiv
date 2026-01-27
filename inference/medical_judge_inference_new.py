import json
import argparse
from typing import List, Dict, Any, Optional
import asyncio
import aiohttp
from tqdm import tqdm
import re
import os
import math

# Fixed seeds for reproducibility (matching local inference version)
FIXED_SEEDS = [
    42, 123, 456, 789, 1024, 2048, 3141, 5926, 
    8192, 16384, 32768, 65536, 271828, 314159, 161803, 628318,
    100, 200, 300, 400, 500, 600, 700, 800,
    1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888
]

class MedicalJudgeInference:
    """
    Inference service for medical judge agent with search tool support
    Uses the vLLM-based API service with logprob extraction
    Supports multiple samples per instruction for difficulty filtering
    """
    
    def __init__(self, api_base_url: str, model_name: str, debug: bool = False):
        self.api_base_url = api_base_url.rstrip('/')
        self.model_name = model_name
        self.session = None
        self.debug = debug
        self.seeds = FIXED_SEEDS
        
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
            digits = re.findall(r'\b[01]\b', response)
            predicted_label = int(digits[-1]) if digits else 0
        
        return {
            "prm_evaluation": response,
            "predicted_label": predicted_label
        }

    def _extract_token_logprobs(self, logprobs_data: Optional[Dict]) -> List[Dict]:
        if logprobs_data is None:
            return []
        
        if isinstance(logprobs_data, list):
            return logprobs_data
        
        if isinstance(logprobs_data, dict):
            if "content" in logprobs_data:
                return logprobs_data["content"]
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
        DEFAULT_LOGPROB = -99.0
        
        token_logprobs = self._extract_token_logprobs(logprobs_data)
        if not token_logprobs:
            return {"prob_0": 0.5, "prob_1": 0.5}

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
            return {"prob_0": 0.5, "prob_1": 0.5}

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
            
            break

        return {"prob_0": 0.5, "prob_1": 0.5}

    async def inference_single_sample_async(
        self, 
        instruction: str,
        sample_idx: int,
        max_tokens: int = 4096, 
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_logprobs: int = 20
    ) -> Dict[str, Any]:
        """Run inference for a single sample of an instruction"""
        await self._ensure_session()
        
        messages = self.create_messages(instruction)
        
        # Use fixed seed for this sample index
        seed = self.seeds[sample_idx % len(self.seeds)]
        
        request_body = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "logprobs": True,
            "top_logprobs": top_logprobs,
            "seed": seed  # Add seed for reproducibility
        }
        
        sample_result = {
            "sample_idx": sample_idx,
            "seed": seed,
            "prm_evaluation": "",
            "predicted_label": 0,
            "prob_0": 0.5,
            "prob_1": 0.5,
            "prm_score": 0.5,
            "error": None
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
            logprobs_data = choice.get("logprobs")
            
            parsed = self.parse_response(generated_text)
            answer_probs = self._extract_answer_probs(logprobs_data)
            
            sample_result["prm_evaluation"] = parsed["prm_evaluation"]
            sample_result["predicted_label"] = parsed["predicted_label"]
            sample_result["prob_0"] = answer_probs["prob_0"]
            sample_result["prob_1"] = answer_probs["prob_1"]
            sample_result["prm_score"] = answer_probs["prob_1"]
                
        except Exception as e:
            self._debug_print(f"Error during inference: {str(e)}")
            sample_result["error"] = str(e)
        
        return sample_result

    async def inference_multi_sample_async(
        self, 
        data_item: Dict[str, Any],
        item_idx: int,
        num_samples: int = 4,
        max_tokens: int = 4096, 
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_logprobs: int = 20
    ) -> Dict[str, Any]:
        """Run inference multiple times for a single data item"""
        
        instruction = data_item["instruction"]
        
        # Run all samples concurrently
        tasks = [
            self.inference_single_sample_async(
                instruction, i, max_tokens, temperature, top_p, top_logprobs
            )
            for i in range(num_samples)
        ]
        
        samples = await asyncio.gather(*tasks)
        
        # Aggregate results
        predicted_labels = [s["predicted_label"] for s in samples]
        count_0 = sum(1 for l in predicted_labels if l == 0)
        count_1 = sum(1 for l in predicted_labels if l == 1)
        
        # Determine difficulty category based on agreement ratio
        max_count = max(count_0, count_1)
        min_count = min(count_0, count_1)
        
        if min_count == 0:
            difficulty = "easy"  # All agree (8-0)
        elif min_count <= 3 and num_samples == 8:  # 7-1, 6-2, 5-3 for 8 samples
            difficulty = "hard"
        elif min_count == 1 and num_samples == 4:  # 3-1 for 4 samples
            difficulty = "hard"
        elif count_0 == count_1:
            difficulty = "medium"  # Even split (4-4 or 2-2)
        else:
            difficulty = "mixed"
        
        # Extract metadata fields if present
        metadata = data_item.get("metadata", {})
        
        result = {
            "item_idx": item_idx,
            "instruction": instruction,
            "label": data_item.get("label"),
            "query_id": metadata.get("query_id"),
            "response_id": metadata.get("response_id"),
            "is_correct_trace": metadata.get("is_correct_trace"),
            "predicted_answer": metadata.get("predicted_answer"),
            "ground_truth": metadata.get("ground_truth"),
            # Preserve all original fields from input
            **{k: v for k, v in data_item.items() if k not in ["instruction", "label", "metadata"]},
            "metadata": metadata,  # Keep original metadata too
            "samples": samples,
            "predicted_labels": predicted_labels,
            "count_0": count_0,
            "count_1": count_1,
            "difficulty": difficulty,
            "majority_vote": 1 if count_1 > count_0 else 0,
            "avg_prob_1": sum(s["prob_1"] for s in samples) / num_samples,
        }
        
        # Check correctness if label exists
        if "label" in data_item:
            result["majority_correct"] = (result["majority_vote"] == data_item["label"])
            result["individual_correct"] = [
                (s["predicted_label"] == data_item["label"]) for s in samples
            ]
        
        return result

    async def inference_batch_async(
        self, 
        data_items: List[Dict], 
        num_samples: int,
        max_tokens: int,
        temperature: float, 
        top_p: float, 
        max_concurrent: int,
        top_logprobs: int = 20
    ) -> List[Dict]:
        """Run multi-sample inference on a batch with concurrency control"""
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(item, index):
            async with semaphore:
                result = await self.inference_multi_sample_async(
                    item, index, num_samples, max_tokens, temperature, top_p, top_logprobs
                )
                return (index, result)
        
        tasks = [process_with_semaphore(item, i) for i, item in enumerate(data_items)]
        
        with tqdm(total=len(tasks), desc=f"Running inference ({num_samples} samples each)") as pbar:
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


def filter_by_difficulty(results: List[Dict], difficulty_type: str) -> List[Dict]:
    """Filter results by difficulty type"""
    return [r for r in results if r["difficulty"] == difficulty_type]


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
                args.num_samples,
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
        print(f"\n{'='*50}")
        print("RESULTS SUMMARY")
        print(f"{'='*50}")
        print(f"Total instructions processed: {len(results)}")
        print(f"Samples per instruction: {args.num_samples}")
        print(f"Total samples generated: {len(results) * args.num_samples}")
        
        # Difficulty distribution
        difficulty_counts = {}
        for r in results:
            d = r.get("difficulty", "unknown")
            difficulty_counts[d] = difficulty_counts.get(d, 0) + 1
        
        print(f"\nDifficulty Distribution:")
        for diff in ["easy", "hard", "medium", "mixed"]:
            if diff in difficulty_counts:
                count = difficulty_counts[diff]
                pct = count/len(results)*100
                if diff == "easy":
                    desc = "all agree, 8-0"
                elif diff == "hard":
                    desc = "5-3, 6-2, 7-1"
                elif diff == "medium":
                    desc = "even split, 4-4"
                else:
                    desc = "other"
                print(f"  {diff.title():10s} ({desc}): {count} ({pct:.1f}%)")
        
        # Accuracy stats if labels exist
        results_with_labels = [r for r in results if r.get("label") is not None]
        if results_with_labels:
            majority_correct = sum(1 for r in results_with_labels if r.get("majority_correct", False))
            print(f"\nMajority Vote Accuracy: {majority_correct}/{len(results_with_labels)} = {majority_correct/len(results_with_labels):.4f}")
            
            # Per-sample accuracy
            all_individual = []
            for r in results_with_labels:
                all_individual.extend(r.get("individual_correct", []))
            if all_individual:
                individual_correct = sum(all_individual)
                print(f"Individual Sample Accuracy: {individual_correct}/{len(all_individual)} = {individual_correct/len(all_individual):.4f}")
        
        avg_prob_1 = sum(r.get("avg_prob_1", 0) for r in results) / len(results)
        print(f"\nAverage prob_1 (PRM score): {avg_prob_1:.4f}")
    
    # Save full results
    print(f"\nSaving full results to {args.output_file}")
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save filtered results if requested
    if args.save_hard_only:
        hard_results = filter_by_difficulty(results, "hard")
        hard_output = args.output_file.replace('.json', '_hard_only.json')
        with open(hard_output, 'w') as f:
            json.dump(hard_results, f, indent=2)
        print(f"Saved {len(hard_results)} hard samples to {hard_output}")
    
    # Clean up checkpoint
    if args.resume and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("Checkpoint file removed")
    
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description="PRM Multi-Sample Inference for Difficulty Filtering")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON/JSONL file")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file")
    parser.add_argument("--api_base_url", type=str, default="http://localhost:5000", help="API base URL")
    parser.add_argument("--model_name", type=str, required=True, help="Model name for API")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of samples per instruction")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (use >0 for diversity)")
    parser.add_argument("--top_p", type=float, default=0.8, help="Top-p sampling")
    parser.add_argument("--top_logprobs", type=int, default=20, help="Number of top logprobs to return")
    parser.add_argument("--max_concurrent", type=int, default=8, help="Max concurrent requests")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--checkpoint_every", type=int, default=5000, help="Checkpoint frequency")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of items (0=no limit)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--save_hard_only", action="store_true", help="Also save filtered hard samples (3-1 split)")
    
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()