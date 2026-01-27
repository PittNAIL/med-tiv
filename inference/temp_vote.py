#!/usr/bin/env python3
import json
import argparse
from collections import defaultdict, Counter
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Advanced PRM Scaling Benchmarks')
    parser.add_argument('--evaluations_file', type=str, required=True, help='Path to JSONL file')
    return parser.parse_args()

def run_voting_benchmarks(query_map, n_count):
    """
    Computes four distinct selection strategies for scaling analysis.
    """
    total_queries = len(query_map)
    baseline_correct = 0
    best_of_n_correct = 0
    prm_maj_correct = 0
    weighted_maj_correct = 0

    for q_id, traces in query_map.items():
        subset = traces[:n_count]
        ground_truth = subset[0]['ground_truth'].lower()
        
        # --- 1. Baseline: Standard Majority Vote ---
        all_answers = [t['answer'] for t in subset if t['answer']]
        if all_answers:
            if Counter(all_answers).most_common(1)[0][0].lower() == ground_truth:
                baseline_correct += 1

        # --- 2. Best-of-N (Greedy Selection) ---
        # Picks the single trace with the highest continuous confidence score
        best_trace = max(subset, key=lambda x: x['prob_1'])
        if best_trace['answer'] and best_trace['answer'].lower() == ground_truth:
            best_of_n_correct += 1

        # --- 3. PRM Maj: Hard-Filtered Vote (Reverted to Old Logic) ---
        # Uses the binary 'predicted_label' (0/1) for filtering
        approved = [t['answer'] for t in subset if t['prm_approved'] == 1 and t['answer']]
        # FALLBACK: If PRM rejected all, use all N traces for the vote
        final_pool = approved if approved else all_answers
        if final_pool:
            if Counter(final_pool).most_common(1)[0][0].lower() == ground_truth:
                prm_maj_correct += 1

        # --- 4. Confidence-Weighted Majority Vote ---
        # Each answer's weight is the sum of its continuous prob_1 scores
        weight_map = defaultdict(float)
        for t in subset:
            if t['answer']:
                weight_map[t['answer'].lower()] += t['prob_1']
        
        if weight_map:
            weighted_voted = max(weight_map, key=weight_map.get)
            if weighted_voted == ground_truth:
                weighted_maj_correct += 1
           
    return {
        'baseline': (baseline_correct / total_queries) * 100,
        'best_of_n': (best_of_n_correct / total_queries) * 100,
        'prm_maj': (prm_maj_correct / total_queries) * 100,
        'weighted_maj': (weighted_maj_correct / total_queries) * 100
    }

def main():
    args = parse_args()
    raw_data = []
    
    with open(args.evaluations_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
        # Try JSON array first, fall back to JSONL
        if content.startswith('['):
            raw_data = json.loads(content)
        else:
            for line in content.split('\n'):
                if line.strip():
                    raw_data.append(json.loads(line))
   
    query_map = defaultdict(list)
    for item in raw_data:
        meta = item.get('metadata', {})
        q_id = meta.get('query_id')
        if q_id is None: continue
       
        query_map[q_id].append({
            'response_id': meta.get('response_id', 0),
            'prob_1': item.get('prob_1', 0.0), # Continuous score
            'prm_approved': item.get('predicted_label', 0), # Reverted to binary label
            'answer': meta.get('predicted_answer'),
            'ground_truth': meta.get('ground_truth')
        })

    for q_id in query_map:
        query_map[q_id].sort(key=lambda x: x['response_id'])

    n_values = [1, 2, 4, 8, 16, 32]
    header = f"{'N':<4} | {'Baseline':<10} | {'Best-of-N':<10} | {'PRM Maj':<10} | {'Weighted':<10}"
    print(f"\n{'='*65}\n{header}\n{'-'*65}")
   
    for n in n_values:
        res = run_voting_benchmarks(query_map, n)
        print(f"{n:<4} | {res['baseline']:>8.2f}% | {res['best_of_n']:>8.2f}% | {res['prm_maj']:>8.2f}% | {res['weighted_maj']:>8.2f}%")
    print(f"{'='*65}\n")

if __name__ == "__main__":
    main()