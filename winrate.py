#!/usr/bin/env python3
"""
Calculate win rates and confidence intervals for comparing research ideas.
Supports resume functionality via log files.
"""

import os
import json
import csv
import argparse
from scipy import stats
import threading
import traceback
import numpy as np
import pandas as pd
import concurrent.futures
from tqdm import tqdm

from data_loader import load_eval_data_winrate
from prompt_manager import get_winrate_eval_template_feasibility, get_winrate_eval_template_novelty
from openai_client import OpenAIClient

_thread_local = threading.local()
_model_name = ""
_api_provider = ""

def get_thread_client():
    """Get thread-local OpenAI client instance."""
    if not hasattr(_thread_local, "client"):
        _thread_local.client = OpenAIClient(model=_model_name, api_provider=_api_provider)
    return _thread_local.client

def wilson_confidence_interval(successes, total, confidence=0.95):
    """
    Calculate Wilson score confidence interval for binomial distribution.
    
    Args:
        successes: Number of successes (B wins)
        total: Total number of trials (excluding ties)
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        Tuple of (lower, upper) bounds
    """
    if total == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf((1 + confidence) / 2)
    p_hat = successes / total
    denominator = 1 + (z**2 / total)
    center = (p_hat + (z**2 / (2 * total))) / denominator
    margin = (z / denominator) * np.sqrt((p_hat * (1 - p_hat) / total) + (z**2 / (4 * total**2)))
    return (max(0.0, center - margin), min(1.0, center + margin))

def calculate_ci_from_winners_list(winners_list):
    """
    Calculate confidence interval from list of winners.
    
    Args:
        winners_list: List containing "A", "B", "Tie", or None
    
    Returns:
        JSON string of confidence interval [lower, upper]
    """
    if not winners_list:
        return json.dumps([0.0, 0.0])
    # Count wins for B and A
    b_wins = sum(1 for w in winners_list if str(w).strip().upper() == "B")
    a_wins = sum(1 for w in winners_list if str(w).strip().upper() == "A")
    total = b_wins + a_wins
    if total == 0:
        return json.dumps([0.0, 0.0])
    lower, upper = wilson_confidence_interval(b_wins, total)
    return json.dumps([round(lower, 4), round(upper, 4)])

def main():
    parser = argparse.ArgumentParser(description="Calculate win rates and confidence intervals")
    parser.add_argument("--indicator", type=str, default="novelty", help="novelty or feasibility")
    parser.add_argument("--type", type=str, default="single", help="single, rag, or iterative")
    parser.add_argument("--resume", action="store_true", help="Resume from log file checkpoint")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of concurrent threads")
    parser.add_argument("--dataset", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Judge model name")
    parser.add_argument("--api_provider", type=str, default="openai", help="API provider (openai or deepinfra)")
    args = parser.parse_args()
    
    global _model_name, _api_provider
    _model_name = args.model
    _api_provider = args.api_provider

    # Configure log file path
    log_dir = os.path.join(os.path.dirname(args.dataset), "logs")
    os.makedirs(log_dir, exist_ok=True)
    dataset_basename = os.path.splitext(os.path.basename(args.dataset))[0]
    
    # Sanitize model name to prevent file creation failures
    safe_model_name = "".join([c if c.isalnum() else "_" for c in _model_name])
    log_filename = f"winrate_log_{dataset_basename}_{args.type}_{args.indicator}_{safe_model_name}.log"
    log_file_path = os.path.join(log_dir, log_filename)

    # Load completed IDs for resume functionality
    completed_ids = set()
    if args.resume and os.path.exists(log_file_path):
        with open(log_file_path, 'r') as f:
            completed_ids = {line.strip() for line in f if line.strip()}
        print(f"▶ Resume Mode: Found {len(completed_ids)} completed tasks in log.")

    # Load and filter data
    data = load_eval_data_winrate(args, args.dataset)
    all_papers = data["papers"]
    
    tasks = [p for p in all_papers if str(p.get("targetPaperId")) not in completed_ids]
    print(f"▶ Total: {len(all_papers)} | Skipped: {len(all_papers)-len(tasks)} | To Process: {len(tasks)}")

    if not tasks:
        print("✅ All tasks completed.")
        return

    # Read DataFrame for saving results
    df = pd.read_csv(args.dataset)
    
    # Define column names
    col_name_winrate = f"winrate_{args.type}_{args.indicator}" 
    col_name_winner = f"winner_{args.type}_{args.indicator}"
    col_name_winners_list = f"winrate_winners_{args.type}_{args.indicator}"
    col_name_ci = f"confidence_interval_{args.type}_{args.indicator}"

    df_lock = threading.Lock()
    log_lock = threading.Lock()

    def process_task(paper_record):
        """Process a single paper evaluation task."""
        tid = paper_record.get("targetPaperId")
        try:
            client = get_thread_client()
            # Get appropriate prompt template
            if args.indicator == "novelty":
                eval_prompt = get_winrate_eval_template_novelty()
            else:
                eval_prompt = get_winrate_eval_template_feasibility()
            
            # Call API for evaluation
            # Returns: score (win rate), winner (final winner A/B), winners_list (list of winners per iteration)
            score, winner, winners_list = client.generate_winrate(paper_record, eval_prompt, args.indicator)
            
            if winner is None:
                return tid, None, False, None, [], "[]"
                
            ci_str = calculate_ci_from_winners_list(winners_list)
            return tid, score, True, winner, winners_list, ci_str
        except Exception as e:
            print(f"Error processing {tid}: {str(e)}")
            return tid, str(e), False, None, [], "[]"

    # Parallel processing with multithreading
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_tid = {executor.submit(process_task, p): p.get("targetPaperId") for p in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_tid), total=len(future_to_tid), desc="Evaluating"):
            tid, res, success, winner, w_list, ci = future.result()

            if success:
                with df_lock:
                    # Ensure columns exist
                    for c in [col_name_winrate, col_name_winner, col_name_winners_list, col_name_ci]:
                        if c not in df.columns:
                            df[c] = None
                    
                    mask = df["targetPaperId"].astype(str) == str(tid)
                    if mask.any():
                        df.loc[mask, col_name_winrate] = json.dumps(res)
                        df.loc[mask, col_name_winner] = json.dumps(winner)
                        df.loc[mask, col_name_winners_list] = json.dumps(w_list)
                        df.loc[mask, col_name_ci] = ci
                        
                        # Save with all fields quoted to prevent encoding issues
                        df.to_csv(args.dataset, index=False, quoting=csv.QUOTE_ALL)
                
                with log_lock:
                    with open(log_file_path, 'a') as f:
                        f.write(f"{tid}\n")

if __name__ == "__main__":
    main()