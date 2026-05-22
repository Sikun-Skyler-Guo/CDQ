#!/usr/bin/env python3
"""
Generate RAG output for papers using web search or simulated retrieval.
Supports resume functionality to skip already processed papers.
"""

from data_loader import load_initial_data
from prompt_manager import (
    get_context_rag_withoutq,
    get_context_rag_withq
)
from openai_client import OpenAIClient
import pandas as pd
import argparse
import threading
import concurrent.futures
from tqdm import tqdm
import traceback
import csv
import os

# Thread-local storage to keep one client per thread
_thread_local = threading.local()

# Global variables
_model_name = "gpt-4o"
_api_provider = "openai"

def get_thread_client():
    if not hasattr(_thread_local, "client"):
        _thread_local.client = OpenAIClient(model=_model_name, api_provider=_api_provider)
    return _thread_local.client

def main():
    parser = argparse.ArgumentParser(description="Generate RAG output for papers with Resume support")
    parser.add_argument("--max_workers", type=int, default=10, help="max parallel workers")
    parser.add_argument("--withQ", type=int, default=0, help="1: rag_output_q, 0: rag_output_noq")
    parser.add_argument("--dataset", type=str, required=True, help="path to input csv")
    parser.add_argument("--model", type=str, default="gpt-4o", help="model name")
    parser.add_argument("--api_provider", type=str, default="openai", choices=["openai", "deepinfra"])
    parser.add_argument("--limit", type=int, default=None, help="Limit total papers for debugging")
    parser.add_argument("--resume", action="store_true", help="Resume from last run (skip already processed rows)")
    
    args = parser.parse_args()
    
    withQ = args.withQ == 1
    dataset_file = args.dataset
    col_name = "rag_output_q" if withQ else "rag_output_noq"
    
    global _model_name, _api_provider
    _model_name = args.model
    _api_provider = args.api_provider

    print(f"Step 1: Loading initial data from {dataset_file}...")
    data = load_initial_data(withQ, dataset_file)
    papers = data["papers"]
    
    if args.limit is not None and args.limit > 0:
        papers = papers[:args.limit]
        print(f"Limited to {len(papers)} papers.")

    try:
        # Read CSV to check for resume points
        df = pd.read_csv(dataset_file)
        if "targetPaperId" not in df.columns:
            raise KeyError("dataset_file must contain 'targetPaperId' column")
        
        df_lock = threading.Lock()

        # Filter logic: if resume is enabled, skip already processed rows
        tasks = []
        skipped_count = 0
        
        for paper in papers:
            tid = paper.get("targetPaperId")
            if args.resume and col_name in df.columns:
                # Check if this ID has a value that is not an error
                mask = df["targetPaperId"] == tid
                if mask.any():
                    existing_val = df.loc[mask, col_name].iloc[0]
                    # If value is not empty and doesn't start with Error or Exception, consider it done
                    if pd.notna(existing_val) and str(existing_val).strip() != "" and \
                       not str(existing_val).startswith("Error") and \
                       not str(existing_val).startswith("Exception"):
                        skipped_count += 1
                        continue
            tasks.append(paper)

        if args.resume:
            print(f"Resume mode enabled: Skipped {skipped_count} already processed rows.")
        print(f"Remaining tasks to process: {len(tasks)}")
        print("-" * 50)

        if not tasks:
            print("No tasks left to process. Exiting.")
            return

        # Define worker function
        def worker(paper):
            target_id = paper.get("targetPaperId")
            try:
                client = get_thread_client()
                if withQ:
                    if not paper.get("questions"):
                        return target_id, f"Error: no questions found", False
                    rag_prompt_template = get_context_rag_withq()
                    rag_output = client.generate_context_with_rag(paper, rag_prompt_template, withQ=True)
                else:
                    rag_prompt_template = get_context_rag_withoutq()
                    rag_output = client.generate_context_with_rag(paper, rag_prompt_template, withQ=False)
                
                if not rag_output:
                    return target_id, "Error: empty output", False
                return target_id, rag_output, True

            except Exception as e:
                return target_id, f"Exception: {str(e)}", False

        # Execute in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_tid = {executor.submit(worker, p): p.get("targetPaperId") for p in tasks}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_tid), total=len(future_to_tid)):
                tid = future_to_tid[future]
                try:
                    target_id, result, success = future.result()
                except Exception as e:
                    target_id, result, success = tid, f"Unhandled thread error: {str(e)}", False

                # Thread-safe write
                with df_lock:
                    if col_name not in df.columns:
                        df[col_name] = ""
                    mask = df["targetPaperId"] == target_id
                    df.loc[mask, col_name] = result
                    # Save after each completion to ensure checkpoint information is persisted
                    df.to_csv(dataset_file, index=False, quoting=csv.QUOTE_NONNUMERIC)

                if not success:
                    print(f" [ERR] id={target_id}: {result[:50]}...")

        print(f"\nFinished! Results saved to {dataset_file}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()