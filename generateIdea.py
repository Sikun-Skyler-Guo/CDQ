#!/usr/bin/env python3
"""
Generate research ideas using single, RAG, or iterative methods.
Supports both with-question and without-question modes.
"""

from data_loader import load_initial_data
from prompt_manager import (
    get_idea_generation_prompt_template_rag,
    get_idea_generation_prompt_template_single,
    review_prompt,
    review_prompt_q,
    get_idea_generation_prompt_template_generate,
    get_idea_generation_prompt_template_single_questions,
    get_idea_generation_prompt_template_refine_questions
)
from openai_client import OpenAIClient
import pandas as pd
import json
import argparse
import os
import threading
import concurrent.futures
from tqdm import tqdm
import traceback

# thread-local storage to keep one client per thread
_thread_local = threading.local()
_model_name = None  # Global variable to store model name
_api_provider = "openai"  # Global variable to store API provider

def get_thread_client():
    if not hasattr(_thread_local, "client"):
        _thread_local.client = OpenAIClient(model=_model_name, api_provider=_api_provider)
    return _thread_local.client

def main():
    # arg parsing
    parser = argparse.ArgumentParser(description="This script accepts command line arguments.")
    parser.add_argument("--indicator", type=str, default="novelty", help="indicator")
    parser.add_argument("--type", type=str, default="single", help="which type (single|rag|iterative)")
    parser.add_argument("--max_workers", type=int, default=10, help="max parallel workers (threads)")
    parser.add_argument("--withQ", type=int, default=0, help="1: with question, 0: no question")
    parser.add_argument("--dataset", type=str, required=True, help="path to input csv")
    parser.add_argument("--model", type=str, default="gpt-4o", help="model name (e.g., gpt-4o, gpt-4o-mini, meta-llama/Llama-3.3-70B-Instruct-Turbo)")
    parser.add_argument("--api_provider", type=str, default="openai", choices=["openai", "deepinfra"], 
                        help="API provider to use: 'openai' or 'deepinfra' (default: openai)")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of papers to process (for debugging, e.g., --limit 2)")
    parser.add_argument("--resume", action="store_true", help="Resume from log file, skip already completed papers")
    args = parser.parse_args()
    
    withQ = args.withQ == 1
    dataset_file = args.dataset
    model_name = args.model
    
    # Set global model name and API provider for thread clients
    global _model_name, _api_provider
    _model_name = model_name
    _api_provider = args.api_provider

    # Generate log file path based on dataset and parameters
    log_dir = os.path.join(os.path.dirname(dataset_file), "logs")
    os.makedirs(log_dir, exist_ok=True)
    dataset_basename = os.path.splitext(os.path.basename(dataset_file))[0]
    log_filename = f"{dataset_basename}_{args.type}_{args.indicator}_{'withQ' if withQ else 'noQ'}_{model_name.replace('/', '_')}.log"
    log_file_path = os.path.join(log_dir, log_filename)
    print(f"Log file: {log_file_path}")
    
    # Load completed paper IDs if resuming
    completed_paper_ids = set()
    if args.resume:
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r') as f:
                completed_paper_ids = {line.strip() for line in f if line.strip()}
            print(f"Resume mode: Found {len(completed_paper_ids)} completed papers in log file")
        else:
            print(f"Resume mode: Log file not found, starting from scratch")
    else:
        print(f"Normal mode: Progress will be logged to {log_file_path}")

    # 1. Load initial data
    print("Step 1: Loading initial data...")
    data = load_initial_data(withQ, dataset_file)
    papers = data["papers"]
    
    # Filter out completed papers if resuming
    if args.resume and completed_paper_ids:
        original_count = len(papers)
        papers = [p for p in papers if p.get("targetPaperId") not in completed_paper_ids]
        print(f"Filtered out {original_count - len(papers)} completed papers, {len(papers)} remaining")
    
    # Limit papers if --limit is specified (for debugging)
    if args.limit is not None and args.limit > 0:
        papers = papers[:args.limit]
        print(f"Limited to {len(papers)} papers (--limit {args.limit})")
    
    print(f"Loaded {len(papers)} papers to process.")
    print("-" * 50)

    try:
        # 0. read CSV once
        df = pd.read_csv(dataset_file)
        if "targetPaperId" not in df.columns:
            raise KeyError("dataset_file must contain 'targetPaperId' column")
        
        # Filter df to only include papers we're processing (if limit is specified)
        if args.limit is not None and args.limit > 0:
            target_ids = [p.get("targetPaperId") for p in papers]
            df = df[df["targetPaperId"].isin(target_ids)].copy()
            print(f"Filtered CSV to {len(df)} rows matching the limited papers.")

        df_lock = threading.Lock()
        log_lock = threading.Lock()
        
        # Track completion status for each paper: {paper_id: set of completed task indices}
        paper_completion = {p.get("targetPaperId"): set() for p in papers}

        # --- WORKER FOR NO QUESTION (Baseline) ---
        def process_task_noq(paper, i):
            target_id = paper.get("targetPaperId")
            col_name = f"{args.type}_{args.indicator}_{i+1}"
            try:
                client = get_thread_client()
                final_idea = None

                if args.type == "rag":
                    target_rows = df[df["targetPaperId"] == target_id]
                    if target_rows.empty:
                        return target_id, col_name, f"Error: no row for targetPaperId {target_id}", False
                    context_x1 = target_rows.iloc[0].get("rag_output_noq", "")
                    idea_prompt = get_idea_generation_prompt_template_rag()
                    final_idea = client.generate_idea_rag(paper, context_x1, idea_prompt, args.indicator)

                elif args.type == "single":
                    idea_prompt = get_idea_generation_prompt_template_single()
                    final_idea = client.generate_idea_withoutQ_single(paper, idea_prompt, args.indicator)

                else: # Iterative (NoQ)
                    first_generator_prompt = get_idea_generation_prompt_template_single()
                    previous_idea = client.generate_idea_withoutQ_single(paper, first_generator_prompt, args.indicator)
                    
                    reviewer_prompt = review_prompt()
                    review = client.generate_review_noq(previous_idea, reviewer_prompt, args.indicator)
                    
                    generator_prompt = get_idea_generation_prompt_template_generate()
                    
                    for x in range(2):
                        # Standard generator without Question context
                        idea = client.generate_idea_withoutQ_generator(previous_idea, review, generator_prompt, args.indicator)
                        review = client.generate_review_noq(idea, reviewer_prompt, args.indicator)
                        previous_idea = idea
                    final_idea = idea

                if not final_idea or final_idea == "":
                    return target_id, col_name, f"Error: generation returned empty", False
                return target_id, col_name, final_idea, True

            except Exception as e:
                tb = traceback.format_exc()
                return target_id, col_name, f"Exception: {str(e)}\n{tb}", False

        # --- WORKER FOR WITH QUESTION (Your Method) ---
        def process_task(paper, i):
            target_id = paper.get("targetPaperId")
            col_name = f"question_{args.type}_{args.indicator}_{i+1}"
            try:
                client = get_thread_client()
                final_idea = None

                if args.type == "rag":
                    target_rows = df[df["targetPaperId"] == target_id]
                    if target_rows.empty:
                        return target_id, col_name, f"Error: no row for targetPaperId {target_id}", False
                    context_x1 = target_rows.iloc[0].get("rag_output_q", "")
                    idea_prompt = get_idea_generation_prompt_template_rag()
                    final_idea = client.generate_idea_rag(paper, context_x1, idea_prompt, args.indicator)

                elif args.type == "single":
                    idea_prompt = get_idea_generation_prompt_template_single_questions()
                    final_idea = client.generate_idea_withQ_single(paper, idea_prompt, args.indicator)

                else:  # Iterative (With Question)
                    # Step 1: Generate initial idea
                    first_generator_prompt = get_idea_generation_prompt_template_single_questions()
                    previous_idea = client.generate_idea_withQ_single(paper, first_generator_prompt, args.indicator)
                    
                    # Step 2: Review with question context
                    reviewer_prompt = review_prompt_q()
                    review = client.generate_review_q(paper, previous_idea, reviewer_prompt, args.indicator)
                    
                    # Step 3: Refinement loop (2 iterations)
                    generator_prompt = get_idea_generation_prompt_template_refine_questions()
                    idea = previous_idea
                    for x in range(2):
                        idea = client.generate_idea_withQ_generator(paper, previous_idea, review, generator_prompt, args.indicator)
                        review = client.generate_review_q(paper, idea, reviewer_prompt, args.indicator)
                        previous_idea = idea
                    final_idea = idea

                if not final_idea:
                    return target_id, col_name, f"Error: generation returned empty", False
                return target_id, col_name, final_idea, True

            except Exception as e:
                tb = traceback.format_exc()
                return target_id, col_name, f"Exception: {str(e)}\n{tb}", False

        # Prepare tasks
        tasks = []
        for paper in papers:
            for i in range(3):
                tasks.append((paper, i))

        # Run tasks concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            if withQ:
                future_to_task = {executor.submit(process_task, paper, i): (paper.get("targetPaperId"), i) for (paper, i) in tasks}
            else:
                future_to_task = {executor.submit(process_task_noq, paper, i): (paper.get("targetPaperId"), i) for (paper, i) in tasks}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(future_to_task)):
                target_id, i = future_to_task[future]
                try:
                    tid, col_name, result, success = future.result()
                except Exception as e:
                    tid, col_name, result, success = target_id, f"ERR", f"Unhandled exception: {str(e)}", False

                # Write to CSV safely
                with df_lock:
                    if col_name not in df.columns:
                        df[col_name] = ""
                    mask = df["targetPaperId"] == tid
                    if mask.any():
                        try:
                            df.loc[mask, col_name] = json.dumps(result)
                        except:
                            df.loc[mask, col_name] = str(result)
                    else:
                        print(f"Warning: no df row for {tid}")

                    df.to_csv(dataset_file, index=False, quotechar='"')
                
                # Update completion status and write to log if all tasks for this paper are done
                if success:
                    with log_lock:
                        if tid in paper_completion:
                            paper_completion[tid].add(i)
                            # Check if all 3 tasks are completed for this paper
                            if len(paper_completion[tid]) == 3:
                                # Append paper ID to log file
                                with open(log_file_path, 'a') as f:
                                    f.write(f"{tid}\n")
                                print(f"[COMPLETED] All tasks finished for paper id={tid}")

                print(f"[{'OK' if success else 'ERR'}] {col_name} for id={tid}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()

