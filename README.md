# Curiosity-Driven Questioning for Engine-Agnostic LLM Research Ideation
This is the official repo for Curiosity-Driven Questioning for Engine-Agnostic LLM Research Ideation.

## Overview
This repository contains two connected pipelines:

- Curiosity-driven question generation (`cdq_v3/`).
- Research ideation baselines (Single, RAG, Iterative) plus evaluation scripts.

## Project Layout

```
.
├── cdq_v3/                   # Curiosity-driven question generation
├── datasets/
│   └── data_rag_500.csv      # Input dataset (500 topics)
├── generateRagOutput.py      # Step 1: Generate RAG context
├── generateIdea.py           # Step 2: Generate research ideas
├── winrate.py                # Step 3: Compare winrates (withQ vs noQ)
├── data_loader.py            # CSV loading utilities
├── prompt_manager.py         # Prompt templates
├── openai_client.py          # API client wrapper
├── config.py                 # Environment-variable API key shim for baseline scripts
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

```bash
pip install -r requirements.txt
```

## API Keys

### For cdq_v3 (question generation)
Create a `.env` file in the repo root:

```
OPENAI_API_KEY=your_openai_key
# Optional:
DEEPINFRA_API_KEY=your_deepinfra_key
```

### For baseline idea generation
Set API keys in your shell before running the baseline scripts:

```bash
export OPENAI_API_KEY=your_openai_key
export DEEPINFRA_API_KEY=your_deepinfra_key  # optional
```

Keep API keys out of version control.

## Workflow

### Step 0: Generate curiosity-driven questions (cdq_v3)

```bash
python -m cdq_v3.cli \
  --dataset datasets/data_rag_500.csv \
  --env .env \
  --target Novelty \
  --rounds 5 \
  --per-round 30 \
  --survivors 15 \
  --model gpt-4o
```

Outputs are written under `cdq_v3/out/<run_name>/`, including:
- `best_questions.csv`: top questions per topic
- `events.jsonl`: full run logs
- `prompt_cache.sqlite`: sqlite prompt cache

To use the questions for ideation, merge `best_questions.csv` back into your dataset
(e.g., attach the top question(s) to the `questions` column keyed by `topic_id`/`targetPaperId`).

### Step 1: Generate RAG context (optional, only for RAG baseline)

```bash
python generateRagOutput.py \
  --dataset datasets/data_rag_500.csv \
  --withQ 1 \
  --model gpt-4o \
  --api_provider openai \
  --max_workers 10
```

This adds `rag_output_q` or `rag_output_noq` columns to the CSV.

### Step 2: Generate research ideas

```bash
python generateIdea.py \
  --dataset datasets/data_rag_500.csv \
  --type single \
  --indicator novelty \
  --withQ 1 \
  --model gpt-4o \
  --api_provider openai \
  --max_workers 10
```

Supported types: `single`, `rag`, `iterative`.

### Step 3: Evaluate winrates

```bash
python winrate.py \
  --dataset datasets/data_rag_500.csv \
  --type single \
  --indicator novelty \
  --model gpt-4o \
  --api_provider openai \
  --max_workers 8
```

## Data Format

The dataset should include the following columns:

- `targetPaperId`
- `target_title`
- `target_abs`
- `ref_titles` (list-like string)
- `ref_abs` (list-like string)
- `questions` (optional; list-like string)

`datasets/data_rag_500.csv` already includes these columns. For ideation scripts, `questions` is required
when running with `--withQ 1`.

## Notes

- Use `--resume` on long-running scripts to skip completed papers.
- `--max_workers` controls parallelism.
- The included `.gitignore` keeps local API secrets, run outputs, caches, and Python artifacts out of version control while tracking the release dataset.
- See `cdq_v3/README.md` for implementation details of the curiosity optimizer.
