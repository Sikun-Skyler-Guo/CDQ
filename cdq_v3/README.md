cdq_v3
======

Curiosity-driven question generation via curiosity optimization. Key features:

- Loads topic-specific corpora from `datasets/data_rag_500.csv` by default.
- Uses OpenAI's GPT-5 family (default `gpt-5.1-mini`) through the official SDK plus a sqlite prompt cache.
- Provides deterministic mock placeholders for non-OpenAI engines (Anthropic/Groq) so future integration only needs a drop-in LLM class.
- Implements generator/evaluator/judge roles with structured prompts, BM25-style retrieval, the three curiosity indices, and an exponentiated-gradient policy update.
- Parallelizes per-question scoring via `multiprocessing.pool.ThreadPool`.
- Logs every step as JSONL under `cdq_v3/out/`.

Running
-------

```bash
python -m cdq_v3.cli \
  --dataset datasets/data_rag_500.csv \
  --env .env \
  --target Novelty \
  --rounds 3 \
  --per-round 15 \
  --survivors 6 \
  --model gpt-5.1-mini
```

Ensure `.env` exports `OPENAI_API_KEY`. Logs (including cached prompts) land in `cdq_v3/out/`.
