from __future__ import annotations

import argparse
from pathlib import Path

from .config import MethodConfig
from .method import CuriosityOptimizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run cdq_v3 curiosity optimizer.")
    defaults = MethodConfig()
    parser.add_argument("--dataset", type=Path, default=defaults.dataset_path, help="Path to data_rag_500.csv")
    parser.add_argument("--env", type=Path, default=defaults.env_path, help=".env file with OPENAI_API_KEY")
    parser.add_argument("--rounds", type=int, default=defaults.rounds, help="Rounds per topic")
    parser.add_argument("--per-round", type=int, default=defaults.per_round_candidates, help="Candidates per round")
    parser.add_argument("--survivors", type=int, default=defaults.survivors, help="Survivors before clarity eval")
    parser.add_argument(
        "--generator-max-tokens",
        type=int,
        default=defaults.generator_max_tokens,
        help="Max output tokens for the question generator",
    )
    parser.add_argument("--target", choices=["Novelty", "Feas"], default=defaults.target_dimension, help="Target dimension")
    parser.add_argument("--provider", type=str, default=defaults.llm.provider, help="LLM provider (openai|deepinfra)")
    parser.add_argument("--model", type=str, default=defaults.llm.model, help="OpenAI model id (e.g., gpt-5.1-mini)")
    parser.add_argument(
        "--timeout",
        type=int,
        default=defaults.llm.timeout,
        help="Timeout in seconds for each LLM request",
    )
    parser.add_argument("--max-workers", type=int, default=defaults.max_workers, help="Thread workers for parallel eval")
    parser.add_argument("--topics", type=int, default=None, help="Limit number of topics from the dataset")
    parser.add_argument(
        "--topic-id",
        action="append",
        default=None,
        help="Run only specific topic_id (can be repeated or comma-separated).",
    )
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir, help="Folder for logs and caches")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name; results saved under out/<run_name>")
    parser.add_argument("--resume", action="store_true", help="Resume from existing run folder and skip completed topics")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    defaults = MethodConfig()
    config = MethodConfig(
        dataset_path=args.dataset,
        env_path=args.env,
        rounds=args.rounds,
        per_round_candidates=args.per_round,
        survivors=args.survivors,
        generator_max_tokens=args.generator_max_tokens,
        target_dimension=args.target,
        max_workers=args.max_workers,
        topic_limit=args.topics,
        output_dir=args.output_dir,
        run_name=args.run_name,
        resume=args.resume,
    )
    if args.topic_id:
        topic_ids = []
        for raw in args.topic_id:
            topic_ids.extend([part.strip() for part in raw.split(",") if part.strip()])
        config.topic_ids = tuple(sorted(set(topic_ids)))
    config.llm.provider = args.provider
    config.llm.model = args.model
    config.llm.timeout = args.timeout
    if args.provider.lower() == "deepinfra" and args.timeout == defaults.llm.timeout:
        config.llm.timeout = 180
    optimizer = CuriosityOptimizer(config)
    optimizer.run()


if __name__ == "__main__":
    main()
