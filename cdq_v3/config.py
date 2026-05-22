from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(slots=True)
class RetrievalConfig:
    """Controls lightweight BM25 retrieval and context construction."""

    top_k_docs: int = 8
    top_k_sentences: int = 6
    k1: float = 1.6
    b: float = 0.75
    reference_only: bool = True
    snippet_sentences: int = 4
    snippet_char_limit: int = 1400


@dataclass(slots=True)
class LLMConfig:
    """Provider + cache configuration."""

    provider: str = "openai"
    model: str = "gpt-5.1-mini"
    temperature: float = 0.2
    max_tokens: int = 700
    timeout: int = 60
    organization: str | None = None
    cache_path: Path = Path("cdq_v3/out/prompt_cache.sqlite")


@dataclass(slots=True)
class MethodConfig:
    """Top-level configuration of the curiosity optimizer."""

    dataset_path: Path = Path("datasets/data_rag_500.csv")
    env_path: Path = Path(".env")
    output_dir: Path = Path("cdq_v3/out")
    run_name: str | None = None
    rounds: int = 3
    per_round_candidates: int = 15
    survivors: int = 6
    generator_max_tokens: int = 2048
    max_rounds: int = 10
    target_dimension: str = "Novelty"  # or "Feas"
    weights: Tuple[float, float, float] = (0.2, 0.3, 0.5)
    eta: float = 1.0
    gap_attempts: int = 3
    disagr_answers: int = 5
    clarity_answers: int = 5
    attr_panel_baseline: int = 2
    attr_panel_clarity: int = 3
    clarity_cap: float = 1.2
    alpha_skip: float = 0.1
    calibration_min_questions: int = 25
    max_workers: int = 10
    topic_limit: int | None = None
    topic_ids: Tuple[str, ...] | None = None
    seed: int = 13
    resume: bool = False
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    judge_temperature: float = 0.7

    def resolved_rounds(self) -> int:
        return min(self.rounds, self.max_rounds)
