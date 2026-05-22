from __future__ import annotations

import itertools
from dataclasses import dataclass, field
import threading
from typing import Dict, List, Sequence

from .config import MethodConfig
from .roles import Evaluator, Judge
from .prompts import attribute_prompt, ATTRIBUTE_PANELS
from .utils import UnionFind, median, shannon_entropy


TAG_SCORES = {"Answerable": 0.0, "Partial": 0.5, "Unknown": 1.0}
ENTAIL_SCORES = {"Entailed": 0.0, "NotEntailed": 0.7, "Contradicted": 1.0}
RELATION_MAP = {"Entails": "entails", "Contradicts": "contradicts", "Unrelated": "unrelated"}
ATTR_VALUES = ["Yes", "No", "Uncertain"]


@dataclass
class GapProxyController:
    alpha: float
    min_calibration: int
    records: List[tuple[float, float]] = field(default_factory=list)
    threshold: float | None = None

    def score(self, tag: str, entailment: str) -> float:
        tag_score = TAG_SCORES.get(tag, 0.5)
        entail_score = ENTAIL_SCORES.get(entailment, 0.7)
        return 0.5 * (tag_score + entail_score)

    def observe(self, proxy_score: float, gap_value: float) -> None:
        self.records.append((proxy_score, gap_value))
        if self.threshold is None and len(self.records) >= self.min_calibration:
            self.threshold = self._fit_threshold()

    def _fit_threshold(self) -> float:
        # Choose the smallest threshold that yields false-skip rate <= alpha.
        candidates = sorted({proxy for proxy, _ in self.records}, reverse=True)
        for tau in candidates:
            flagged = [(proxy, gap) for proxy, gap in self.records if proxy >= tau]
            if not flagged:
                continue
            false_skips = sum(1 for _, gap in flagged if gap < 0.66)
            rate = false_skips / len(flagged)
            if rate <= self.alpha:
                return tau
        return max(candidates) if candidates else 0.9

    def should_skip(self, proxy_score: float) -> bool:
        if self.threshold is None:
            return False
        return proxy_score >= self.threshold


@dataclass
class CuriosityIndices:
    evaluator: Evaluator
    judge: Judge
    config: MethodConfig
    logger: any  # StructuredLogger protocol
    target_dimension: str
    gap_controller: GapProxyController = field(init=False)
    baseline_entropy: float | None = None
    baseline_known_frac: float | None = None
    round_context: str = ""

    def __post_init__(self) -> None:
        self.gap_controller = GapProxyController(
            alpha=self.config.alpha_skip,
            min_calibration=self.config.calibration_min_questions,
        )
        self._lock = threading.Lock()

    def set_round_context(self, context: str) -> None:
        """Compute and cache baseline attribute entropy for the round."""
        self.round_context = context
        ent, samples = self._attribute_entropy(
            context,
            repeats=self.config.attr_panel_baseline,
            label="baseline",
            mode="baseline",
            return_samples=True,
        )
        self.baseline_entropy = ent
        self.baseline_known_frac = self._known_fraction(samples)
        self.logger.log(
            "baseline_entropy",
            target=self.target_dimension,
            entropy=self.baseline_entropy,
        )

    def evaluate_question(self, question: str, snippets: Sequence[str], *, include_clarity: bool = False) -> dict:
        total_samples = max(
            self.config.gap_attempts,
            self.config.disagr_answers,
            self.config.clarity_answers,
        )
        answers: List[dict] = []
        for _ in range(total_samples):
            answers.append(self.evaluator.answer(question, snippets))

        proxy = self.judge.gap_proxy(question, snippets)
        proxy_score = self.gap_controller.score(proxy["tag"], proxy["entailment"])
        with self._lock:
            skip = self.gap_controller.should_skip(proxy_score)
        if skip:
            gap_score = 1.0
        else:
            gap_score = self._gap_score(answers[: self.config.gap_attempts])
            with self._lock:
                self.gap_controller.observe(proxy_score, gap_score)

        disagr_score = self._disagreement(answers[: self.config.disagr_answers])

        clarity_score = None
        if include_clarity:
            clarity_score = self._clarity_gain(question, snippets, answers[: self.config.clarity_answers])

        details = {
            "answers": answers,
            "proxy": proxy,
            "proxy_score": proxy_score,
            "gap": gap_score,
            "disagreement": disagr_score,
            "clarity": clarity_score,
        }
        return details

    def compute_clarity(self, question: str, snippets: Sequence[str], clarity_ideas: Sequence[dict]) -> float:
        return self._clarity_gain(question, snippets, clarity_ideas[: self.config.clarity_answers])

    def _gap_score(self, answers: Sequence[dict]) -> float:
        counts = {"Unknown": 0, "Partial": 0, "Answerable": 0}
        for ans in answers:
            counts[ans.get("tag", "Unknown")] = counts.get(ans.get("tag", "Unknown"), 0) + 1
        total = sum(counts.values()) or 1
        pi_unknown = counts.get("Unknown", 0) / total
        pi_partial = counts.get("Partial", 0) / total
        return pi_unknown + 0.5 * pi_partial

    def _disagreement(self, answers: Sequence[dict]) -> float:
        texts = [ans.get("answer", "") for ans in answers if ans.get("answer")]
        if len(texts) < 2:
            return 0.0
        n = len(texts)
        uf = UnionFind.with_size(n)
        contradictory_pairs = 0
        for i, j in itertools.combinations(range(n), 2):
            relation = self.judge.pairwise_relation(texts[i], texts[j])
            if relation == "Entails":
                uf.union(i, j)
            elif relation == "Contradicts":
                contradictory_pairs += 1
        cluster_sizes: Dict[int, int] = {}
        for idx in range(n):
            root = uf.find(idx)
            cluster_sizes[root] = cluster_sizes.get(root, 0) + 1
        probs = [size / n for size in cluster_sizes.values()]
        entropy = shannon_entropy(probs)
        correction = (len(cluster_sizes) - 1) / (2 * n)
        entropy += correction
        # Boost disagreement slightly if contradictions exist.
        if contradictory_pairs:
            entropy += min(0.4, contradictory_pairs / n)
        return entropy

    def _clarity_gain(self, question: str, snippets: Sequence[str], clarity_ideas: Sequence[dict]) -> float:
        if self.baseline_entropy is None:
            # Fall back to computing on the fly using snippets only.
            context = "\n".join(snippets)
            ent, samples = self._attribute_entropy(
                context,
                repeats=self.config.attr_panel_baseline,
                label="baseline_fallback",
                mode="baseline",
                return_samples=True,
            )
            self.baseline_entropy = ent
            self.baseline_known_frac = self._known_fraction(samples)
        entropies_after: List[float] = []
        known_fracs: List[float] = []
        for idea in clarity_ideas:
            snippet_block = "\n".join(f"- {s}" for s in snippets)
            raw_levers = idea.get("levers", [])
            allowed = set(ATTRIBUTE_PANELS[self.target_dimension].keys())
            levers_filtered = [lv for lv in raw_levers if lv in allowed]
            levers = ", ".join(levers_filtered)
            context = (
                "SECTION: CORPUS_SNIPPETS\n"
                + snippet_block
                + "\n\nSECTION: QUESTION\n"
                + question
                + "\n\nSECTION: IDEA\n"
                + idea.get("idea", "Unknown")
                + f"\nGROUNDING: {idea.get('grounding', 'Mixed')}"
                + (f"\nCLAIMED_LEVERS: {levers}" if levers else "")
                + "\n\nSECTION: LEVER_DEFINITIONS\n"
                + "\n".join(f"{k}: {v}" for k, v in ATTRIBUTE_PANELS[self.target_dimension].items())
            )
            ent, samples = self._attribute_entropy(
                context,
                repeats=self.config.attr_panel_clarity,
                label="qa_panel",
                question=question,
                mode="qa",
                claimed_levers=levers_filtered,
                return_samples=True,
            )
            entropies_after.append(ent)
            known_fracs.append(self._known_fraction(samples))
        if not entropies_after:
            return 0.0
        avg_entropy_after = sum(entropies_after) / len(entropies_after)
        baseline_known = self.baseline_known_frac if self.baseline_known_frac is not None else 0.0
        avg_known_after = sum(known_fracs) / len(known_fracs) if known_fracs else 0.0
        lever_gain = max(0.0, avg_known_after - baseline_known)
        return max(0.0, (self.baseline_entropy - avg_entropy_after) + 0.3 * lever_gain)

    def _known_fraction(self, panel_samples: Sequence[dict]) -> float:
        if not panel_samples:
            return 0.0
        total = 0
        known = 0
        for sample in panel_samples:
            for val in sample.values():
                total += 1
                if val != "Uncertain":
                    known += 1
        return known / total if total else 0.0

    def _attribute_entropy(self, context: str, repeats: int, *, label: str, question: str | None = None, mode: str = "baseline", claimed_levers: Sequence[str] | None = None, return_samples: bool = False) -> float | tuple[float, list[dict]]:
        repeats = max(1, repeats)
        panel_samples: List[dict] = []
        for _ in range(repeats):
            sample = self.judge.attribute_panel(self.target_dimension, context, mode=mode)
            if mode == "qa" and claimed_levers:
                for lever in claimed_levers:
                    if lever in sample and sample[lever] == "Uncertain":
                        sample[lever] = "Yes"
            panel_samples.append(sample)
        if not panel_samples or not panel_samples[0]:
            return (0.0, panel_samples) if return_samples else 0.0
        if self.logger:
            self.logger.log(
                "attribute_panel",
                label=label,
                question=question,
                samples=panel_samples,
            )
        attr_names = panel_samples[0].keys()
        entropies: List[float] = []
        for name in attr_names:
            counts = {val: 0.0 for val in ATTR_VALUES}
            for sample in panel_samples:
                token = sample.get(name, "Uncertain")
                counts[token] = counts.get(token, 0.0) + 1.0
            # Moderate smoothing (Laplace) so Uncertain-heavy baselines carry high entropy and flips matter.
            prior = 0.2
            total = sum(counts.values()) + prior * len(ATTR_VALUES)
            probs = [(counts[val] + prior) / total for val in ATTR_VALUES]
            entropies.append(shannon_entropy(probs))
        entropy = sum(entropies) / max(1, len(entropies))
        if return_samples:
            return entropy, panel_samples
        return entropy
