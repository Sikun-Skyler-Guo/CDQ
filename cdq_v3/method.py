from __future__ import annotations

import math
import json
from dataclasses import dataclass
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
import csv
import threading

from .cache import PromptCache
from .config import MethodConfig
from .corpus import CorpusLoader, RetrievalResult, TopicCorpus
from .env import load_env_file
from .indices import CuriosityIndices
from .llm import CachedLLM, LLM, MockAnthropicLLM, MockGroqLLM, OpenAIChatLLM
from .logger import StructuredLogger
from .policies import Policy, default_policies
from .prompts import ATTRIBUTE_PANELS
from .roles import Evaluator, Generator, Judge
from .utils import min_max_normalize, split_sentences, seed_everything, median, normalize_whitespace


@dataclass
class CandidateQuestion:
    text: str
    policy: Policy
    round_idx: int


class CuriosityOptimizer:
    """End-to-end driver for the curiosity optimization method."""

    def __init__(self, config: MethodConfig, *, logger: StructuredLogger | None = None):
        self.config = config
        self.pkg_root = Path(__file__).resolve().parent.parent
        self.base_output_dir = self._root_join(Path(self.config.output_dir))
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_name = self.config.run_name or f"run_{timestamp}"
        self.run_dir = self.base_output_dir / self.run_name
        if self.run_dir.exists() and not self.config.resume:
            shutil.rmtree(self.run_dir)
        self.logger = logger or StructuredLogger(self.run_dir, filename="events.jsonl")
        seed_everything(self.config.seed)
        load_env_file(Path(self.config.env_path))
        cache_path = self._root_join(Path(self.config.llm.cache_path))
        self.cache = PromptCache(cache_path)
        self.non_openai_placeholders = {
            "anthropic": MockAnthropicLLM(),
            "groq": MockGroqLLM(),
        }
        self.base_llm = self._build_llm()
        self.cached_llm: LLM = CachedLLM(self.base_llm, self.cache)
        self.generator = Generator(self.cached_llm, logger=self.logger)
        self.evaluator = Evaluator(self.cached_llm)
        self.judge = Judge(self.cached_llm, config=self.config)
        self.indices = CuriosityIndices(
            evaluator=self.evaluator,
            judge=self.judge,
            config=self.config,
            logger=self.logger,
            target_dimension=self.config.target_dimension,
        )
        self.policies = default_policies()
        self.best_records: List[dict] = []
        self._best_lock = threading.Lock()

    def _build_llm(self) -> LLM:
        provider = self.config.llm.provider.lower()
        if provider == "deepinfra":
            return OpenAIChatLLM(
                model=self.config.llm.model,
                api_key_env="DEEPINFRA_API_KEY",
                base_url="https://api.deepinfra.com/v1/openai",
                timeout=self.config.llm.timeout,
            )
        if provider != "openai":
            # Placeholder path (uses deterministic mocks). Real integration would swap in real SDK clients.
            return self.non_openai_placeholders.get(provider, MockGroqLLM())
        return OpenAIChatLLM(
            model=self.config.llm.model,
            organization=self.config.llm.organization,
            timeout=self.config.llm.timeout,
        )

    def run(self) -> None:
        dataset_path = self._resolve_dataset_path()
        self.logger.log(
            "run_initialized",
            run_name=self.run_name,
            dataset=str(dataset_path),
            target=self.config.target_dimension,
            model=self.config.llm.model,
        )
        loader = CorpusLoader(dataset_path)
        corpora = loader.load(limit=self.config.topic_limit)
        if self.config.topic_ids:
            requested = set(self.config.topic_ids)
            corpora = [topic for topic in corpora if topic.topic_id in requested]
            missing = sorted(requested - {topic.topic_id for topic in corpora})
            if missing:
                self.logger.log("topic_id_missing", topic_ids=missing)
        run_corpora = corpora
        if self.config.resume:
            run_corpora = []
            for topic in corpora:
                topic_dir = self.run_dir / topic.topic_id
                if self._topic_completed(topic_dir, self.config.resolved_rounds()):
                    self.logger.log("topic_skipped", topic=topic.topic_id, reason="completed")
                    continue
                run_corpora.append(topic)
        self.logger.log(
            "start_run",
            topics=len(corpora),
            target=self.config.target_dimension,
            remaining=len(run_corpora),
            resume=self.config.resume,
        )
        futures = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            for topic in run_corpora:
                topic_dir = self.run_dir / topic.topic_id
                topic_logger = StructuredLogger(topic_dir, filename="events.jsonl")
                futures.append(pool.submit(self._run_topic, topic, topic_logger))
            for fut in as_completed(futures):
                fut.result()
        self._merge_topic_events(corpora)
        self._write_best_records()

    def _run_topic(self, topic: TopicCorpus, logger: StructuredLogger | None = None) -> None:
        local_logger = logger or self.logger
        retriever = topic.build_retriever(self.config.retrieval, references_only=self.config.retrieval.reference_only)
        base_snippets = self._seed_snippets(topic)
        self.indices.logger = local_logger
        self.generator.logger = local_logger
        self.indices.set_round_context("\n".join(base_snippets))
        local_logger.log(
            "generator_context",
            topic=topic.topic_id,
            snippet_count=len(base_snippets),
            snippets=base_snippets,
        )

        weights = [1.0 / len(self.policies)] * len(self.policies)
        best_round_utility = -1.0
        rounds_done = 0
        for round_idx in range(self.config.resolved_rounds()):
            candidates = self._generate_candidates(topic, weights, base_snippets, round_idx)
            if not candidates:
                break
            evaluated = self._score_candidates(candidates, retriever, base_snippets, local_logger)
            if not evaluated:
                break

            survivors = self._select_survivors(evaluated)
            if not survivors:
                break

            self._compute_clarity(topic, survivors, retriever, base_snippets, local_logger)

            round_utility = self._assign_utilities(evaluated, survivors, local_logger)
            weights = self._update_weights(weights, survivors)
            self._record_survivors(topic, survivors)
            rounds_done = round_idx + 1

            local_logger.log(
                "round_complete",
                topic=topic.topic_id,
                round=round_idx,
                survivors=len(survivors),
                round_utility=round_utility,
                weights=weights,
            )

            if best_round_utility >= 0 and round_utility - best_round_utility < 1e-3:
                break
            best_round_utility = max(best_round_utility, round_utility)
        status = "completed" if rounds_done >= self.config.resolved_rounds() else "early_stop"
        local_logger.log(
            "topic_complete",
            topic=topic.topic_id,
            rounds_done=rounds_done,
            status=status,
        )

    def _seed_snippets(self, topic: TopicCorpus) -> List[str]:
        snippets: List[str] = []
        reference_docs = topic.reference_documents()
        doc_slice = reference_docs[: self.config.retrieval.top_k_docs]
        if not doc_slice:
            doc_slice = topic.documents[: self.config.retrieval.top_k_docs]
        for doc in doc_slice:
            sentences = split_sentences(doc.text)
            selected = sentences[: self.config.retrieval.snippet_sentences]
            block = " ".join(selected)
            block = block[: self.config.retrieval.snippet_char_limit]
            snippets.append(f"{doc.title}: {block}")
        if not snippets:
            snippets.append(topic.summary)
        return snippets

    def _generate_candidates(
        self,
        topic: TopicCorpus,
        weights: List[float],
        base_snippets: List[str],
        round_idx: int,
    ) -> List[CandidateQuestion]:
        total = self.config.per_round_candidates
        counts = self._policy_counts(weights, total)
        candidates: List[CandidateQuestion] = []
        for policy, count in zip(self.policies, counts):
            if count <= 0:
                continue
            questions = self.generator.propose_questions(
                policy=policy,
                topic=topic,
                snippets=base_snippets,
                target_dimension=self.config.target_dimension,
                num_questions=count,
                round_idx=round_idx,
                max_tokens=self.config.generator_max_tokens,
            )
            if not questions:
                self.logger.log(
                    "generator_no_questions",
                    topic=topic.topic_id,
                    round=round_idx,
                    policy=policy.name,
                )
                continue
            for q in questions:
                cleaned = normalize_whitespace(q)
                if not cleaned:
                    self.logger.log(
                        "candidate_rejected_empty",
                        topic=topic.topic_id,
                        round=round_idx,
                        policy=policy.name,
                    )
                    continue
                candidates.append(CandidateQuestion(text=cleaned, policy=policy, round_idx=round_idx))
        self.logger.log(
            "candidates_generated",
            topic=topic.topic_id,
            round=round_idx,
            total=len(candidates),
        )
        return candidates

    def _policy_counts(self, weights: List[float], total: int) -> List[int]:
        base = [max(1, int(round(w * total))) for w in weights]
        diff = sum(base) - total
        while diff > 0:
            idx = base.index(max(base))
            if base[idx] > 1:
                base[idx] -= 1
                diff -= 1
            else:
                break
        while diff < 0:
            idx = base.index(max(base))
            base[idx] += 1
            diff += 1
        return base

    def _score_candidates(
        self,
        candidates: List[CandidateQuestion],
        retriever,
        base_snippets: List[str],
        logger: StructuredLogger,
    ) -> List[dict]:
        tasks = []
        for cand in candidates:
            snippets = self._question_snippets(retriever, cand.text, base_snippets)
            tasks.append((cand, snippets))

        def worker(item):
            cand, snippets = item
            stats = self.indices.evaluate_question(cand.text, snippets, include_clarity=False)
            result = {
                "question": cand.text,
                "policy": cand.policy.name,
                "round": cand.round_idx,
                "snippets": snippets,
                **stats,
            }
            logger.log(
                "question_scored",
                round=cand.round_idx,
                policy=cand.policy.name,
                question=cand.text,
                gap=stats["gap"],
                disagr=stats["disagreement"],
                proxy=stats["proxy"],
                answers=stats["answers"],
            )
            logger.log(
                "qa_trace",
                round=cand.round_idx,
                policy=cand.policy.name,
                question=cand.text,
                snippets=snippets,
                answers=stats["answers"],
                proxy=stats["proxy"],
            )
            return result

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            results = list(pool.map(worker, tasks))
        return results

    def _question_snippets(
        self,
        retriever,
        question: str,
        fallback: List[str],
    ) -> List[str]:
        retrieved: List[RetrievalResult] = retriever.query(question, top_k=self.config.retrieval.top_k_docs)
        snippets: List[str] = []
        for res in retrieved:
            if res.support_sentences:
                for sent in res.support_sentences:
                    snippets.append(f"{res.document.title}: {sent}")
            else:
                snippets.append(res.document.text[:200])
        return snippets or fallback

    def _select_survivors(self, evaluated: List[dict]) -> List[dict]:
        alpha, beta, _ = self.config.weights
        scored = [
            (
                alpha * record["gap"] + beta * record["disagreement"],
                record,
            )
            for record in evaluated
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        survivors = [record for _, record in scored[: self.config.survivors]]
        self.logger.log("survivors_selected", survivors=len(survivors))
        return survivors

    def _compute_clarity(self, topic: TopicCorpus, survivors: List[dict], retriever, fallback: List[str], logger: StructuredLogger) -> None:
        def worker(record: dict) -> float:
            snippets = record.get("snippets") or self._question_snippets(retriever, record["question"], fallback)
            clarity_answers = [
                self.evaluator.clarity_answer(record["question"], snippets, self.config.target_dimension)
                for _ in range(self.config.clarity_answers)
            ]
            logger.log(
                "clarity_answers",
                topic=topic.topic_id,
                round=record.get("round"),
                question=record["question"],
                answers=clarity_answers,
            )
            clarity = self.indices.compute_clarity(record["question"], snippets, clarity_answers)
            record["clarity"] = clarity
            # Aggregate claimed levers and groundings for logging/analysis.
            claimed = []
            groundings = []
            for ans in clarity_answers:
                if ans.get("levers"):
                    claimed.extend(ans["levers"])
                if ans.get("grounding"):
                    groundings.append(ans["grounding"])
            record["claimed_levers"] = sorted(set(claimed))
            if groundings:
                record["grounding"] = sorted(set(groundings))
            return clarity

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            list(pool.map(worker, survivors))

    def _assign_utilities(self, evaluated: List[dict], survivors: List[dict], logger: StructuredLogger) -> float:
        gamma = self.config.weights[2]
        gaps = [record["gap"] for record in evaluated]
        disagr = [record["disagreement"] for record in evaluated]
        clarity = [record.get("clarity", 0.0) for record in survivors]
        clarity_capped = [min(c, self.config.clarity_cap) for c in clarity] if clarity else []

        gap_norm = dict(zip([id(record) for record in evaluated], min_max_normalize(gaps)))
        disagr_norm = dict(zip([id(record) for record in evaluated], min_max_normalize(disagr)))

        clarity_norm = min_max_normalize(clarity_capped)
        for idx, survivor in enumerate(survivors):
            survivor["utility"] = (
                self.config.weights[0] * gap_norm[id(survivor)]
                + self.config.weights[1] * disagr_norm[id(survivor)]
                + gamma * (clarity_norm[idx] if clarity_norm else 0.0)
            )
            survivor["clarity_scaled"] = clarity_capped[idx] if clarity_capped else 0.0
            # Persist claimed levers count for downstream analysis.
            survivor["claimed_lever_count"] = len(survivor.get("claimed_levers", []))
        utilities = [s["utility"] for s in survivors]
        if survivors:
            metrics = [
                {
                    "question": s.get("question", "")[:280],
                    "policy": s.get("policy"),
                    "round": s.get("round"),
                    "clarity": s.get("clarity"),
                    "claimed_levers": s.get("claimed_levers", []),
                    "claimed_lever_count": s.get("claimed_lever_count", 0),
                    "gap": s.get("gap"),
                    "disagreement": s.get("disagreement"),
                    "utility": s.get("utility"),
                }
                for s in survivors
            ]
            logger.log(
                "survivor_metrics",
                round=survivors[0].get("round"),
                survivors=metrics,
            )
        return median(utilities)

    def _record_survivors(self, topic: TopicCorpus, survivors: List[dict]) -> None:
        with self._best_lock:
            allowed_levers = set(ATTRIBUTE_PANELS[self.config.target_dimension].keys())
            for s in survivors:
                claimed_raw = s.get("claimed_levers", [])
                if isinstance(claimed_raw, str):
                    claimed_list = [c.strip() for c in claimed_raw.split(",") if c.strip()]
                else:
                    claimed_list = list(claimed_raw) if claimed_raw else []
                claimed_filtered = [c for c in claimed_list if c in allowed_levers]
                self.best_records.append(
                    {
                        "topic_id": topic.topic_id,
                        "round": s.get("round"),
                        "policy": s.get("policy"),
                        "question": s.get("question"),
                        "utility": s.get("utility"),
                        "gap": s.get("gap"),
                        "disagreement": s.get("disagreement"),
                        "clarity": s.get("clarity"),
                        "claimed_levers": ",".join(claimed_filtered) if claimed_filtered else "",
                        "claimed_lever_count": len(claimed_filtered),
                        "grounding": ",".join(s.get("grounding", [])) if isinstance(s.get("grounding"), list) else s.get("grounding", ""),
                        "source": "survivor",
                    }
                )

    def _topic_completed(self, topic_dir: Path, resolved_rounds: int) -> bool:
        ev_path = topic_dir / "events.jsonl"
        if not ev_path.exists():
            return False
        rounds = 0
        for line in ev_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue
            if data.get("event") == "topic_complete":
                status = data.get("status")
                if status == "early_stop" and self.config.resume:
                    return False
                return True
            if data.get("event") == "round_complete":
                rounds += 1
        return rounds >= resolved_rounds

    def _merge_topic_events(self, topics: List[TopicCorpus]) -> None:
        out_path = self.run_dir / "events.jsonl"
        with out_path.open("w", encoding="utf-8") as out_fp:
            for topic in topics:
                topic_path = self.run_dir / topic.topic_id / "events.jsonl"
                if not topic_path.exists():
                    continue
                out_fp.write(topic_path.read_text())

    def _write_best_records(self) -> None:
        per_topic: Dict[str, Dict[str, List[dict]]] = {}

        def bundle(topic_id: str) -> Dict[str, List[dict]]:
            return per_topic.setdefault(topic_id, {"survivors": [], "candidates": []})

        # Seed with in-memory survivors.
        for rec in self.best_records:
            topic_id = rec.get("topic_id")
            if not topic_id:
                continue
            bundle(topic_id)["survivors"].append(rec)

        topic_dirs = [p for p in self.run_dir.iterdir() if p.is_dir()]
        for tdir in topic_dirs:
            ev_path = tdir / "events.jsonl"
            if not ev_path.exists():
                continue
            for line in ev_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                event = data.get("event")
                if event == "survivor_metrics":
                    for s in data.get("survivors", []):
                        rec = {
                            "topic_id": tdir.name,
                            "round": s.get("round"),
                            "policy": s.get("policy"),
                            "question": s.get("question"),
                            "utility": s.get("utility"),
                            "gap": s.get("gap"),
                            "disagreement": s.get("disagreement"),
                            "clarity": s.get("clarity"),
                            "claimed_levers": ",".join(s.get("claimed_levers", []))
                            if isinstance(s.get("claimed_levers"), list)
                            else s.get("claimed_levers", ""),
                            "claimed_lever_count": s.get("claimed_lever_count", 0),
                            "grounding": s.get("grounding", ""),
                            "source": "survivor",
                        }
                        bundle(tdir.name)["survivors"].append(rec)
                elif event == "question_scored":
                    cand = {
                        "topic_id": tdir.name,
                        "round": data.get("round"),
                        "policy": data.get("policy"),
                        "question": data.get("question"),
                        "gap": data.get("gap"),
                        "disagreement": data.get("disagr")
                        if data.get("disagr") is not None
                        else data.get("disagreement"),
                    }
                    bundle(tdir.name)["candidates"].append(cand)

        if not per_topic:
            return

        out_path = self.run_dir / "best_questions.csv"
        fieldnames = [
            "topic_id",
            "round",
            "policy",
            "question",
            "utility",
            "gap",
            "disagreement",
            "clarity",
            "claimed_levers",
            "claimed_lever_count",
            "grounding",
            "source",
        ]
        with out_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for topic_id, data in per_topic.items():
                survivors = data["survivors"]
                seen = set()
                deduped = []
                for rec in survivors:
                    key = (rec.get("question"), rec.get("round"), rec.get("policy"))
                    if key in seen:
                        continue
                    seen.add(key)
                    rec["source"] = rec.get("source", "survivor")
                    deduped.append(rec)
                deduped = sorted(deduped, key=lambda r: r.get("utility") or 0.0, reverse=True)

                if len(deduped) < 10:
                    candidates = []
                    candidate_seen = {rec.get("question") for rec in deduped}
                    for cand in data["candidates"]:
                        q = cand.get("question")
                        if not q or q in candidate_seen:
                            continue
                        candidates.append(cand)
                        candidate_seen.add(q)
                    if candidates:
                        gaps = [c.get("gap", 0.0) or 0.0 for c in candidates]
                        disagrs = [c.get("disagreement", 0.0) or 0.0 for c in candidates]
                        gap_norm = min_max_normalize(gaps)
                        disagr_norm = min_max_normalize(disagrs)
                        alpha, beta, _ = self.config.weights
                        fallback = []
                        for idx, cand in enumerate(candidates):
                            score = alpha * gap_norm[idx] + beta * disagr_norm[idx]
                            fallback.append(
                                {
                                    "topic_id": topic_id,
                                    "round": cand.get("round"),
                                    "policy": cand.get("policy"),
                                    "question": cand.get("question"),
                                    "utility": score,
                                    "gap": cand.get("gap", 0.0),
                                    "disagreement": cand.get("disagreement", 0.0),
                                    "clarity": 0.0,
                                    "claimed_levers": "",
                                    "claimed_lever_count": 0,
                                    "grounding": "",
                                    "source": "fallback",
                                }
                            )
                        fallback = sorted(fallback, key=lambda r: r.get("utility") or 0.0, reverse=True)
                        deduped.extend(fallback[: max(0, 10 - len(deduped))])
                if len(deduped) < 10 and deduped:
                    # As a last resort, repeat top entries to guarantee 10 rows.
                    idx = 0
                    while len(deduped) < 10:
                        base = deduped[idx % len(deduped)]
                        repeated = dict(base)
                        repeated["source"] = "repeat"
                        deduped.append(repeated)
                        idx += 1

                writer.writerows(deduped[:10])

    def _update_weights(self, weights: List[float], survivors: List[dict]) -> List[float]:
        policy_scores: Dict[str, List[float]] = {}
        for survivor in survivors:
            policy_scores.setdefault(survivor["policy"], []).append(survivor.get("utility", 0.0))
        medians = {name: median(vals) for name, vals in policy_scores.items()}
        exp_weights: List[float] = []
        for policy, w in zip(self.policies, weights):
            score = medians.get(policy.name, 0.0)
            exp_weights.append(w * math.exp(self.config.eta * score))
        total = sum(exp_weights) or 1.0
        new_weights = [w / total for w in exp_weights]
        return new_weights

    def _resolve_dataset_path(self) -> Path:
        configured = self._root_join(Path(self.config.dataset_path))
        candidates = [configured]
        default_dataset = self.pkg_root / "datasets" / configured.name
        candidates.append(default_dataset)
        for cand in candidates:
            path = Path(cand)
            if path.exists():
                return path
        raise FileNotFoundError(f"Dataset not found. Tried: {[str(c) for c in candidates]}")

    def _root_join(self, candidate: Path) -> Path:
        if candidate.is_absolute():
            return candidate
        parts = candidate.parts
        if parts and parts[0] == "CDQG":
            candidate = Path(*parts[1:]) if len(parts) > 1 else Path()
        return (self.pkg_root / candidate).resolve()
