from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .config import RetrievalConfig
from .corpus import RetrievalResult, TopicCorpus
from .llm import LLM
from .config import MethodConfig
from .logger import StructuredLogger
from .prompts import (
    ATTRIBUTE_PANELS,
    EVALUATOR_SYSTEM,
    CLARITY_EVALUATOR_SYSTEM,
    JUDGE_SYSTEM,
    generator_prompt,
    GENERATOR_SYSTEM,
    evaluator_prompt,
    clarity_evaluator_prompt,
    gap_proxy_prompt,
    pairwise_prompt,
    attribute_prompt,
)
from .utils import normalize_whitespace


QUESTION_LINE_RE = re.compile(
    r"^\s*(?:[-*\u2022]|\d+[\).]|Q\d+:)?\s*(?P<body>.+?)\s*$",
    flags=re.IGNORECASE,
)
THINK_BLOCK_RE = re.compile(r"<(think|analysis)>.*?</\\1>", flags=re.IGNORECASE | re.DOTALL)


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    stripped = THINK_BLOCK_RE.sub("", stripped).strip()
    if stripped.lower().startswith(("<think>", "<analysis>")):
        match = re.search(r"[\\[{]", stripped)
        if match:
            stripped = stripped[match.start() :].strip()
        else:
            stripped = re.sub(r"^<[^>]+>\\s*", "", stripped).strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, count=1, flags=re.IGNORECASE).strip()
        if stripped.endswith("```"):
            stripped = stripped[: -3]
    return stripped.strip()


def _parse_json_questions(text: str, key: str) -> Tuple[List[str], bool]:
    cleaned = _strip_code_fences(text)

    def load_candidate(candidate: str) -> List[str] | None:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        value = None
        if isinstance(payload, list):
            value = payload
        elif isinstance(payload, dict):
            for key_name in (key, "questions", "research_questions", "question_list", "items", "queries"):
                if key_name in payload:
                    value = payload[key_name]
                    break
            if value is None and isinstance(payload.get("question"), str):
                return [normalize_whitespace(payload["question"])]
        if isinstance(value, list):
            questions: List[str] = []
            for item in value:
                if isinstance(item, str):
                    cleaned = normalize_whitespace(item)
                    if cleaned:
                        questions.append(cleaned)
                    continue
                if isinstance(item, dict):
                    for field in ("question", "text", "query", "q"):
                        if field in item and isinstance(item[field], str):
                            cleaned = normalize_whitespace(item[field])
                            if cleaned:
                                questions.append(cleaned)
                            break
            return questions
        if isinstance(value, str):
            pieces = [normalize_whitespace(part) for part in re.split(r"[\n;]+", value)]
            return [p for p in pieces if p]
        return None

    questions = load_candidate(cleaned)
    if questions:
        return questions, True

    if "{" in cleaned and "}" in cleaned:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        snippet = cleaned[start:end]
        questions = load_candidate(snippet)
        if questions:
            return questions, True
    if "[" in cleaned and "]" in cleaned:
        start = cleaned.find("[")
        end = cleaned.rfind("]") + 1
        snippet = cleaned[start:end]
        questions = load_candidate(snippet)
        if questions:
            return questions, True
    return [], False


def _regex_questions(text: str) -> List[str]:
    cleaned = _strip_code_fences(text)
    questions: List[str] = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped or stripped in {"{", "}"}:
            continue
        match = QUESTION_LINE_RE.match(stripped)
        candidate = match.group("body") if match else stripped
        candidate = normalize_whitespace(candidate)
        if candidate and candidate.endswith("?"):
            questions.append(candidate)
    return questions


def extract_questions(text: str, key: str = "questions") -> Tuple[List[str], List[str], bool]:
    warnings: List[str] = []
    questions, used_json = _parse_json_questions(text, key)
    if not used_json:
        warnings.append("json_parse_failed")
    if not questions:
        fallback = _regex_questions(text)
        if fallback:
            warnings.append("fallback_regex")
            questions = fallback
    if not questions:
        warnings.append("no_questions_extracted")
    return questions, warnings, used_json


def parse_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


@dataclass
class Generator:
    llm: LLM
    logger: StructuredLogger | None = None

    def propose_questions(
        self,
        policy,
        topic: TopicCorpus,
        snippets: Sequence[str],
        target_dimension: str,
        num_questions: int,
        round_idx: int,
        max_tokens: int,
    ) -> List[str]:
        prompt = generator_prompt(
            policy=policy,
            target_dimension=target_dimension,
            topic_summary=topic.summary,
            corpus_snippets=snippets,
            num_questions=num_questions,
        )
        response = self.llm.generate(prompt, system=GENERATOR_SYSTEM, temperature=0.5, max_tokens=max_tokens)
        raw_text = response.text or ""
        questions, warnings, _ = extract_questions(response.text)
        if self.logger:
            self.logger.log(
                "generator_output",
                topic=topic.topic_id,
                round=round_idx,
                policy=policy.name,
                raw_preview=raw_text[:400],
                questions=questions,
            )
        if warnings and self.logger:
            self.logger.log(
                "generator_parse_warning",
                topic=topic.topic_id,
                round=round_idx,
                policy=policy.name,
                warnings=warnings,
                raw_preview=raw_text[:400],
            )
        return questions


@dataclass
class Evaluator:
    llm: LLM

    def answer(self, question: str, snippets: Sequence[str]) -> dict:
        prompt = evaluator_prompt(question, snippets)
        # Slightly higher temperature to surface diverse, corpus-grounded answers for disagreement.
        response = self.llm.generate(prompt, system=EVALUATOR_SYSTEM, temperature=0.3, max_tokens=256)
        payload = parse_json_object(response.text)
        answer = normalize_whitespace(payload.get("answer", "Unknown"))
        tag = payload.get("tag", "Unknown")
        return {"answer": answer, "tag": tag}

    def clarity_answer(self, question: str, snippets: Sequence[str], target_dimension: str) -> dict:
        prompt = clarity_evaluator_prompt(question, snippets, target_dimension)
        response = self.llm.generate(prompt, system=CLARITY_EVALUATOR_SYSTEM, temperature=0.75, max_tokens=320)
        payload = parse_json_object(response.text)
        idea = normalize_whitespace(payload.get("idea", "Unknown"))
        grounding = payload.get("grounding", "Mixed")
        levers = payload.get("levers", [])
        if isinstance(levers, str):
            levers = [token.strip() for token in levers.split(",") if token.strip()]
        elif not isinstance(levers, list):
            levers = []
        levers = [lv for lv in levers if lv]
        # Keep only levers that belong to the current target dimension to avoid cross-dimension leakage.
        allowed = set(ATTRIBUTE_PANELS[target_dimension].keys())
        levers = [lv for lv in levers if lv in allowed]
        return {"idea": idea, "grounding": grounding, "levers": levers}


@dataclass
class Judge:
    llm: LLM
    config: MethodConfig | None = None

    def gap_proxy(self, question: str, snippets: Sequence[str]) -> dict:
        prompt = gap_proxy_prompt(question, snippets)
        temp = self.config.judge_temperature if self.config else 0.2
        response = self.llm.generate(prompt, system=JUDGE_SYSTEM, temperature=temp, max_tokens=64)
        payload = parse_json_object(response.text)
        return {
            "tag": payload.get("tag", "Unknown"),
            "entailment": payload.get("entailment", "NotEntailed"),
        }

    def pairwise_relation(self, answer_a: str, answer_b: str) -> str:
        prompt = pairwise_prompt(answer_a, answer_b)
        temp = self.config.judge_temperature if self.config else 0.2
        response = self.llm.generate(prompt, system=JUDGE_SYSTEM, temperature=temp, max_tokens=16)
        payload = parse_json_object(response.text)
        return payload.get("relation", "Unrelated")

    def attribute_panel(self, target_dimension: str, context: str, mode: str = "baseline") -> dict:
        if target_dimension not in ATTRIBUTE_PANELS:
            raise ValueError(f"Unknown target dimension: {target_dimension}")
        prompt = attribute_prompt(target_dimension, context, mode=mode)
        temp = self.config.judge_temperature if self.config else 0.2
        response = self.llm.generate(prompt, system=JUDGE_SYSTEM, temperature=temp, max_tokens=128)
        payload = parse_json_object(response.text)
        return {k: payload.get(k, "Uncertain") for k in ATTRIBUTE_PANELS[target_dimension].keys()}
