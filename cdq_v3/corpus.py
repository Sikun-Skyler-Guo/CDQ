from __future__ import annotations

import ast
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

from .config import RetrievalConfig
from .utils import normalize_whitespace, split_sentences, tokenize


@dataclass(slots=True)
class Document:
    doc_id: str
    title: str
    abstract: str
    topic_id: str
    role: str = "reference"

    @property
    def text(self) -> str:
        pieces = [self.title.strip(), self.abstract.strip()]
        return normalize_whitespace(". ".join(p for p in pieces if p))


@dataclass(slots=True)
class TopicCorpus:
    topic_id: str
    title: str
    abstract: str
    documents: Sequence[Document] = field(default_factory=list)

    def reference_documents(self) -> List[Document]:
        return [doc for doc in self.documents if doc.role != "target"]

    def build_retriever(self, config: RetrievalConfig, *, references_only: bool = False) -> "BM25Retriever":
        docs = self.reference_documents() if references_only else list(self.documents)
        if not docs:
            docs = list(self.documents)
        return BM25Retriever(docs, config)

    @property
    def summary(self) -> str:
        return normalize_whitespace(f"{self.title}. {self.abstract}")


class CorpusLoader:
    """Loads datasets/data_rag_500.csv into topic-specific corpora."""

    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path

    def _parse_list(self, raw: str) -> List[str]:
        if not raw or raw == "[]":
            return []
        try:
            parsed = ast.literal_eval(raw)
            return [normalize_whitespace(str(item)) for item in parsed]
        except Exception:
            return [normalize_whitespace(token) for token in raw.split("|") if token.strip()]

    def load(self, limit: int | None = None) -> List[TopicCorpus]:
        corpora: List[TopicCorpus] = []
        with self.dataset_path.open("r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                if "\ufefftargetPaperId" in row:
                    row["targetPaperId"] = row.pop("\ufefftargetPaperId")
                if limit is not None and len(corpora) >= limit:
                    break
                topic_id = str(row["targetPaperId"])
                documents: List[Document] = [
                    Document(
                        doc_id=topic_id,
                        title=row.get("target_title", ""),
                        abstract=row.get("target_abs", ""),
                        topic_id=topic_id,
                        role="target",
                    )
                ]
                ref_titles = self._parse_list(row.get("ref_titles", ""))
                ref_abs = self._parse_list(row.get("ref_abs", ""))
                for ridx, title in enumerate(ref_titles):
                    abstract = ref_abs[ridx] if ridx < len(ref_abs) else ""
                    documents.append(
                        Document(
                            doc_id=f"{topic_id}_ref_{ridx}",
                            title=title,
                            abstract=abstract,
                            topic_id=topic_id,
                            role="reference",
                        )
                    )
                corpora.append(
                    TopicCorpus(
                        topic_id=topic_id,
                        title=row.get("target_title", ""),
                        abstract=row.get("target_abs", ""),
                        documents=documents,
                    )
                )
        return corpora


@dataclass
class RetrievalResult:
    document: Document
    score: float
    support_sentences: Sequence[str]


class BM25Retriever:
    """Lightweight BM25 implementation for short documents."""

    def __init__(self, documents: Sequence[Document], config: RetrievalConfig):
        self.documents = list(documents)
        self.config = config
        self.doc_tokens: List[List[str]] = [tokenize(doc.text) for doc in self.documents]
        self.doc_freq = self._compute_doc_freq()
        self.avg_len = sum(len(tokens) for tokens in self.doc_tokens) / max(1, len(self.doc_tokens))

    def _compute_doc_freq(self) -> dict[str, int]:
        freq: dict[str, int] = {}
        for tokens in self.doc_tokens:
            for token in set(tokens):
                freq[token] = freq.get(token, 0) + 1
        return freq

    def query(self, text: str, top_k: int | None = None) -> List[RetrievalResult]:
        tokens = tokenize(text)
        if not tokens:
            return []
        top_k = top_k or self.config.top_k_docs
        scores: List[tuple[float, int]] = []
        for idx, doc_tokens in enumerate(self.doc_tokens):
            score = self._bm25(tokens, doc_tokens)
            scores.append((score, idx))
        scores.sort(reverse=True)
        results: List[RetrievalResult] = []
        for score, doc_idx in scores[:top_k]:
            doc = self.documents[doc_idx]
            support = self._support_sentences(doc.text, tokens)
            results.append(RetrievalResult(document=doc, score=score, support_sentences=support))
        return results

    def _bm25(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        token_counts: dict[str, int] = {}
        for token in doc_tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        score = 0.0
        for token in query_tokens:
            if token not in token_counts:
                continue
            df = self.doc_freq.get(token, 0) + 1
            idf = max(0.0, ((len(self.doc_tokens) - df + 0.5) / (df + 0.5)))
            freq = token_counts[token]
            denom = freq + self.config.k1 * (1 - self.config.b + self.config.b * len(doc_tokens) / self.avg_len)
            score += idf * freq * (self.config.k1 + 1) / denom
        return score

    def _support_sentences(self, text: str, query_tokens: Sequence[str]) -> List[str]:
        sentences = split_sentences(text)
        scored = []
        query_set = set(query_tokens)
        for sent in sentences:
            sent_tokens = set(tokenize(sent))
            overlap = len(query_set & sent_tokens)
            scored.append((overlap, sent))
        scored.sort(reverse=True)
        return [sent for overlap, sent in scored[: self.config.top_k_sentences] if overlap > 0] or sentences[:1]
