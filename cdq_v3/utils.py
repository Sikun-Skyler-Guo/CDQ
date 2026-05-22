from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence, Tuple, TypeVar


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in TOKEN_PATTERN.findall(text or "")]


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text or "")
    return [normalize_whitespace(sent) for sent in sentences if normalize_whitespace(sent)]


def batched(seq: Sequence, size: int) -> Iterator[Sequence]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def seed_everything(seed: int) -> None:
    random.seed(seed)


T = TypeVar("T")


def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    mid = len(sorted_vals) // 2
    if len(sorted_vals) % 2 == 0:
        return 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])
    return sorted_vals[mid]


def min_max_normalize(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if math.isclose(vmin, vmax):
        # If all values are identical, return 0.5 to avoid collapsing utilities.
        return [0.5 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def shannon_entropy(probs: Sequence[float]) -> float:
    eps = 1e-9
    return -sum(p * math.log(max(p, eps)) for p in probs if p > 0)


@dataclass
class UnionFind:
    parent: List[int]

    @classmethod
    def with_size(cls, size: int) -> "UnionFind":
        return cls(parent=list(range(size)))

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        rx = self.find(x)
        ry = self.find(y)
        if rx == ry:
            return
        self.parent[ry] = rx
