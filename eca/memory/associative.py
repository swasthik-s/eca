"""Simple in-memory associative store for the first ECA prototype."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MemoryRecord:
    key: str
    vector: NDArray[np.float32]


class AssociativeMemory:
    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}

    def store(self, key: str, vector: NDArray[np.float32]) -> None:
        if not key:
            raise ValueError("memory key must not be empty")
        if vector.ndim != 1:
            raise ValueError("memory vectors must be one-dimensional")
        self._records[key] = MemoryRecord(key=key, vector=vector.astype(np.float32, copy=True))

    def retrieve(
        self, query: NDArray[np.float32], top_k: int = 5
    ) -> list[tuple[str, float]]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if query.ndim != 1:
            raise ValueError("query must be one-dimensional")
        query_norm = float(np.linalg.norm(query))
        if query_norm == 0.0:
            return []
        scored: list[tuple[str, float]] = []
        for record in self._records.values():
            if record.vector.shape != query.shape:
                continue
            denom = query_norm * float(np.linalg.norm(record.vector))
            score = float(np.dot(query, record.vector) / denom) if denom else 0.0
            scored.append((record.key, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._records)
