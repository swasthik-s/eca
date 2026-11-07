"""A compact bipolar vector-symbolic representation baseline.

This module provides deterministic symbol vectors and basic VSA operations.
It is a baseline representation mechanism to measure, compare and potentially
replace—not a claim that vector similarity alone provides semantics.
"""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray


class VectorSymbolicEncoder:
    def __init__(self, dimensions: int = 4096, seed: int = 7) -> None:
        if dimensions < 64:
            raise ValueError("dimensions must be at least 64")
        self.dimensions = dimensions
        self.seed = seed
        self._cache: dict[str, NDArray[np.float32]] = {}

    def encode(self, symbol: str) -> NDArray[np.float32]:
        if not symbol or not symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        key = symbol.strip()
        if key not in self._cache:
            digest = hashlib.sha256(f"{self.seed}:{key}".encode("utf-8")).digest()
            local_seed = int.from_bytes(digest[:8], "little")
            rng = np.random.default_rng(local_seed)
            self._cache[key] = rng.choice(
                np.array([-1.0, 1.0], dtype=np.float32), size=self.dimensions
            )
        return self._cache[key].copy()

    @staticmethod
    def bind(left: NDArray[np.float32], right: NDArray[np.float32]) -> NDArray[np.float32]:
        """Bind two equal-sized bipolar vectors (role/value composition)."""
        if left.shape != right.shape:
            raise ValueError("vectors must have identical shapes")
        return (left * right).astype(np.float32, copy=False)

    @staticmethod
    def bundle(vectors: list[NDArray[np.float32]]) -> NDArray[np.float32]:
        """Bundle vectors by majority sign."""
        if not vectors:
            raise ValueError("at least one vector is required")
        shape = vectors[0].shape
        if any(vector.shape != shape for vector in vectors):
            raise ValueError("all vectors must have identical shapes")
        total = np.sum(np.stack(vectors), axis=0)
        # Resolve exact ties deterministically using the first vector.
        result = np.sign(total)
        ties = result == 0
        result[ties] = vectors[0][ties]
        return result.astype(np.float32)

    @staticmethod
    def similarity(left: NDArray[np.float32], right: NDArray[np.float32]) -> float:
        """Cosine similarity, returned in [-1, 1]."""
        if left.shape != right.shape:
            raise ValueError("vectors must have identical shapes")
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator == 0.0:
            return 0.0
        return float(np.dot(left, right) / denominator)
