"""Small orchestration layer connecting representations and memory."""

from __future__ import annotations

from eca.core.state import CognitiveState
from eca.memory.associative import AssociativeMemory
from eca.representations.vector_symbolic import VectorSymbolicEncoder


class CognitiveEngine:
    """Minimal runnable cognitive loop.

    The first prototype supports symbolic observations and associative recall.
    It is a testable substrate, not yet a multimodal or generally intelligent
    system. Modality adapters and predictive learning will be added behind
    stable interfaces as they are implemented.
    """

    def __init__(self, dimensions: int = 4096, seed: int = 7) -> None:
        self.encoder = VectorSymbolicEncoder(dimensions=dimensions, seed=seed)
        self.memory = AssociativeMemory()
        self.state = CognitiveState()

    def observe(self, concept: str, attributes: dict[str, str] | None = None) -> None:
        """Encode and store a concept with optional role-value attributes."""
        vector = self.encoder.encode(concept)
        if attributes:
            pairs = [
                self.encoder.bind(self.encoder.encode(role), self.encoder.encode(value))
                for role, value in sorted(attributes.items())
            ]
            vector = self.encoder.bundle([vector, *pairs])
        self.memory.store(concept, vector)
        self.state.advance(concept)

    def recall(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Retrieve stored concepts most similar to a symbolic query."""
        return self.memory.retrieve(self.encoder.encode(query), top_k=top_k)
