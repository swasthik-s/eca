"""Explicit, serialisable state container for one cognitive process."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CognitiveState:
    """Tracks the current step and lightweight symbolic context.

    This is deliberately not a claim of a complete world model. It is the
    inspectable state boundary that later learning and perception modules use.
    """

    step: int = 0
    active_concepts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def advance(self, concept: str | None = None) -> None:
        self.step += 1
        if concept is not None and concept not in self.active_concepts:
            self.active_concepts.append(concept)

    def reset_context(self) -> None:
        self.active_concepts.clear()
        self.metadata.clear()
