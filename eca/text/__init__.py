"""Native byte-stream text model for ECA (no tokenizer or Transformer stack)."""
from eca.text.model import ECAConfig, ECAStreamModel, StreamState
from eca.text.parallel import (
    ECAParallelMatrixModel,
    ParallelMatrixConfig,
    ParallelStreamState,
)

__all__ = [
    "ECAConfig", "ECAStreamModel", "StreamState",
    "ECAParallelMatrixModel", "ParallelMatrixConfig", "ParallelStreamState",
]
