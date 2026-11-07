import numpy as np
import pytest

from eca import CognitiveEngine
from eca.memory import AssociativeMemory
from eca.representations import VectorSymbolicEncoder


def test_symbol_encoding_is_deterministic():
    first = VectorSymbolicEncoder(dimensions=256, seed=11)
    second = VectorSymbolicEncoder(dimensions=256, seed=11)
    np.testing.assert_array_equal(first.encode("tree"), second.encode("tree"))


def test_binding_and_bundle_validate_shapes():
    encoder = VectorSymbolicEncoder(dimensions=256)
    a, b = encoder.encode("a"), encoder.encode("b")
    assert encoder.bind(a, b).shape == (256,)
    with pytest.raises(ValueError):
        encoder.bind(a, np.ones(128, dtype=np.float32))
    with pytest.raises(ValueError):
        encoder.bundle([])


def test_memory_retrieves_exact_vector_first():
    encoder = VectorSymbolicEncoder(dimensions=256)
    memory = AssociativeMemory()
    memory.store("river", encoder.encode("river"))
    memory.store("mountain", encoder.encode("mountain"))
    results = memory.retrieve(encoder.encode("river"), top_k=2)
    assert results[0][0] == "river"
    assert results[0][1] == pytest.approx(1.0)


def test_engine_observation_updates_state_and_memory():
    engine = CognitiveEngine(dimensions=256)
    engine.observe("bird", {"can": "fly", "has": "wings"})
    assert engine.state.step == 1
    assert "bird" in engine.state.active_concepts
    assert len(engine.memory) == 1
    assert engine.recall("bird")[0][0] == "bird"
