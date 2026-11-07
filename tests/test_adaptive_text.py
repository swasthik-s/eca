import torch

from eca.text import ECAConfig, ECAStreamModel
from eca.training.streaming import documents_to_byte_stream


def test_utf8_byte_stream_round_trip():
    text = "ECA — café 🧠"
    values = documents_to_byte_stream([text], max_bytes=1000)
    assert bytes(values[:-1]).decode("utf-8") == text
    assert values[-1] == 256


def test_model_accepts_arbitrary_chunk_lengths_and_carries_state():
    model = ECAStreamModel(ECAConfig(hidden_size=32, byte_embedding_size=16))
    first = torch.tensor([[65, 66, 67]])
    second = torch.tensor([[68, 69, 70, 71, 72]])
    logits1, boundaries1, state = model(first)
    logits2, boundaries2, state2 = model(second, state)
    assert logits1.shape == (1, 3, 257)
    assert logits2.shape == (1, 5, 257)
    assert boundaries1.shape == (1, 3)
    assert boundaries2.shape == (1, 5)
    assert state2.fast.shape == (1, 32)


def test_causal_prefix_is_independent_of_future_suffix():
    torch.manual_seed(4)
    model = ECAStreamModel(ECAConfig(hidden_size=32, byte_embedding_size=16, dropout=0.0))
    model.eval()
    prefix = torch.tensor([[65, 66, 67, 68]])
    extended = torch.tensor([[65, 66, 67, 68, 90, 91, 92]])
    a, ba, _ = model(prefix)
    b, bb, _ = model(extended)
    torch.testing.assert_close(a, b[:, :4])
    torch.testing.assert_close(ba, bb[:, :4])


def test_model_has_no_attention_modules():
    model = ECAStreamModel(ECAConfig(hidden_size=32, byte_embedding_size=16))
    assert not any("Attention" in type(module).__name__ for module in model.modules())
