import torch
import torch.nn.functional as F

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


def test_chunked_processing_matches_single_stream_processing():
    torch.manual_seed(9)
    model = ECAStreamModel(ECAConfig(hidden_size=32, byte_embedding_size=16, dropout=0.0))
    model.eval()
    stream = torch.tensor([[65, 66, 67, 68, 69, 70, 71]])
    whole, whole_policy, _ = model(stream)
    first, first_policy, state = model(stream[:, :3])
    second, second_policy, _ = model(stream[:, 3:], state)
    torch.testing.assert_close(whole, torch.cat((first, second), dim=1))
    torch.testing.assert_close(
        whole_policy, torch.cat((first_policy, second_policy), dim=1)
    )


def test_causal_prefix_is_independent_of_future_suffix():
    torch.manual_seed(4)
    model = ECAStreamModel(ECAConfig(hidden_size=32, byte_embedding_size=16, dropout=0.0))
    model.eval()
    prefix = torch.tensor([[65, 66, 67, 68]])
    extended = torch.tensor([[65, 66, 67, 68, 90, 91, 92]])
    a, pa, _ = model(prefix)
    b, pb, _ = model(extended)
    torch.testing.assert_close(a, b[:, :4])
    torch.testing.assert_close(pa, pb[:, :4])


def test_policy_can_hold_slow_state_without_a_fixed_boundary_threshold():
    model = ECAStreamModel(ECAConfig(hidden_size=32, byte_embedding_size=16))
    with torch.no_grad():
        model.consolidation_policy[-1].weight.zero_()
        model.consolidation_policy[-1].bias.copy_(torch.tensor([10.0, -10.0]))
    _, _, state = model(torch.tensor([[65, 66, 67, 68]]))
    torch.testing.assert_close(state.slow, torch.zeros_like(state.slow))


def test_counterfactual_training_returns_policy_supervision():
    torch.manual_seed(12)
    model = ECAStreamModel(
        ECAConfig(hidden_size=32, byte_embedding_size=16, dropout=0.0)
    )
    x = torch.randint(0, 256, (2, 12))
    targets = torch.randint(0, 257, (2, 12))
    logits, probabilities, _, auxiliary = model(
        x, targets=targets, return_auxiliary=True
    )
    assert logits.shape == (2, 12, 257)
    assert probabilities.shape == (2, 12)
    assert auxiliary["hard_boundary"].shape == (2, 12)
    assert 0.0 <= auxiliary["oracle_update_rate"].item() <= 1.0
    assert torch.isfinite(auxiliary["policy_loss"])
    (F.cross_entropy(logits.reshape(-1, 257), targets.reshape(-1))
     + auxiliary["policy_loss"]).backward()
    grad = model.consolidation_policy[-1].weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_eos_is_a_structural_control_symbol_not_a_text_tokenizer():
    model = ECAStreamModel(ECAConfig(hidden_size=32, byte_embedding_size=16))
    logits, _, _ = model(torch.tensor([[65, 256, 66]]))
    assert logits.shape == (1, 3, 257)


def test_model_has_no_attention_modules():
    model = ECAStreamModel(ECAConfig(hidden_size=32, byte_embedding_size=16))
    assert not any("Attention" in type(module).__name__ for module in model.modules())
