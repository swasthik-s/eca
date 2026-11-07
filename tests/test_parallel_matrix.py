import torch

from eca.text.parallel import (
    ECAParallelMatrixModel,
    ParallelMatrixConfig,
)


def small_model():
    return ECAParallelMatrixModel(
        ParallelMatrixConfig(
            hidden_size=32, byte_embedding_size=16, depth=2,
            kernel_size=4, dropout=0.0
        )
    )


def test_affine_parallel_scan_matches_sequential_recurrence():
    torch.manual_seed(2)
    retention = 0.1 + 0.8 * torch.rand(2, 13, 7)
    write = torch.randn(2, 13, 7)
    initial = torch.randn(2, 7)
    parallel, final = ECAParallelMatrixModel.affine_parallel_scan(
        retention, write, initial
    )
    current = initial
    reference = []
    for t in range(retention.shape[1]):
        current = retention[:, t] * current + write[:, t]
        reference.append(current)
    torch.testing.assert_close(parallel, torch.stack(reference, dim=1))
    torch.testing.assert_close(final, current)


def test_parallel_model_supports_variable_chunk_lengths():
    model = small_model()
    x = torch.randint(0, 257, (2, 11))
    logits, writes, state = model(x[:, :3])
    logits2, writes2, state2 = model(x[:, 3:], state)
    assert logits.shape == (2, 3, 257)
    assert logits2.shape == (2, 8, 257)
    assert writes.shape == (2, 3)
    assert writes2.shape == (2, 8)
    assert state2.memory.shape == (2, 32)


def test_parallel_model_chunking_matches_whole_stream():
    torch.manual_seed(7)
    model = small_model().eval()
    x = torch.randint(0, 257, (2, 17))
    whole, whole_write, _ = model(x)
    first, first_write, state = model(x[:, :6])
    second, second_write, _ = model(x[:, 6:], state)
    torch.testing.assert_close(whole, torch.cat((first, second), dim=1), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        whole_write, torch.cat((first_write, second_write), dim=1),
        atol=1e-5, rtol=1e-5
    )


def test_parallel_model_is_causal():
    torch.manual_seed(5)
    model = small_model().eval()
    prefix = torch.tensor([[65, 66, 67, 68]])
    extended = torch.tensor([[65, 66, 67, 68, 90, 91, 92]])
    a, aw, _ = model(prefix)
    b, bw, _ = model(extended)
    torch.testing.assert_close(a, b[:, :4], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(aw, bw[:, :4], atol=1e-5, rtol=1e-5)


def test_parallel_model_backpropagates_and_has_no_attention():
    model = small_model()
    x = torch.randint(0, 257, (2, 9))
    target = torch.randint(0, 257, (2, 9))
    logits, write_intensity, _ = model(x)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 257), target.reshape(-1)
    ) + write_intensity.mean() * 0.001
    loss.backward()
    assert model.input_projection.weight.grad is not None
    assert torch.isfinite(model.input_projection.weight.grad).all()
    assert not any("Attention" in type(module).__name__ for module in model.modules())


def test_parallel_model_handles_empty_input():
    model = small_model()
    logits, writes, state = model(torch.empty((2, 0), dtype=torch.long))
    assert logits.shape == (2, 0, 257)
    assert writes.shape == (2, 0)
    assert state.memory.shape == (2, 32)
