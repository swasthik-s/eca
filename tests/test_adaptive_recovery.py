from pathlib import Path

import torch

from eca.text.parallel import ECAParallelMatrixModel, ParallelMatrixConfig
from eca.training.train_adaptive import atomic_save, build_parser, checkpoint_payload


def test_adaptive_checkpoint_round_trips_model_optimizer_and_metadata(tmp_path: Path):
    config = ParallelMatrixConfig(
        hidden_size=32, byte_embedding_size=16, depth=1, kernel_size=3
    )
    model = ECAParallelMatrixModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    args = build_parser().parse_args(["--dataset", "smoke"])
    payload = checkpoint_payload(
        model, optimizer, scaler, config, args,
        step=25, valid_bpb=4.25, recovery_count=1
    )
    path = tmp_path / "best.pt"
    atomic_save(payload, path)

    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded["architecture"] == "eca_parallel_matrix_dynamics_0.4"
    assert loaded["step"] == 25
    assert loaded["validation_bpb"] == 4.25
    assert loaded["recovery_count"] == 1
    assert loaded["config"]["hidden_size"] == 32
    assert set(loaded["model"]) == set(model.state_dict())
    assert "param_groups" in loaded["optimizer"]
    assert isinstance(loaded["scaler"], dict)


def test_adaptive_parser_has_bounded_recovery_defaults():
    args = build_parser().parse_args([])
    assert args.patience == 3
    assert args.max_recoveries == 3
    assert args.recovery_extra_steps == 200
    assert args.hidden_size == 256
    assert args.depth == 2
