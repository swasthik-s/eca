"""ECA-0.4 trainer with validation-gated checkpoints and bounded recovery.

The controller saves the best validation checkpoint. If validation stalls or
regresses for a configured number of evaluations, it restores that checkpoint,
reduces the learning rate, resets transient recurrent state, and spends a
bounded additional compute budget. It never treats training loss alone as proof
of learning quality.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from eca.text.parallel import ECAParallelMatrixModel, ParallelMatrixConfig
from eca.training.streaming import (
    documents_to_byte_stream,
    iter_hf_text,
    make_next_byte_batch,
)


def evaluate(model, stream: list[int], sequence_length: int, device: str) -> float:
    model.eval()
    state = None
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(stream) - 1, sequence_length):
            valid = min(sequence_length, len(stream) - 1 - start)
            x, y = make_next_byte_batch(stream, start, 1, sequence_length, device)
            with torch.autocast(
                device_type=device, dtype=torch.float16, enabled=device == "cuda"
            ):
                logits, _, state = model(x, state)
            loss_sum += F.cross_entropy(
                logits[:, :valid].float().reshape(-1, 257),
                y[:, :valid].reshape(-1),
                reduction="sum",
            ).item()
            count += valid
            state = state.detach()
    return loss_sum / max(1, count) / math.log(2)


def atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(
    model: ECAParallelMatrixModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: ParallelMatrixConfig,
    args: argparse.Namespace,
    step: int,
    valid_bpb: float,
    recovery_count: int,
) -> dict[str, Any]:
    return {
        "architecture": "eca_parallel_matrix_dynamics_0.4",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config.__dict__,
        "step": step,
        "validation_bpb": valid_bpb,
        "recovery_count": recovery_count,
        "seed": args.seed,
        "training_config": vars(args),
    }


def train(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if args.dataset == "hf":
        documents = list(
            iter_hf_text(args.dataset_id, args.dataset_config, max_documents=args.documents)
        )
    else:
        base = (
            "ECA learns representations from continuous byte streams. "
            "Parallel matrix dynamics process many sequence positions together. "
        )
        documents = [base * 30 for _ in range(40)]

    if len(documents) < 2:
        raise RuntimeError("Need at least two documents for train/validation split.")
    split = max(1, min(len(documents) - 1, int(len(documents) * 0.95)))
    train_stream = documents_to_byte_stream(documents[:split], args.max_bytes)
    valid_stream = documents_to_byte_stream(
        documents[split:], max(10_000, int(args.max_bytes * 0.1))
    )
    if len(train_stream) < args.sequence_length + 2 or len(valid_stream) < 2:
        raise RuntimeError("Insufficient data; increase documents or max-bytes.")

    config = ParallelMatrixConfig(
        hidden_size=args.hidden_size,
        byte_embedding_size=args.embedding_size,
        depth=args.depth,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    )
    model = ECAParallelMatrixModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    lane_stride = len(train_stream) // args.batch_size
    if lane_stride <= args.sequence_length + 1:
        raise RuntimeError("Reduce batch size or sequence length.")
    segments = max(1, (lane_stride - args.sequence_length - 1) // args.sequence_length)

    best_bpb = float("inf")
    best_step = 0
    stale_evaluations = 0
    recovery_count = 0
    bad_gradient_steps = 0
    step = 0
    target_steps = args.steps
    state = None
    best_path = Path(args.best_checkpoint)
    latest_path = Path(args.output)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    parameter_count = sum(p.numel() for p in model.parameters())
    print(
        f"architecture=ECAParallelMatrixModel_0.4 device={device} "
        f"parameters={parameter_count:,} hidden={args.hidden_size} depth={args.depth}"
    )
    print(
        f"train_bytes={len(train_stream):,} valid_bytes={len(valid_stream):,} "
        f"initial_steps={args.steps} eval_every={args.eval_every}"
    )

    window_started = time.perf_counter()
    window_bytes = 0
    while step < target_steps:
        step += 1
        model.train()
        segment = (step - 1) % segments
        if segment == 0:
            state = None
        start = segment * args.sequence_length
        x, y = make_next_byte_batch(
            train_stream, start, args.batch_size, args.sequence_length,
            device, lane_stride
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device, dtype=torch.float16, enabled=device == "cuda"
        ):
            logits, write_intensity, next_state = model(x, state)
            prediction_loss = F.cross_entropy(
                logits.float().reshape(-1, 257), y.reshape(-1)
            )
            loss = prediction_loss + args.write_cost_weight * write_intensity.mean()

        if not torch.isfinite(loss):
            bad_gradient_steps += 1
            print(f"step={step} non_finite_loss; update skipped")
            state = None
            if bad_gradient_steps >= args.bad_step_limit and best_path.exists():
                recovery_count += 1
                if recovery_count > args.max_recoveries:
                    raise RuntimeError("Recovery limit reached after non-finite losses.")
                saved = torch.load(best_path, map_location=device, weights_only=True)
                model.load_state_dict(saved["model"])
                optimizer.load_state_dict(saved["optimizer"])
                scaler.load_state_dict(saved["scaler"])
                for group in optimizer.param_groups:
                    group["lr"] *= args.lr_decay
                bad_gradient_steps = 0
                target_steps += args.recovery_extra_steps
                print(
                    f"RECOVERY {recovery_count}: restored step={saved['step']} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} "
                    f"extra_steps={args.recovery_extra_steps}"
                )
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            bad_gradient_steps += 1
            optimizer.zero_grad(set_to_none=True)
            state = None
            print(f"step={step} non_finite_gradient; update skipped")
            continue
        scaler.step(optimizer)
        scaler.update()
        state = next_state.detach()
        window_bytes += x.numel()

        if step == 1 or step % args.eval_every == 0 or step == target_steps:
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = max(time.perf_counter() - window_started, 1e-9)
            throughput = window_bytes / elapsed
            valid_bpb = evaluate(model, valid_stream, args.sequence_length, device)
            peak_memory = (
                torch.cuda.max_memory_allocated() / (1024 ** 3)
                if device == "cuda" else 0.0
            )
            print(
                f"step={step:5d} train_loss={prediction_loss.item():.4f} "
                f"valid_bpb={valid_bpb:.4f} bytes_per_sec={throughput:.1f} "
                f"peak_vram_gb={peak_memory:.3f} lr={optimizer.param_groups[0]['lr']:.2e}"
            )
            window_bytes = 0
            window_started = time.perf_counter()

            improved = math.isfinite(valid_bpb) and valid_bpb < best_bpb - args.min_improvement
            if improved:
                best_bpb = valid_bpb
                best_step = step
                stale_evaluations = 0
                atomic_save(
                    checkpoint_payload(
                        model, optimizer, scaler, config, args, step,
                        valid_bpb, recovery_count
                    ),
                    best_path,
                )
                print(f"BEST_CHECKPOINT saved={best_path} step={step} valid_bpb={best_bpb:.4f}")
            else:
                stale_evaluations += 1

            atomic_save(
                checkpoint_payload(
                    model, optimizer, scaler, config, args, step,
                    valid_bpb, recovery_count
                ),
                latest_path,
            )

            if stale_evaluations >= args.patience:
                if recovery_count >= args.max_recoveries:
                    print("STOP: no validation improvement; maximum recoveries reached.")
                    break
                if not best_path.exists():
                    raise RuntimeError("Cannot recover: no valid best checkpoint exists.")
                saved = torch.load(best_path, map_location=device, weights_only=True)
                model.load_state_dict(saved["model"])
                optimizer.load_state_dict(saved["optimizer"])
                scaler.load_state_dict(saved["scaler"])
                for group in optimizer.param_groups:
                    group["lr"] *= args.lr_decay
                recovery_count += 1
                target_steps += args.recovery_extra_steps
                stale_evaluations = 0
                state = None  # Recompute transient recurrent state under restored weights.
                print(
                    f"RECOVERY {recovery_count}: restored best step={best_step} "
                    f"valid_bpb={best_bpb:.4f}; lr={optimizer.param_groups[0]['lr']:.2e}; "
                    f"new_target_steps={target_steps}"
                )

    print(f"best_checkpoint={best_path.resolve()}")
    print(f"latest_checkpoint={latest_path.resolve()}")
    print(f"best_step={best_step} best_validation_bpb={best_bpb:.4f}")
    print(f"total_steps={step} recovery_count={recovery_count}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["hf", "smoke"], default="hf")
    parser.add_argument("--dataset-id", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--documents", type=int, default=2000)
    parser.add_argument("--max-bytes", type=int, default=2_000_000)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--write-cost-weight", type=float, default=0.001)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-improvement", type=float, default=0.002)
    parser.add_argument("--lr-decay", type=float, default=0.5)
    parser.add_argument("--recovery-extra-steps", type=int, default=200)
    parser.add_argument("--max-recoveries", type=int, default=3)
    parser.add_argument("--bad-step-limit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", default="checkpoints/eca_0_4_latest.pt")
    parser.add_argument("--best-checkpoint", default="checkpoints/eca_0_4_best.pt")
    parser.add_argument("--cpu", action="store_true")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
