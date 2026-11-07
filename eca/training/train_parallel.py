"""T4-oriented trainer for ECA-0.3 parallel matrix dynamics."""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

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


def train(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    torch.manual_seed(args.seed)
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
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    parameter_count = sum(p.numel() for p in model.parameters())
    print(
        f"architecture=ECAParallelMatrixModel device={device} "
        f"parameters={parameter_count:,}"
    )
    print(
        f"train_bytes_and_controls={len(train_stream):,} "
        f"valid_bytes_and_controls={len(valid_stream):,}"
    )
    print(
        f"depth={args.depth} kernel_size={args.kernel_size} "
        f"sequence_length={args.sequence_length} batch_size={args.batch_size}"
    )

    lane_stride = len(train_stream) // args.batch_size
    if lane_stride <= args.sequence_length + 1:
        raise RuntimeError("Reduce batch size or sequence length.")
    segments = max(
        1, (lane_stride - args.sequence_length - 1) // args.sequence_length
    )
    state = None
    window_started = time.perf_counter()
    window_bytes = 0

    for step in range(1, args.steps + 1):
        model.train()
        segment = (step - 1) % segments
        start = segment * args.sequence_length
        if segment == 0:
            state = None
        x, y = make_next_byte_batch(
            train_stream, start, args.batch_size, args.sequence_length,
            device, lane_stride
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device, dtype=torch.float16, enabled=device == "cuda"
        ):
            logits, write_intensity, state = model(x, state)
            prediction_loss = F.cross_entropy(
                logits.float().reshape(-1, 257), y.reshape(-1)
            )
            loss = prediction_loss + args.write_cost_weight * write_intensity.mean()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        state = state.detach()
        window_bytes += x.numel()

        if step == 1 or step % args.log_every == 0:
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = max(time.perf_counter() - window_started, 1e-9)
            throughput = window_bytes / elapsed
            valid_bpb = evaluate(model, valid_stream, args.sequence_length, device)
            window_bytes = 0
            window_started = time.perf_counter()
            peak_memory = (
                torch.cuda.max_memory_allocated() / (1024 ** 3)
                if device == "cuda" else 0.0
            )
            print(
                f"step={step:5d} train_loss={prediction_loss.item():.4f} "
                f"valid_bpb={valid_bpb:.4f} "
                f"write_intensity={write_intensity.detach().mean().item():.4f} "
                f"bytes_per_sec={throughput:.1f} "
                f"peak_vram_gb={peak_memory:.3f}"
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": config.__dict__,
            "step": args.steps,
            "seed": args.seed,
            "dataset": args.dataset_id if args.dataset == "hf" else "smoke",
            "architecture": "eca_parallel_matrix_dynamics_0.3",
        },
        output,
    )
    print(f"checkpoint={output.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["hf", "smoke"], default="hf")
    parser.add_argument("--dataset-id", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--documents", type=int, default=2000)
    parser.add_argument("--max-bytes", type=int, default=2_000_000)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    # Near-parameter-matched to the user's ~252k ECA-0.2 run.
    parser.add_argument("--hidden-size", type=int, default=144)
    parser.add_argument("--embedding-size", type=int, default=32)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--write-cost-weight", type=float, default=0.001)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output", default="checkpoints/eca_parallel_matrix_0_3.pt"
    )
    parser.add_argument("--cpu", action="store_true")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
