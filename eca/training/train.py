"""Minimal first-run trainer for ECA's adaptive recurrent byte model."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from eca.text import ECAConfig, ECAStreamModel
from eca.training.streaming import documents_to_byte_stream, iter_hf_text, make_next_byte_batch


def train(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    torch.manual_seed(args.seed)

    if args.dataset == "hf":
        documents = list(
            iter_hf_text(args.dataset_id, args.dataset_config, max_documents=args.documents)
        )
    else:
        base = (
            "ECA is an experimental recurrent cognitive architecture. "
            "It reads a continuous stream of UTF-8 bytes and learns adaptive units. "
            "It carries a fast state and a slow state through the stream. "
        )
        documents = [base * 30 for _ in range(40)]

    if len(documents) < 2:
        raise RuntimeError("Need at least two documents for a document-level train/validation split.")

    split_at = max(1, min(len(documents) - 1, int(len(documents) * 0.95)))
    train_stream = documents_to_byte_stream(documents[:split_at], max_bytes=args.max_bytes)
    valid_stream = documents_to_byte_stream(
        documents[split_at:], max_bytes=max(10_000, int(args.max_bytes * 0.1))
    )
    if len(train_stream) < args.sequence_length + 2:
        raise RuntimeError("Not enough training bytes. Increase --max-bytes or dataset documents.")
    if len(valid_stream) < args.sequence_length + 2:
        raise RuntimeError("Not enough validation bytes. Increase --documents.")

    config = ECAConfig(hidden_size=args.hidden_size, byte_embedding_size=args.embedding_size)
    model = ECAStreamModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,}")
    print(
        f"train_bytes_and_controls={len(train_stream):,} "
        f"valid_bytes_and_controls={len(valid_stream):,}"
    )

    lane_stride = len(train_stream) // args.batch_size
    if lane_stride <= args.sequence_length + 1:
        raise RuntimeError("Reduce batch size or sequence length; each training lane is too short.")
    segments_per_lane = max(
        1, (lane_stride - args.sequence_length - 1) // args.sequence_length
    )
    state = None

    for step in range(1, args.steps + 1):
        model.train()
        segment_index = (step - 1) % segments_per_lane
        start = segment_index * args.sequence_length
        if segment_index == 0:
            state = None

        x, y = make_next_byte_batch(
            train_stream, start, args.batch_size, args.sequence_length, device, lane_stride
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device, dtype=torch.float16, enabled=(device == "cuda")
        ):
            logits, boundaries, state = model(x, state)
            prediction_loss = F.cross_entropy(logits.reshape(-1, 257), y.reshape(-1))
            loss = prediction_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        state = state.detach()

        if step == 1 or step % args.log_every == 0:
            model.eval()
            with torch.no_grad():
                vx, vy = make_next_byte_batch(
                    valid_stream,
                    0,
                    1,
                    min(args.sequence_length, len(valid_stream) - 1),
                    device,
                )
                with torch.autocast(
                    device_type=device, dtype=torch.float16, enabled=(device == "cuda")
                ):
                    vlogits, vboundaries, _ = model(vx)
                    vloss = F.cross_entropy(vlogits.reshape(-1, 257), vy.reshape(-1))
            bpb = float(vloss) / math.log(2)
            print(
                f"step={step:5d} train_loss={float(prediction_loss):.4f} "
                f"valid_bpb={bpb:.4f} boundary_rate={float(boundaries.mean()):.3f} "
                f"valid_boundary_rate={float(vboundaries.mean()):.3f}"
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
        },
        output,
    )
    print(f"checkpoint={output.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["hf", "smoke"], default="hf")
    p.add_argument("--dataset-id", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--documents", type=int, default=2000)
    p.add_argument("--max-bytes", type=int, default=2_000_000)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument(
        "--sequence-length",
        type=int,
        default=256,
        help="TBPTT segment length, not model context limit",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--hidden-size", type=int, default=192)
    p.add_argument("--embedding-size", type=int, default=48)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--output", default="checkpoints/eca_adaptive_text.pt")
    p.add_argument("--cpu", action="store_true")
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
