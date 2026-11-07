"""ECA-0.5: parallel lane streaming + batched matrix training."""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from eca.text.parallel import ECAParallelMatrixModel, ParallelMatrixConfig
from eca.training.parallel_stream import ParallelLaneByteStream
from eca.training.streaming import documents_to_byte_stream, iter_hf_text


def evaluate(model, stream: list[int], length: int, device: str) -> float:
    model.eval()
    state = None
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(stream) - 1, length):
            chunk = stream[start:start + length + 1]
            valid = len(chunk) - 1
            if valid <= 0:
                continue
            if valid < length:
                chunk += [256] * (length + 1 - len(chunk))
            x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits, _, state = model(x, state)
            total += F.cross_entropy(
                logits[:, :valid].float().reshape(-1, 257),
                y[:, :valid].reshape(-1), reduction="sum"
            ).item()
            count += valid
            state = state.detach()
    return total / max(1, count) / math.log(2)


def train(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if args.dataset == "hf":
        source = iter(iter_hf_text(args.dataset_id, args.dataset_config, max_documents=args.documents))
        validation_docs = []
        validation_bytes = 0
        while validation_bytes < args.validation_bytes:
            try:
                doc = next(source)
            except StopIteration:
                break
            validation_docs.append(doc)
            validation_bytes += len(doc.encode("utf-8", errors="replace")) + 1
        train_documents = source
    else:
        docs = [("ECA learns from raw UTF-8 bytes. Parallel matrix processing preserves lane state. " * 40) for _ in range(100)]
        validation_docs, train_documents = docs[:5], iter(docs[5:])

    valid_stream = documents_to_byte_stream(validation_docs, args.validation_bytes)
    if len(valid_stream) < args.sequence_length + 1:
        raise RuntimeError("Validation stream too small. Increase --validation-bytes or documents.")

    config = ParallelMatrixConfig(
        hidden_size=args.hidden_size, byte_embedding_size=args.embedding_size,
        depth=args.depth, kernel_size=args.kernel_size, dropout=args.dropout
    )
    model = ECAParallelMatrixModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    best_path, latest_path = Path(args.best_checkpoint), Path(args.output)
    best_bpb = float("inf")
    stale = 0
    recoveries = 0
    target_steps = args.steps
    states = None
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print(f"ECA-0.5 parallel lanes={args.batch_size} device={device} "
          f"parameters={sum(p.numel() for p in model.parameters()):,}")
    print(f"bounded_buffer_symbols={args.buffer_symbols} per_lane_capacity={args.buffer_symbols // args.batch_size} "
          f"validation_bytes={len(valid_stream):,}")

    with ParallelLaneByteStream(
        train_documents, lanes=args.batch_size,
        capacity_per_lane=max(2048, args.buffer_symbols // args.batch_size),
        low_watermark=args.low_watermark, high_watermark=args.high_watermark
    ) as stream:
        started = time.perf_counter()
        byte_window = 0
        step = 0
        while step < target_steps:
            rows = stream.read_batch(args.sequence_length)
            if not rows:
                print("DATASET_EXHAUSTED")
                break
            if len(rows) != args.batch_size:
                print(f"DATASET_EXHAUSTED active_lanes={len(rows)}")
                break
            batch = torch.tensor(rows, dtype=torch.long, device=device)
            x, y = batch[:, :-1], batch[:, 1:]
            if states is not None and states.memory.shape[0] != x.shape[0]:
                states = None
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits, write_intensity, next_state = model(x, states)
                prediction_loss = F.cross_entropy(logits.float().reshape(-1, 257), y.reshape(-1))
                loss = prediction_loss + args.write_cost_weight * write_intensity.mean()
            if not torch.isfinite(loss):
                states = None
                print(f"step={step} non_finite_loss skipped")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                states = None
                print(f"step={step} non_finite_gradient skipped")
                continue
            scaler.step(optimizer)
            scaler.update()
            states = next_state.detach()
            step += 1
            byte_window += x.numel()

            if step == 1 or step % args.eval_every == 0 or step == target_steps:
                if device == "cuda":
                    torch.cuda.synchronize()
                elapsed = max(time.perf_counter() - started, 1e-9)
                valid_bpb = evaluate(model, valid_stream, args.sequence_length, device)
                peak = torch.cuda.max_memory_allocated() / (1024 ** 3) if device == "cuda" else 0.0
                print(f"step={step} train_loss={prediction_loss.item():.4f} "
                      f"valid_bpb={valid_bpb:.4f} bytes_per_sec={byte_window / elapsed:.1f} "
                      f"buffered={stream.buffered_symbols} peak_vram_gb={peak:.3f} "
                      f"lr={optimizer.param_groups[0]['lr']:.2e}")
                started, byte_window = time.perf_counter(), 0
                improved = math.isfinite(valid_bpb) and valid_bpb < best_bpb - args.min_improvement
                if improved:
                    best_bpb, stale = valid_bpb, 0
                    payload = {"architecture": "eca_parallel_matrix_dynamics_0.5",
                               "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                               "scaler": scaler.state_dict(), "config": config.__dict__,
                               "step": step, "validation_bpb": best_bpb, "seed": args.seed}
                    best_path.parent.mkdir(parents=True, exist_ok=True)
                    temp = best_path.with_suffix(best_path.suffix + ".tmp")
                    torch.save(payload, temp)
                    os.replace(temp, best_path)
                    print(f"BEST_CHECKPOINT {best_path} valid_bpb={best_bpb:.4f}")
                else:
                    stale += 1

                payload = {"architecture": "eca_parallel_matrix_dynamics_0.5",
                           "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "scaler": scaler.state_dict(), "config": config.__dict__,
                           "step": step, "validation_bpb": valid_bpb, "seed": args.seed}
                latest_path.parent.mkdir(parents=True, exist_ok=True)
                temp = latest_path.with_suffix(latest_path.suffix + ".tmp")
                torch.save(payload, temp)
                os.replace(temp, latest_path)

                if stale >= args.patience:
                    if recoveries >= args.max_recoveries or not best_path.exists():
                        print("STOP: validation stalled; recovery budget exhausted.")
                        break
                    saved = torch.load(best_path, map_location=device, weights_only=True)
                    model.load_state_dict(saved["model"])
                    optimizer.load_state_dict(saved["optimizer"])
                    scaler.load_state_dict(saved["scaler"])
                    for group in optimizer.param_groups:
                        group["lr"] *= args.lr_decay
                    recoveries += 1
                    target_steps += args.recovery_extra_steps
                    stale = 0
                    states = None
                    print(f"RECOVERY {recoveries}: restored step={saved['step']} "
                          f"lr={optimizer.param_groups[0]['lr']:.2e} target_steps={target_steps}")

    print(f"best_validation_bpb={best_bpb} best_checkpoint={best_path}")
    print(f"latest_checkpoint={latest_path} recoveries={recoveries} final_step={step}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["hf", "smoke"], default="hf")
    p.add_argument("--dataset-id", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--documents", type=int, default=100000)
    p.add_argument("--validation-bytes", type=int, default=200000)
    p.add_argument("--buffer-symbols", type=int, default=262144)
    p.add_argument("--low-watermark", type=float, default=0.30)
    p.add_argument("--high-watermark", type=float, default=0.90)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--sequence-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--embedding-size", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--write-cost-weight", type=float, default=0.001)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min-improvement", type=float, default=0.002)
    p.add_argument("--lr-decay", type=float, default=0.5)
    p.add_argument("--recovery-extra-steps", type=int, default=200)
    p.add_argument("--max-recoveries", type=int, default=3)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--output", default="checkpoints/eca_0_5_latest.pt")
    p.add_argument("--best-checkpoint", default="checkpoints/eca_0_5_best.pt")
    p.add_argument("--cpu", action="store_true")
    return p


if __name__ == "__main__":
    train(parser().parse_args())
