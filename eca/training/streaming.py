"""Hugging Face text streaming and byte-level batching helpers."""
from __future__ import annotations

from collections.abc import Iterable, Iterator
import torch


def iter_hf_text(
    dataset_id: str = "HuggingFaceFW/fineweb-edu",
    config_name: str = "sample-10BT",
    split: str = "train",
    max_documents: int = 10000,
) -> Iterator[str]:
    """Yield text documents without downloading the full corpus."""
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, name=config_name, split=split, streaming=True)
    count = 0
    for row in dataset:
        text = row.get("text")
        if isinstance(text, str) and text.strip():
            yield text
            count += 1
            if count >= max_documents:
                break


def documents_to_byte_stream(
    documents: Iterable[str], max_bytes: int = 5_000_000, eos_id: int = 256
) -> list[int]:
    """Encode UTF-8 bytes and insert a learned document-end target marker."""
    stream: list[int] = []
    for document in documents:
        encoded = document.encode("utf-8", errors="replace")
        if not encoded:
            continue
        remaining = max_bytes - len(stream)
        if remaining <= 1:
            break
        stream.extend(encoded[: remaining - 1])
        stream.append(eos_id)
        if len(stream) >= max_bytes:
            break
    return stream


def make_next_byte_batch(
    stream: list[int], start: int, batch_size: int, sequence_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create input bytes and next-byte/EOS targets for one TBPTT segment."""
    x_rows, y_rows = [], []
    stride = sequence_length
    for b in range(batch_size):
        offset = start + b * stride
        segment = stream[offset : offset + sequence_length + 1]
        if len(segment) < sequence_length + 1:
            segment = segment + [256] * (sequence_length + 1 - len(segment))
        x_rows.append([v if v < 256 else 0 for v in segment[:-1]])
        y_rows.append(segment[1:])
    return (
        torch.tensor(x_rows, dtype=torch.long, device=device),
        torch.tensor(y_rows, dtype=torch.long, device=device),
    )
