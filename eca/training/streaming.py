"""Hugging Face streaming and bounded byte-level dataset helpers."""
from __future__ import annotations

from array import array
from collections.abc import Iterable, Iterator, Sequence
import torch


def iter_hf_text(
    dataset_id: str = "HuggingFaceFW/fineweb-edu",
    config_name: str = "sample-10BT",
    split: str = "train",
    max_documents: int = 10000,
) -> Iterator[str]:
    """Yield text documents from a Hugging Face streaming dataset."""
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


def collect_bounded_byte_streams(
    documents: Iterable[str],
    train_max_bytes: int,
    valid_max_bytes: int,
    eos_id: int = 256,
    validation_modulus: int = 20,
) -> tuple[array, array, int]:
    """Consume documents once into bounded uint16 train/validation buffers.

    Every Nth document goes to validation; all others go to training. This
    deterministic document-level split avoids keeping complete documents in
    RAM. uint16 stores byte IDs and EOS=256 in two bytes per symbol.
    """
    if train_max_bytes < 1 or valid_max_bytes < 1:
        raise ValueError("Both train and validation byte budgets must be positive.")
    if validation_modulus < 2:
        raise ValueError("validation_modulus must be at least 2.")

    train = array("H")
    valid = array("H")
    consumed = 0

    for document_index, document in enumerate(documents):
        if len(train) >= train_max_bytes and len(valid) >= valid_max_bytes:
            break
        if not isinstance(document, str) or not document.strip():
            continue

        target = valid if document_index % validation_modulus == 0 else train
        budget = valid_max_bytes if target is valid else train_max_bytes
        remaining = budget - len(target)
        if remaining <= 1:
            consumed += 1
            continue

        encoded = document.encode("utf-8", errors="replace")
        if not encoded:
            consumed += 1
            continue
        target.extend(encoded[: remaining - 1])
        target.append(eos_id)
        consumed += 1

    return train, valid, consumed


def documents_to_byte_stream(
    documents: Iterable[str], max_bytes: int = 5_000_000, eos_id: int = 256
) -> list[int]:
    """Encode UTF-8 bytes and insert a document-end control marker."""
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
    stream: Sequence[int],
    start: int,
    batch_size: int,
    sequence_length: int,
    device: str,
    lane_stride: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create input symbols and next-symbol/EOS targets for one TBPTT segment."""
    x_rows, y_rows = [], []
    stride = sequence_length
    for b in range(batch_size):
        offset = start + b * (lane_stride if lane_stride is not None else stride)
        segment = list(stream[offset : offset + sequence_length + 1])
        if len(segment) < sequence_length + 1:
            segment.extend([256] * (sequence_length + 1 - len(segment)))
        x_rows.append(segment[:-1])
        y_rows.append(segment[1:])
    return (
        torch.tensor(x_rows, dtype=torch.long, device=device),
        torch.tensor(y_rows, dtype=torch.long, device=device),
    )
