"""ECA-0.3 parallel matrix dynamics prototype.

The sequence axis is processed with batched projections and causal depthwise
convolution. Persistent state uses an associative affine scan, not attention.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ParallelMatrixConfig:
    hidden_size: int = 192
    byte_embedding_size: int = 48
    depth: int = 3
    kernel_size: int = 5
    dropout: float = 0.05
    eos_id: int = 256

    def __post_init__(self) -> None:
        if self.hidden_size < 16 or self.byte_embedding_size < 8:
            raise ValueError("Model dimensions are too small.")
        if self.depth < 1 or self.kernel_size < 2:
            raise ValueError("depth must be positive and kernel_size must be at least 2.")


@dataclass
class ParallelStreamState:
    memory: torch.Tensor
    conv_history: torch.Tensor

    def detach(self) -> "ParallelStreamState":
        return ParallelStreamState(self.memory.detach(), self.conv_history.detach())


class ParallelMatrixBlock(nn.Module):
    """Position-wise gated matrix transformation applied to all positions."""

    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size * 2)
        self.gate = nn.Linear(hidden_size, hidden_size * 2)
        self.output = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = F.gelu(self.value(self.norm(x)))
        gate = torch.sigmoid(self.gate(self.norm(x)))
        update = self.output(value * gate)
        return x + self.dropout(update)


class ECAParallelMatrixModel(nn.Module):
    """Causal byte model with parallel feature transforms and persistent scan state.

    No self-attention, Transformer blocks, positional embeddings or subword
    tokenizer are used. The scan composes elementwise affine state transitions:
    h[t] = retention[t] * h[t-1] + write[t].
    """

    input_alphabet_size = 257

    def __init__(self, config: ParallelMatrixConfig | None = None) -> None:
        super().__init__()
        self.config = config or ParallelMatrixConfig()
        h = self.config.hidden_size
        e = self.config.byte_embedding_size
        k = self.config.kernel_size

        self.byte_embedding = nn.Embedding(self.input_alphabet_size, e)
        self.input_projection = nn.Linear(e, h)
        self.causal_conv = nn.Conv1d(
            h, h, kernel_size=k, groups=h, bias=False
        )
        self.input_norm = nn.LayerNorm(h)
        self.blocks = nn.ModuleList(
            ParallelMatrixBlock(h, self.config.dropout)
            for _ in range(self.config.depth)
        )
        self.retention_projection = nn.Linear(h, h)
        self.write_projection = nn.Linear(h, h)
        self.predictor = nn.Sequential(
            nn.LayerNorm(h * 2),
            nn.Linear(h * 2, h),
            nn.GELU(),
            nn.Linear(h, self.config.eos_id + 1),
        )

    def initial_state(
        self, batch_size: int, device: torch.device | str
    ) -> ParallelStreamState:
        h = self.config.hidden_size
        history = self.config.kernel_size - 1
        return ParallelStreamState(
            memory=torch.zeros(batch_size, h, device=device),
            conv_history=torch.zeros(batch_size, h, history, device=device),
        )

    @staticmethod
    def affine_parallel_scan(
        retention: torch.Tensor,
        write: torch.Tensor,
        initial: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inclusive associative scan over h[t] = a[t] * h[t-1] + b[t].

        Each scan level doubles the covered prefix. Work over positions is
        tensorized; the loop count is logarithmic in sequence length.
        """
        if retention.shape != write.shape or retention.ndim != 3:
            raise ValueError("retention and write must have matching [B,T,H] shapes.")
        batch, steps, _ = retention.shape
        if initial.shape != (batch, retention.shape[-1]):
            raise ValueError("initial state must have shape [B,H].")
        if steps == 0:
            return write, initial

        a = retention
        b = write
        offset = 1
        while offset < steps:
            shifted_a = torch.cat(
                (torch.ones_like(a[:, :offset]), a[:, :-offset]), dim=1
            )
            shifted_b = torch.cat(
                (torch.zeros_like(b[:, :offset]), b[:, :-offset]), dim=1
            )
            next_a = a * shifted_a
            next_b = b + a * shifted_b
            a, b = next_a, next_b
            offset *= 2

        states = b + a * initial.unsqueeze(1)
        return states, states[:, -1]

    def forward(
        self,
        byte_ids: torch.Tensor,
        state: ParallelStreamState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, ParallelStreamState]:
        if byte_ids.ndim != 2:
            raise ValueError("byte_ids must have shape [batch, time].")
        if byte_ids.numel() and (
            byte_ids.min() < 0 or byte_ids.max() > self.config.eos_id
        ):
            raise ValueError("Input IDs must be UTF-8 bytes [0,255] or EOS [256].")

        batch, steps = byte_ids.shape
        if state is None:
            state = self.initial_state(batch, byte_ids.device)
        if state.memory.shape[0] != batch:
            raise ValueError("State batch size does not match input batch size.")
        if steps == 0:
            empty = byte_ids.new_zeros((batch, 0, self.config.eos_id + 1), dtype=torch.float)
            return empty, byte_ids.new_zeros((batch, 0), dtype=torch.float), state

        embedded = self.byte_embedding(byte_ids)
        projected = self.input_projection(embedded)
        projected_channels = projected.transpose(1, 2)
        memory_initial = state.memory.to(device=byte_ids.device, dtype=projected.dtype)
        conv_history = state.conv_history.to(device=byte_ids.device, dtype=projected.dtype)

        # Cached left context makes causal convolution invariant to chunk splits.
        conv_input = torch.cat((conv_history, projected_channels), dim=-1)
        local_features = self.causal_conv(conv_input).transpose(1, 2)
        features = self.input_norm(projected + local_features)
        for block in self.blocks:
            features = block(features)

        # Learned per-feature retention/write strengths. The recurrence is
        # parallelized with an associative affine scan across sequence positions.
        retention = 0.01 + 0.98 * torch.sigmoid(
            self.retention_projection(features)
        )
        write = (1.0 - retention) * torch.tanh(self.write_projection(features))
        memory_sequence, final_memory = self.affine_parallel_scan(
            retention, write, memory_initial
        )

        logits = self.predictor(torch.cat((features, memory_sequence), dim=-1))
        write_intensity = (1.0 - retention).mean(dim=-1)

        joined_history = torch.cat((conv_history, projected_channels), dim=-1)
        next_history = joined_history[..., -(self.config.kernel_size - 1):]
        next_state = ParallelStreamState(final_memory, next_history)
        return logits, write_intensity, next_state
