"""ECA adaptive recurrent byte-stream model.

The model has no attention layers and no fixed architectural context window.
State is passed between arbitrary-length chunks by the caller. Training still
uses truncated backpropagation for bounded GPU memory; this is not an inference
context limit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ECAConfig:
    hidden_size: int = 192
    byte_embedding_size: int = 48
    dropout: float = 0.05
    boundary_threshold: float = 0.5
    eos_id: int = 256

    def __post_init__(self) -> None:
        if self.hidden_size < 16 or self.byte_embedding_size < 8:
            raise ValueError("Model dimensions are too small for a useful experiment.")
        if not 0.0 < self.boundary_threshold < 1.0:
            raise ValueError("boundary_threshold must be between zero and one.")


@dataclass
class StreamState:
    fast: torch.Tensor
    slow: torch.Tensor
    previous_logits: Optional[torch.Tensor] = None

    def detach(self) -> "StreamState":
        return StreamState(
            self.fast.detach(),
            self.slow.detach(),
            None if self.previous_logits is None else self.previous_logits.detach(),
        )


class ECAStreamModel(nn.Module):
    """Causal recurrent processor with learned adaptive consolidation.

    Byte IDs 0..255 carry UTF-8 text; ID 256 is a structural EOS control
    symbol. The output predicts the next byte or EOS as a learned stop action.
    """

    input_alphabet_size = 257  # 0..255 bytes plus one structural EOS control symbol

    def __init__(self, config: ECAConfig | None = None) -> None:
        super().__init__()
        self.config = config or ECAConfig()
        h, e = self.config.hidden_size, self.config.byte_embedding_size
        self.byte_embedding = nn.Embedding(self.input_alphabet_size, e)
        self.fast_dynamics = nn.GRUCell(e, h)
        self.boundary_controller = nn.Sequential(
            nn.Linear(h * 2 + 1, h // 2),
            nn.Tanh(),
            nn.Linear(h // 2, 1),
        )
        self.slow_dynamics = nn.GRUCell(h, h)
        self.dropout = nn.Dropout(self.config.dropout)
        self.predictor = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.GELU(),
            nn.Linear(h, self.config.eos_id + 1),
        )

    def initial_state(self, batch_size: int, device: torch.device | str) -> StreamState:
        h = self.config.hidden_size
        z = torch.zeros(batch_size, h, device=device)
        return StreamState(z, z.clone(), None)

    def forward(
        self, byte_ids: torch.Tensor, state: StreamState | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, StreamState]:
        if byte_ids.ndim != 2:
            raise ValueError("byte_ids must have shape [batch, time].")
        if byte_ids.numel() and (byte_ids.min() < 0 or byte_ids.max() > self.config.eos_id):
            raise ValueError("Input IDs must be UTF-8 bytes [0, 255] or the EOS control symbol [256].")

        batch, steps = byte_ids.shape
        if state is None:
            state = self.initial_state(batch, byte_ids.device)
        fast, slow = state.fast, state.slow
        previous_logits = state.previous_logits
        logits_out, boundary_out = [], []

        for t in range(steps):
            current_byte = byte_ids[:, t]
            embedded = self.byte_embedding(current_byte)
            fast = self.fast_dynamics(embedded, fast)

            if previous_logits is None:
                surprise = torch.zeros(batch, 1, device=byte_ids.device)
            else:
                surprise = F.cross_entropy(
                    previous_logits.float(), current_byte, reduction="none"
                ).detach().unsqueeze(-1)
                surprise = torch.log1p(surprise)

            boundary_input = torch.cat((fast, slow, surprise), dim=-1)
            boundary_probability = torch.sigmoid(self.boundary_controller(boundary_input))
            hard_boundary = (
                boundary_probability >= self.config.boundary_threshold
            ).to(boundary_probability.dtype)
            # Straight-through estimator: hard causal decision in forward pass,
            # differentiable probability in backward pass.
            boundary_gate = hard_boundary.detach() - boundary_probability.detach() + boundary_probability

            proposed_slow = self.slow_dynamics(fast, slow)
            slow = slow + boundary_gate * (proposed_slow - slow)

            combined = self.dropout(torch.cat((fast, slow), dim=-1))
            logits = self.predictor(combined)
            logits_out.append(logits)
            boundary_out.append(boundary_probability.squeeze(-1))
            previous_logits = logits

        if steps == 0:
            empty_logits = byte_ids.new_zeros((batch, 0, self.config.eos_id + 1), dtype=torch.float)
            empty_scalar = byte_ids.new_zeros((batch, 0), dtype=torch.float)
            return empty_logits, empty_scalar, StreamState(fast, slow, previous_logits)

        return (
            torch.stack(logits_out, dim=1),
            torch.stack(boundary_out, dim=1),
            StreamState(fast, slow, previous_logits),
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: bytes | str,
        max_new_bytes: int = 512,
        temperature: float = 0.8,
        device: torch.device | str | None = None,
    ) -> bytes:
        """Generate until the model emits EOS; max_new_bytes is a safety budget.

        The safety budget is not a context window. Generation state is recurrent
        and can be continued across calls by extending this API in later versions.
        """
        if max_new_bytes < 1:
            raise ValueError("max_new_bytes must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        device = device or next(self.parameters()).device
        prompt_bytes = prompt.encode("utf-8") if isinstance(prompt, str) else prompt
        ids = torch.tensor([list(prompt_bytes)], dtype=torch.long, device=device)
        state = None
        if ids.shape[1]:
            logits, _, state = self(ids)
            next_logits = logits[:, -1, :]
        else:
            state = self.initial_state(1, device)
            next_logits = torch.zeros(1, self.config.eos_id + 1, device=device)

        generated = bytearray()
        for _ in range(max_new_bytes):
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())
            if next_id == self.config.eos_id:
                break
            generated.append(next_id)
            token = torch.tensor([[next_id]], dtype=torch.long, device=device)
            logits, _, state = self(token, state)
            next_logits = logits[:, -1, :]
        return bytes(generated)
