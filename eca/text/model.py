"""ECA adaptive recurrent byte-stream model.

ECA carries recurrent state across arbitrary input chunks. Training may use
truncated backpropagation to bound activation memory; that is not an inference
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
    eos_id: int = 256

    def __post_init__(self) -> None:
        if self.hidden_size < 16 or self.byte_embedding_size < 8:
            raise ValueError("Model dimensions are too small for a useful experiment.")


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
    """Causal recurrent processor with a counterfactual consolidation policy.

    Byte IDs 0..255 carry UTF-8 text; ID 256 is a structural EOS control
    symbol. The policy chooses between holding the slow state and updating it.
    """

    input_alphabet_size = 257

    def __init__(self, config: ECAConfig | None = None) -> None:
        super().__init__()
        self.config = config or ECAConfig()
        h, e = self.config.hidden_size, self.config.byte_embedding_size
        self.byte_embedding = nn.Embedding(self.input_alphabet_size, e)
        self.fast_dynamics = nn.GRUCell(e, h)
        self.consolidation_policy = nn.Sequential(
            nn.Linear(h * 2 + 1, h // 2),
            nn.Tanh(),
            nn.Linear(h // 2, 2),  # action 0: hold, action 1: consolidate
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

    def _predict(self, fast: torch.Tensor, slow: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.dropout(torch.cat((fast, slow), dim=-1)))

    def forward(
        self,
        byte_ids: torch.Tensor,
        state: StreamState | None = None,
        targets: torch.Tensor | None = None,
        return_auxiliary: bool = False,
    ):
        if byte_ids.ndim != 2:
            raise ValueError("byte_ids must have shape [batch, time].")
        if byte_ids.numel() and (
            byte_ids.min() < 0 or byte_ids.max() > self.config.eos_id
        ):
            raise ValueError("Input IDs must be UTF-8 bytes [0, 255] or EOS [256].")
        if targets is not None and targets.shape != byte_ids.shape:
            raise ValueError("targets must have the same [batch, time] shape as byte_ids.")

        batch, steps = byte_ids.shape
        if state is None:
            state = self.initial_state(batch, byte_ids.device)
        fast, slow = state.fast, state.slow
        previous_logits = state.previous_logits
        logits_out, probability_out, hard_out = [], [], []
        policy_losses, oracle_actions = [], []

        for t in range(steps):
            current_byte = byte_ids[:, t]
            fast = self.fast_dynamics(self.byte_embedding(current_byte), fast)

            if previous_logits is None:
                surprise = torch.zeros(batch, 1, device=byte_ids.device)
            else:
                surprise = F.cross_entropy(
                    previous_logits.float(), current_byte, reduction="none"
                ).detach().unsqueeze(-1)
                surprise = torch.log1p(surprise)

            policy_input = torch.cat((fast, slow, surprise), dim=-1)
            action_logits = self.consolidation_policy(policy_input)
            action_probabilities = F.softmax(action_logits.float(), dim=-1)
            action = action_logits.argmax(dim=-1)
            update_mask = action.eq(1).unsqueeze(-1)

            # The candidate update is computed in both modes. This avoids a
            # per-byte host/device synchronization; sparse state writes are
            # learned now, while compute-skipping is a separate optimization.
            proposed_slow = self.slow_dynamics(fast, slow)

            if targets is not None:
                # Training-only counterfactual: compare the next-byte loss
                # under hold vs update. The policy never sees the target at
                # inference and learns to predict which action is useful.
                hold_logits = self._predict(fast, slow)
                update_logits = self._predict(fast, proposed_slow)
                target_t = targets[:, t]
                hold_loss = F.cross_entropy(
                    hold_logits.float(), target_t, reduction="none"
                )
                update_loss = F.cross_entropy(
                    update_logits.float(), target_t, reduction="none"
                )
                oracle_action = update_loss.detach().lt(hold_loss.detach()).long()
                policy_losses.append(
                    F.cross_entropy(action_logits.float(), oracle_action)
                )
                oracle_actions.append(oracle_action)
                logits = torch.where(update_mask, update_logits, hold_logits)
            else:
                selected_slow = torch.where(update_mask, proposed_slow, slow)
                logits = self._predict(fast, selected_slow)

            slow = torch.where(update_mask, proposed_slow, slow)
            logits_out.append(logits)
            probability_out.append(action_probabilities[:, 1])
            hard_out.append(action.to(torch.float32))
            previous_logits = logits

        if steps == 0:
            empty_logits = byte_ids.new_zeros(
                (batch, 0, self.config.eos_id + 1), dtype=torch.float
            )
            empty_scalar = byte_ids.new_zeros((batch, 0), dtype=torch.float)
            empty_state = StreamState(fast, slow, previous_logits)
            if return_auxiliary:
                return empty_logits, empty_scalar, empty_state, {
                    "policy_loss": empty_scalar.sum(),
                    "hard_boundary": empty_scalar,
                    "oracle_update_rate": empty_scalar,
                }
            return empty_logits, empty_scalar, empty_state

        result = (
            torch.stack(logits_out, dim=1),
            torch.stack(probability_out, dim=1),
            StreamState(fast, slow, previous_logits),
        )
        if return_auxiliary:
            aux = {
                "policy_loss": (
                    torch.stack(policy_losses).mean()
                    if policy_losses
                    else result[0].new_zeros(())
                ),
                "hard_boundary": torch.stack(hard_out, dim=1),
                "oracle_update_rate": (
                    torch.stack(oracle_actions, dim=1).float().mean()
                    if oracle_actions
                    else result[0].new_zeros(())
                ),
            }
            return (*result, aux)
        return result

    @torch.no_grad()
    def generate(
        self,
        prompt: bytes | str,
        max_new_bytes: int = 512,
        temperature: float = 0.8,
        device: torch.device | str | None = None,
    ) -> bytes:
        """Generate until EOS; max_new_bytes is a safety budget, not context."""
        if max_new_bytes < 1:
            raise ValueError("max_new_bytes must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        device = device or next(self.parameters()).device
        prompt_bytes = prompt.encode("utf-8") if isinstance(prompt, str) else prompt
        ids = torch.tensor([list(prompt_bytes)], dtype=torch.long, device=device)
        if ids.shape[1]:
            logits, _, state = self(ids)
            next_logits = logits[:, -1, :]
        else:
            state = self.initial_state(1, device)
            next_logits = torch.zeros(1, self.config.eos_id + 1, device=device)

        generated = bytearray()
        for _ in range(max_new_bytes):
            probabilities = F.softmax(next_logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probabilities, 1).item())
            if next_id == self.config.eos_id:
                break
            generated.append(next_id)
            token = torch.tensor([[next_id]], dtype=torch.long, device=device)
            logits, _, state = self(token, state)
            next_logits = logits[:, -1, :]
        return bytes(generated)
