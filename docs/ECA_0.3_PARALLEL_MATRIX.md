# ECA-0.3 — Parallel Matrix Dynamics (experimental)

## Hypothesis

A causal byte-stream model can move most feature processing from per-byte Python/recurrent calls to batched matrix operations over sequence positions, while preserving stream continuity with compact recurrent state.

This experiment tests whether an elementwise affine recurrence can be evaluated using an associative parallel scan. It is not a Transformer and contains no self-attention.

## Processing path

\`UTF-8 bytes → batched embedding/projection → causal depthwise convolution → gated matrix blocks → affine state scan → byte/EOS prediction\`

The position-wise blocks use dense linear projections and gated nonlinear feature interactions over tensors shaped \`[batch, time, hidden]\`. The causal depthwise convolution caches its left history between chunks.

Persistent memory follows:

\`h[t] = a[t] ⊙ h[t-1] + b[t]\`

where retention \`a[t]\` and write \`b[t]\` are learned from the current causal representation. Affine maps compose associatively:

\`(A₂, B₂) ∘ (A₁, B₁) = (A₂ ⊙ A₁, B₂ + A₂ ⊙ B₁)\`

The implementation uses a logarithmic-depth prefix scan across positions. This makes the recurrence parallelizable across time without attention. The recurrence is elementwise, not a full dense matrix-valued state transition.

## What this does and does not establish

- It establishes a testable parallel recurrence implementation.
- It does not establish that the model learns faster or performs better.
- Dense projections are learned by ordinary joint backpropagation; the model does not rewrite GPU kernels or hardware scheduling.
- The write-intensity penalty regularizes memory writes. It does not currently avoid computing the scan or the feature layers.
- The causal convolution has a finite local receptive field, but persistent state carries information across arbitrary input chunks. State capacity remains finite.
- TBPTT remains a training-time activation-memory constraint, not an inference context limit.

## Required comparison

Run ECA-0.2 and ECA-0.3 on the same data split, byte budget, batch size, sequence length, seed and T4 runtime. Record:

- validation bits per byte;
- training bytes per second, excluding validation time where possible;
- peak allocated VRAM;
- GPU utilization while training;
- parameter count;
- write intensity and retention distribution;
- long-stream and chunk-equivalence tests.

A faster implementation with worse validation BPB is not automatically an improvement. Compare against a parameter-matched baseline and repeat with multiple seeds before drawing conclusions.
