# ECA-0.4 — Validation-Gated Learning and Recovery

ECA-0.4 adds a bounded training supervisor to ECA-0.3. It does not claim that
the model can introspect its own understanding. Instead, an external training
controller evaluates observable learning signals and manages recovery.

## What the controller checks

- Validation bits-per-byte (BPB) on a held-out stream.
- Whether validation BPB is finite.
- Training loss and gradient finiteness.
- Whether validation improves by at least `--min-improvement`.
- Whether the model has stalled for `--patience` consecutive evaluations.

Training loss alone is never used as the success criterion.

## Checkpoint policy

Two checkpoint files are written:

- `eca_0_4_best.pt`: best model and optimizer/scaler state by validation BPB.
- `eca_0_4_latest.pt`: most recent evaluation state for inspection.

A checkpoint contains model weights, optimizer state, mixed-precision scaler,
model configuration, training configuration, step, validation BPB, and recovery
count. Checkpoints are written atomically.

## Recovery policy

After the configured number of non-improving validation checks, the controller:

1. Restores the best validation checkpoint (model, optimizer and scaler).
2. Reduces the restored learning rate by `--lr-decay`.
3. Resets transient recurrent activations so they are recomputed under restored weights.
4. Adds `--recovery-extra-steps` to the training budget.
5. Repeats evaluation, up to `--max-recoveries`.

Recovery is bounded. If the validation metric still does not improve after the
maximum number of recovery attempts, training stops and preserves the best
checkpoint. This prevents endless rollback loops.

This is not a guarantee that a model is learning semantically correctly. BPB
measures next-byte prediction. Later ECA evaluations must add long-range
retention, controlled recall, generalization and generated-text assessments.

## T4 starting configuration

Start with hidden size 256 and depth 2. If that run fits and trains reliably,
repeat with 384/3 and 512/4. Each scale should use a separate checkpoint path.
Do not run all scales concurrently on one T4.

Example:

```bash
python -m eca.training.train_adaptive \
  --dataset hf \
  --dataset-id HuggingFaceFW/fineweb-edu \
  --dataset-config sample-10BT \
  --documents 2000 \
  --max-bytes 2000000 \
  --steps 600 \
  --hidden-size 256 \
  --embedding-size 64 \
  --depth 2 \
  --batch-size 8 \
  --sequence-length 256 \
  --eval-every 50 \
  --patience 3 \
  --recovery-extra-steps 200 \
  --max-recoveries 3 \
  --output /content/drive/MyDrive/ECA/checkpoints/eca_0_4_256_latest.pt \
  --best-checkpoint /content/drive/MyDrive/ECA/checkpoints/eca_0_4_256_best.pt
```

For the next scale, change dimensions and checkpoint names:

| Scale | Hidden | Depth | Embedding |
|---|---:|---:|---:|
| A | 256 | 2 | 64 |
| B | 384 | 3 | 64 |
| C | 512 | 4 | 96 |

The model architecture remains byte-based, causal, recurrent-stateful and
attention-free. Sequence length is a TBPTT training segment, not an inference
context limit.
