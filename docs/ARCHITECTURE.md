# ECA Architecture — Initial Implementations

## ECA-0.0: symbolic baseline

The first implementation remains a small, inspectable CPU baseline:

`CognitiveEngine` orchestrates:
1. `VectorSymbolicEncoder`: deterministic bipolar symbol vectors and composition operations.
2. `AssociativeMemory`: storage and similarity-based retrieval.
3. `CognitiveState`: explicit step counter and active concept context.

This baseline is not a learned language model and its random symbol vectors are not learned semantic concepts.

## ECA-0.1: adaptive recurrent byte-stream model

The first trainable text model used this processing path:

`UTF-8 bytes + structural EOS → byte embedding → fast GRU dynamics → causal boundary controller → gated slow GRU consolidation → next-byte/EOS predictor`

It deliberately excluded Transformer blocks, self-attention, positional embeddings, subword tokenizers and pretrained Transformer weights. GRU cells were used as recurrent update mechanisms, not claimed as a novel recurrent cell.

The model maintained a fast state that updates for every byte and a slow state that updates only when the controller chooses consolidation. Prediction surprise from the previous step was part of the controller input.

### ECA-0.1 experiment and failure

The first T4 run used 252,130 parameters and 1.5 million training bytes. Validation BPB fell from 8.0073 to 3.8084 over 300 steps. The generated text was still incoherent, as expected for a small model trained briefly.

The key failure was controller collapse: mean consolidation probability rose to approximately 0.956. The controller had no explicit action-selection target, so next-byte prediction alone did not teach a useful distinction between holding and updating the slow state. This result is evidence of statistical learning, not evidence of reasoning or intelligence.

## ECA-0.2: counterfactual consolidation policy

ECA-0.2 replaces the fixed probability threshold with a learned two-action policy:

- **Hold:** preserve the current slow state.
- **Consolidate:** write the proposed fast-to-slow state update.

The policy receives only causal information available at inference:
- current fast state;
- current slow state;
- prediction surprise from the preceding prediction.

It emits two action scores. The selected action is the larger score; there is no fixed interval, target update percentage or manually selected probability threshold.

### Counterfactual policy supervision

During training only, the model evaluates the next-byte prediction under both candidate actions:
1. Predict the target while holding the slow state.
2. Predict the same target after a candidate slow-state update.
3. Label whichever action has lower immediate prediction loss as the counterfactual teacher action.
4. Train the policy to predict that action, alongside the normal next-byte objective.

The next-byte target is used only to construct the training signal. It is not passed into the policy input and is never available to the controller at inference.

This is a one-step greedy teacher, not a long-horizon value function. It is a falsifiable intermediate experiment: the controller may still collapse if updating is almost always the better immediate action. The next research step should test longer-horizon retention and prediction utility rather than assume the mechanism solves consolidation.

The current implementation computes the candidate slow update in both modes to avoid a per-byte GPU synchronization. The policy sparsifies slow-state writes, but this version does **not** yet save the GRU update computation. Compute-skipping is a separate optimization that must be benchmarked.

### State, stream and stopping semantics

There is no fixed inference context window. The caller passes recurrent state between arbitrary chunks, so the model can continue processing a stream without retaining every previous byte. This does not create infinite memory: state capacity, persistent storage, compute and information loss remain finite constraints.

The output predictor includes byte IDs 0–255 and EOS ID 256. During generation, EOS is the learned stop decision. A maximum generation length is retained only as an operational safety budget. EOS stopping and slow-state consolidation are separate decisions.

### Training limits

The trainer uses truncated backpropagation through time (TBPTT). Its segment length bounds the activation graph retained for gradient computation on a T4. The recurrent state is carried and detached between consecutive training segments. TBPTT length is not an architectural context limit.

Validation now traverses the full held-out byte stream in sequential chunks while carrying recurrent state.

## Current limitations

- This is a small experimental language model, not a general cognitive system.
- The counterfactual teacher optimizes immediate next-byte loss, not long-term memory utility.
- The current model still uses standard GRU cells; ECA has not yet demonstrated a novel recurrent primitive.
- The policy may learn a biased action distribution. Measure both mean update probability and actual hard update frequency.
- The current policy action is a learned argmax. It is not proof of self-awareness, autonomous cognition or general intelligence.
- Persistent associative memory, concept formation, causal world modelling, reasoning and multimodal processing are not part of ECA-0.2.
- The latest code must be tested in the target Colab runtime before claiming empirical success.

## Required experiments

1. Verify UTF-8 round-trip and causal prefix invariance.
2. Verify recurrent state continuity across chunks of different lengths.
3. Verify the controller can hold and consolidate, and that its policy head receives gradients.
4. Measure validation BPB, probability update rate, hard update rate and counterfactual teacher update rate.
5. Measure mean/median/max hold-run length and slow-state change magnitude.
6. Compare against a parameter- and compute-matched byte-GRU baseline without a slow state.
7. Compare against a fixed-interval consolidation baseline.
8. Test held-out documents, unfamiliar words, noisy text and long-stream retention.
9. Run multiple seeds and policy-loss-weight ablations before making performance claims.

## Research discipline

Every new mechanism must include:
- a falsifiable hypothesis;
- a minimal baseline;
- unit tests;
- a task-level evaluation;
- resource measurements;
- recorded limitations and negative results.

Architecture changes should be adopted only when they improve a defined metric without unacceptable regressions.


## ECA-0.3: parallel matrix dynamics (experimental)

ECA-0.3 is a separate experimental model, not a replacement for the ECA-0.2 baseline.

Processing path:

\`UTF-8 bytes → batched embedding/projection → causal depthwise convolution → gated matrix blocks → affine state scan → byte/EOS prediction\`

The feature projections operate on the full \`[batch, time, hidden]\` tensor. A causal depthwise convolution supplies local temporal features and carries its left-history cache across input chunks.

Persistent state uses a learned elementwise affine recurrence:

\`h[t] = a[t] ⊙ h[t-1] + b[t]\`

The affine transitions compose associatively, allowing a logarithmic-depth parallel prefix scan across sequence positions. This is not self-attention and does not form a pairwise attention matrix. The recurrence uses elementwise retention rather than a dense matrix-valued state transition.

### ECA-0.3 limitations and measurements

- The architecture is an experimental hypothesis; no speed or quality improvement has yet been demonstrated.
- The write-intensity regularizer discourages unnecessary memory writes but does not currently skip GPU arithmetic.
- The learned projections use standard joint gradient-based training. The model does not rewrite CUDA kernels or schedule GPU hardware.
- The local convolution has a finite receptive field. Persistent scan state carries information across arbitrary chunks, subject to finite state capacity.
- Compare validation BPB, training bytes/second, peak VRAM and active GPU utilization against ECA-0.2 on identical data, seed and T4 settings.
- Verify the associative scan against a sequential reference and verify chunk-equivalent outputs before training.
- Run multiple seeds and compare parameter counts before claiming an advantage.
