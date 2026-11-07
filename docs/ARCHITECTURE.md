# ECA Architecture — Initial Implementations

## ECA-0.0: symbolic baseline

The first implementation remains a small, inspectable CPU baseline:

`CognitiveEngine` orchestrates:
1. `VectorSymbolicEncoder`: deterministic bipolar symbol vectors and composition operations.
2. `AssociativeMemory`: storage and similarity-based retrieval.
3. `CognitiveState`: explicit step counter and active concept context.

This baseline is not a learned language model and its random symbol vectors are not learned semantic concepts.

## ECA-0.1: adaptive recurrent byte-stream model

The first trainable text model is implemented in `eca/text/model.py`.

### Processing path

`UTF-8 bytes + structural EOS → byte embedding → fast GRU dynamics → causal boundary controller → gated slow GRU consolidation → next-byte/EOS predictor`

This implementation intentionally does not use:
- Transformer blocks;
- self-attention or multi-head attention;
- positional embeddings;
- BPE, WordPiece or SentencePiece;
- pretrained Transformer weights.

GRU cells are used as recurrent update mechanisms, not as a Transformer substitute. The aim is to test whether adaptive multi-timescale recurrence and learned consolidation provide a useful alternative processing substrate.

### Adaptive state and effective horizon

The model maintains:
- **Fast state:** updates for every input byte.
- **Slow state:** updated only when the learned boundary controller selects consolidation.
- **Prediction state:** previous-step logits provide a causal prediction-surprise signal.

The controller receives fast state, slow state and prediction surprise. Its learned probability produces a hard causal consolidation decision with a straight-through gradient estimator. If the gate is closed, the slow state is held; if open, the slow state incorporates the fast representation.

There is no fixed inference context window. The caller passes recurrent state between arbitrary chunks, so the model can continue processing a stream without retaining every previous byte. This does not create infinite memory: state capacity, persistent storage, compute and information loss remain finite constraints.

The output predictor includes byte IDs 0–255 and EOS ID 256. During generation, EOS is the learned stop decision. A maximum generation length is retained only as an operational safety budget.

### Training limits

The first trainer uses truncated backpropagation through time (TBPTT). Its segment length bounds the activation graph retained for gradient computation on a T4. The recurrent state is carried and detached between consecutive training segments. TBPTT length is not an architectural context limit.

### Current training path

- Hugging Face streaming: `HuggingFaceFW/fineweb-edu`, `sample-10BT` configuration.
- Raw UTF-8 byte encoding; no learned or fixed word/subword vocabulary.
- Next-byte prediction plus a structural EOS target.
- FP16 autocast on CUDA, gradient scaling and gradient clipping.
- Checkpoint saved to a persistent path supplied by the notebook/user.

The Colab notebook is `notebooks/ECA_Adaptive_Text_Training.ipynb`.

## Current limitations

- This is a small experimental language model, not a general cognitive system.
- The current adaptive gate is a simple learned boundary/consolidation mechanism. Its usefulness must be measured against matched baselines.
- EOS learns document-ending behaviour from corpus boundaries; generation stopping quality must be evaluated.
- Persistent associative memory, concept formation, causal world modelling, reasoning and multimodal processing are not part of ECA-0.1.
- Training and GPU execution must be run in Colab before claiming empirical success.

## Required experiments

1. Verify UTF-8 round-trip and causal prefix invariance.
2. Verify recurrent state continuity across chunks of different lengths.
3. Measure bits-per-byte (BPB), boundary rate, average consolidation interval, throughput and peak VRAM.
4. Compare with a parameter- and compute-matched byte-GRU baseline without adaptive consolidation.
5. Compare with a fixed-interval consolidation baseline.
6. Test held-out documents, unfamiliar words, noisy text and long-stream retention.
7. Run multiple seeds before making performance claims.

## Research discipline

Every new mechanism must include:
- a falsifiable hypothesis;
- a minimal baseline;
- unit tests;
- a task-level evaluation;
- resource measurements;
- recorded limitations and negative results.

Architecture changes should be adopted only when they improve a defined metric without unacceptable regressions.
