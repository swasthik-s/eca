# ECA Architecture — Initial Prototype

## Current implementation boundary

The first implementation is intentionally small and inspectable:

`CognitiveEngine` orchestrates:
1. `VectorSymbolicEncoder`: deterministic bipolar symbol vectors and composition operations.
2. `AssociativeMemory`: storage and similarity-based retrieval.
3. `CognitiveState`: explicit step counter and active concept context.

This gives the repository a runnable computational baseline and test surface. It does **not** yet implement perception, learned semantic concepts, a predictive world model, causal reasoning, autonomous learning, or multimodal generation.

## Why start here?

The first experiment is to establish a representation/memory interface that is:
- deterministic and reproducible;
- compositional through binding and bundling;
- independent of a language model;
- small enough to run on CPU and a Colab T4;
- replaceable if benchmarks show a better representation.

## Next engineering milestones

1. Benchmark compositional retrieval under noise and increasing memory size.
2. Add a learned concept-formation module; do not equate random symbol vectors with learned concepts.
3. Add temporal state transitions and a predictive model.
4. Add explicit intervention-based causal tests.
5. Add persistence, replay and continual-learning evaluations.
6. Add modality adapters only after shared representations have measurable utility.

## Research discipline

Every new mechanism must include:
- a falsifiable hypothesis;
- a minimal baseline;
- unit tests;
- a task-level evaluation;
- resource measurements;
- recorded limitations and negative results.

Architecture changes should be adopted only when they improve a defined metric without unacceptable regressions.
