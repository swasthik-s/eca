# ECA — EvolveAI Cognitive Architecture

**An experimental, general-purpose, multimodal cognitive architecture for learning, memory, prediction, reasoning and generation.**

ECA is a research and engineering project exploring computational foundations for AI beyond language-centred next-token prediction. The goal is to develop a reusable Python architecture whose shared cognitive systems can learn from different kinds of experience and support multiple modalities.

> **Core principle:** language is one modality, not the foundation of the entire intelligence system.

**Status:** ECA-0.2 adaptive-consolidation baseline and ECA-0.3 parallel matrix experiment implemented; ECA-0.3 T4 validation pending  
**Language:** Python  
**Initial research hardware:** NVIDIA T4 (15 GB VRAM)  
**Initial framework:** PyTorch  
**Project type:** Open-source research

---

## Vision

Build a system that can:

- Acquire information from multiple permitted public sources and user interactions with explicit consent.
- Form reusable concepts and internal representations.
- Learn relationships between observations and experiences.
- Maintain and reorganise persistent memory.
- Predict how situations and environments may evolve.
- Reason, plan, test hypotheses and revise beliefs.
- Learn continuously without uncontrolled catastrophic forgetting.
- Understand and generate different modalities.

The long-term target includes **text, code, images, video, audio, numerical data and spatial information**. These are not expected to be solved simultaneously. The architecture will be developed and evaluated incrementally.

## Architecture direction

ECA is being developed around six cooperating research systems:

| System | Responsibility |
|---|---|
| **Compositional representation** | Represent concepts, entities, properties, events and relationships in reusable forms. |
| **Adaptive computation** | Provide recurrent, dynamic and learnable computational processes. |
| **Associative memory** | Store, connect, retrieve and consolidate experiences and knowledge. |
| **Predictive world model** | Predict state changes, represent uncertainty and investigate causal hypotheses. |
| **Reasoning and planning** | Perform relational inference, algorithmic problem solving, hypothesis testing and planning. |
| **Autonomous learning** | Select learning objectives, acquire information, evaluate progress and propose adaptations. |

These are design targets, not claims that the systems are already implemented.

### Multimodal design

Modality-specific perception and generation components will connect to shared representations and cognitive systems.

| Modality | Perception | Generation |
|---|---|---|
| Text | Language and meaning | Text |
| Code | Program structure and behaviour | Executable programs |
| Images | Visual concepts and structure | Images |
| Video | Objects, events and temporal dynamics | Video sequences |
| Audio | Sound and speech | Audio and speech |
| Spatial | Geometry and spatial relations | 3D structures and scenes |

Cross-modal transfer is a central research objective: knowledge acquired through one modality should be useful in another where the underlying concepts overlap.

## Research foundations

ECA will investigate, adapt and experimentally compare ideas from:

- Hyperdimensional computing and Vector Symbolic Architectures (HDC/VSA)
- Predictive coding and predictive learning
- Active inference and information-seeking behaviour
- Liquid and continuous-time neural computation
- Associative and external-memory systems
- Energy-based models and equilibrium learning
- Neuro-symbolic reasoning and belief revision
- Causal representation learning
- Neural algorithmic reasoning and program synthesis
- Neuromorphic and event-driven computation
- Continual learning and local plasticity
- World models and joint-embedding predictive architectures
- Developmental networks and neural cellular automata

No single existing method is assumed to be the answer. Spiking neural networks may contribute useful mechanisms, but ECA is not defined as an SNN.

## Learning philosophy

ECA will not make next-token prediction its sole or defining learning objective.

The research will investigate a combination of:

- Self-supervised representation learning
- Predictive learning
- Associative learning
- Local and gradient-based adaptation where appropriate
- Continual learning and memory consolidation
- Learning from actions and outcomes
- Causal experiments and interventions
- Uncertainty-aware information acquisition

The learning controller may update model weights and memory. Updates must be measurable, versioned and recoverable.

## Development roadmap

Development is implementation-driven. Research should answer concrete design questions and lead to code, experiments or engineering decisions.

### Stage 1 — Computational foundation
- Define the internal representation and state model.
- Implement the initial computational substrate.
- Implement memory primitives and persistence.
- Establish reproducible tests and benchmarks.

### Stage 2 — Learning and cognition
- Concept formation and composition.
- Associative recall and memory consolidation.
- Predictive learning and uncertainty estimation.
- Relational reasoning and belief revision.

### Stage 3 — Continual learning
- Stream multiple approved datasets.
- Track source provenance and processing progress.
- Learn incrementally and evaluate retention.
- Recover safely from failed updates.

### Stage 4 — Multimodal capabilities
- Begin with text and structured data.
- Add image perception and generation.
- Add temporal/video and audio processing.
- Evaluate cross-modal knowledge transfer.

### Stage 5 — Autonomous adaptation
- Select learning objectives.
- Identify knowledge gaps.
- Run isolated architecture experiments.
- Evaluate structural changes before adoption.

## Initial hardware and engineering

The first development target is a Google Colab runtime with an **NVIDIA T4 GPU and 15 GB VRAM**.

Initial tools:

- Python
- PyTorch
- NumPy and scientific Python libraries
- Google Colab for experiments and training

Engineering constraints:

- Profile actual GPU memory and throughput.
- Use mixed precision only where numerically appropriate.
- Stream datasets instead of loading entire corpora into VRAM.
- Use CPU RAM and persistent storage for larger memories and archives.
- Save checkpoints and training progress outside temporary Colab storage.
- Keep experiments reproducible through recorded configurations and seeds.

The T4 is a prototype-development constraint, not a claim that a full-scale multimodal foundation model can be trained from scratch on this hardware.

## Reliability and data governance

Continual adaptation must not make the system irrecoverable.

Required safeguards include:

- Versioned and atomic checkpoints.
- Last-known-good model recovery.
- Weight, activation and learning-stability monitoring.
- Replay and retention evaluation.
- Isolated experimental copies.
- Dataset provenance, source validation and licensing review.
- Resource quotas and controlled external access.
- Explicit consent for learning from user interactions.
- Per-user data separation and deletion support before production interaction learning.

External access is intended to be configurable and useful, not unrestricted. The system should be able to discover and request new permitted sources through a controlled acquisition layer.

## Evaluation

ECA must demonstrate capabilities through reproducible experiments, not persuasive demonstrations or training loss alone.

| Evaluation | What it tests |
|---|---|
| Concept acquisition | Learning unfamiliar concepts from examples |
| Compositional generalisation | Applying known concepts in unseen combinations |
| Associative recall | Retrieving relevant relationships and experiences |
| Prediction | Predicting unseen state transitions |
| Causal reasoning | Predicting outcomes of controlled interventions |
| Continual learning | Retaining prior knowledge after new learning |
| Algorithmic reasoning | Solving unfamiliar procedural problems |
| Cross-modal transfer | Reusing knowledge across modalities |
| Generation | Producing outputs that meet task-specific validity criteria |
| Efficiency | Learning within measured compute and memory budgets |

Results should be compared with appropriate simple baselines and established methods. Failed experiments and limitations should be documented.

## Repository principles

- **ECA is the architecture and reusable Python framework.**
- **EvolveAI is a model developed using ECA.**
- Research components must be independently testable.
- Proposed mechanisms must be distinguished from validated implementations.
- Architectural novelty must be checked against prior research.
- No claim of general intelligence without strong, reproducible evidence.

## Contributing

The project welcomes contributions to implementation, experiments, benchmarks, documentation and research replication.

Contributions should include tests where applicable and clearly state whether a capability is proposed, experimental or validated.

## Long-term objective

Develop and experimentally validate an adaptive, multimodal cognitive architecture that can acquire knowledge from experience, form reusable representations, build predictive models, reason about unfamiliar situations, retain useful knowledge and generate different forms of information.

**The objective is not to reproduce the Transformer. ECA-0.1 deliberately excludes self-attention, multi-head attention, positional embeddings, Transformer blocks and pretrained Transformer backbones. It uses byte embeddings, fast/slow recurrent dynamics and a learned causal boundary controller. The project will build and test its own computational method.**

---

*ECA is an experimental research project. Its architecture and capabilities are under active development; the objectives above are not claims of achieved performance.*


## Quick start

Install the package in editable mode:

```bash
git clone https://github.com/swasthik-s/eca.git
cd eca
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
python examples/first_observation.py
```

The initial prototype uses NumPy and runs on CPU. PyTorch and GPU training will be introduced when the first learned components are ready.


## ECA-0.1 and ECA-0.2: adaptive recurrent text model

The first learned model is a compact, from-scratch PyTorch system. ECA-0.1 exposed a controller-collapse failure: its mean consolidation probability rose to approximately 0.956. ECA-0.2 replaces the fixed threshold with a learned two-action policy:

`UTF-8 byte stream → fast recurrent state → adaptive boundary controller → gated slow-state consolidation → next-byte/EOS prediction`

- Input is raw UTF-8 bytes (0–255) plus one structural end-of-document control symbol; there is no BPE, WordPiece, SentencePiece or manually defined word vocabulary.
- A fast recurrent state updates on every byte.
- A learned causal controller chooses **hold** or **consolidate**. During training, both candidate actions are evaluated against the observed next-byte target, and the lower-loss action teaches the policy. At inference, the target is unavailable to the controller.
- The model predicts the next byte and a learned EOS control output. EOS is the model's learned generation-stop decision.
- Recurrent state can be carried across arbitrary input chunks. There is no fixed inference context window in the architecture.
- Training uses truncated backpropagation segments to fit finite GPU memory. This is a training resource limit, not a fixed context window. Inference still has finite physical memory and compute.

### First Colab experiment

Open [the ECA-0.2 Adaptive Text Training notebook](https://colab.research.google.com/github/swasthik-s/eca/blob/eca-0.2-counterfactual-controller/notebooks/ECA_Adaptive_Text_Training.ipynb) in Google Colab, select a T4 GPU, and run the cells in order. The notebook streams a small FineWeb-Edu sample from Hugging Face and saves the checkpoint to Google Drive.

Local smoke run:

```bash
python -m pip install -e ".[train,dev]"
pytest
python -m eca.training.train --dataset smoke --steps 20 --max-bytes 50000 --sequence-length 128 --batch-size 2
```

The ECA-0.2 experiment is a prototype-scale feasibility test, not a GPT-1-scale pretraining run and not evidence of general intelligence. Record validation bits-per-byte, policy loss, mean update probability, actual hard update rate, counterfactual teacher update rate, throughput and peak VRAM. The policy currently optimizes immediate next-byte loss; long-horizon memory utility remains unproven. Compare against a parameter-matched byte-GRU baseline before attributing gains to adaptive consolidation.


## ECA-0.3: parallel matrix dynamics (experimental)

ECA-0.3 tests batched matrix feature transformations and an associative affine scan for persistent state. It is a separate model so ECA-0.2 remains available as a baseline.

- Model: \`ECAParallelMatrixModel\`
- Trainer: \`python -m eca.training.train_parallel\`
- T4 configuration: \`configs/t4_parallel_matrix.yaml\`
- Architecture notes: [ECA-0.3 parallel matrix dynamics](docs/ECA_0.3_PARALLEL_MATRIX.md)

Install and run a small smoke experiment:

\`\`\`bash
python -m pip install -e ".[train,dev]"
pytest
python -m eca.training.train_parallel --dataset smoke --steps 20 --max-bytes 50000 --sequence-length 128 --batch-size 2
\`\`\`

For the T4 experiment, use the same FineWeb-Edu dataset, seed, byte budget, batch size and sequence length as ECA-0.2. Record validation BPB, training bytes/second, peak VRAM and GPU utilization. The new architecture is not yet proven faster or more accurate. Its write-intensity regularizer does not currently skip GPU computation.
