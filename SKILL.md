---
name: universal-all-purpose-weight-space-programming
version: 5.0.0
description: Operational Weight-Space Programming (WSP) Engine: programmatic model inspection, graph reconstruction, activation tracing, circuit mapping, causal ablation, gradient-free search, transactional patching, and regression evaluation.
---

# Universal All-Purpose Weight-Space Programming

## Mission

Engineer improvements to any supported trainable neural network by reconstructing its architecture and design intent, measuring its behavior, identifying evidence-backed improvement opportunities, selecting mathematically appropriate interventions, and validating outcomes.

This skill is not limited to language models or neural networks. It covers neural, probabilistic, classical parameterized, geometric, temporal, generative, multimodal, embodied, scientific, neuromorphic and custom model systems. Use the family catalogue and adapter maturity levels to avoid overstating support.

## Core distinction

- Weight-space analysis: inspect parameter structure and numerical properties.
- Weight-space transformation: change parameters using explicit mathematical operations.
- Training-based optimization: update parameters using an objective and data.
- Architecture engineering: change the computational graph or module design.
- Model-system optimization: improve data, preprocessing, inference, memory, runtime or deployment.

Do not confuse these categories or claim one automatically achieves another.

## Non-negotiable rules

1. Discover the actual architecture and implementation before selecting algorithms.
2. Establish a baseline before modifications.
3. Treat every proposed improvement as a hypothesis until measured.
4. Never infer model quality from weight magnitudes alone.
5. Preserve immutable source checkpoints and record hashes.
6. Use format-aware readers for packed or quantized formats.
7. Use architecture-specific adapters and smoke tests.
8. Record data provenance, licenses, versions, seeds, environment and commands.
9. Protect held-out evaluation data from training leakage.
10. Report regressions and uncertainty; do not cherry-pick metrics.
11. Never claim to have accessed inaccessible files, private internals or unobserved model properties.
12. Require explicit approval for destructive, irreversible, privacy-sensitive or costly operations.

## Universal operating loop

DISCOVER -> RECONSTRUCT -> BASELINE -> DIAGNOSE -> HYPOTHESIZE -> PLAN -> EXPERIMENT -> VALIDATE -> EXPORT -> MONITOR

Default to DISCOVER and read-only operation.

## 1. Scope and model discovery

Identify:
- Model purpose and task.
- Model family and architecture.
- Framework and implementation.
- Checkpoint format and revision.
- Parameter count and tensor inventory.
- Input/output modalities and preprocessing.
- Tokenizer, processor or feature pipeline.
- Optimizer/training metadata if available.
- License and data provenance.
- Hardware, memory and runtime constraints.

Use `references/model-family-catalogue.md` as the broad discovery taxonomy. It includes classical parameterized models and cross-cutting checkpoint representations. Supported families to investigate:
- Dense and sparse feed-forward networks.
- CNNs and residual networks.
- RNN, LSTM and GRU.
- Dense Transformers.
- Mixture-of-Experts.
- State-space and recurrent sequence models.
- Vision Transformers and hybrid vision models.
- Diffusion and score-based generative networks.
- GANs and autoencoders.
- Audio/speech networks.
- Multimodal and embodied models.
- Graph neural networks.
- Reinforcement-learning policy/value networks.
- Scientific, physics-informed and operator-learning networks.
- Custom architectures.

If architecture is unknown, remain in discovery mode until source code, config, module tree and a forward pass establish its structure.

## 2. Architecture and design-intent reconstruction

Reconstruct:
- Computational graph and module order.
- Tensor-to-module mapping.
- Input/output shapes and data types.
- Parameter sharing and tied weights.
- Normalization and residual paths.
- Attention, convolution, recurrence, routing or message passing.
- Positional/spatial/time encoding.
- State/cache behavior.
- Loss functions and task heads.
- Initialization and training assumptions where documented.

Read model cards, papers, source implementations and configuration. Distinguish original design intent from your interpretation.

Deliver architecture map with source locations, evidence class, confidence and unknowns.

## 3. Baseline measurement

Define task-specific metrics before changing anything.

Measure, as relevant:
- Accuracy, exact match, F1, AUROC/AUPRC.
- Perplexity or task loss.
- Calibration and uncertainty.
- Generative quality and diversity.
- Robustness and distribution shift.
- Safety and constraint compliance.
- Control success/reward and stability.
- Physical/scientific constraint violations.
- Latency, throughput, memory, energy and model size.

Create train, validation, held-out test and protected regression sets. Record exact preprocessing, decoding, seeds and software versions.

## 4. Weight and tensor analysis

For each tensor record:
- Name, shape, dtype, element count, storage format.
- Module and layer association.
- Min/max, mean, standard deviation, RMS, median and quantiles.
- Zero/near-zero fraction and explicit outlier thresholds.
- Row/column norms where meaningful.
- Quantization scheme and scales where available.
- Hash and source mapping.

For selected matrices, consider:
- Spectral norm.
- Singular-value spectrum.
- Effective rank.
- Condition estimates.
- Low-rank approximation error.
- Correlations and parameter deltas.

Use chunked processing and format-aware decoding. Do not interpret packed quantized bytes as floating-point weights. Expensive decompositions must be justified by the question.

## 5. Functional diagnostics

Find actual weaknesses by combining:
- Reproducible task failures.
- Activation distributions and saturation.
- Gradient and sensitivity measurements.
- Ablation and controlled perturbation.
- Module-level profiling.
- Data coverage and label quality.
- Preprocessing or tokenizer/processor mismatches.
- Deployment conversion drift.

An unusual weight statistic is only a clue, not proof of a defect.

## 6. Universal improvement-opportunity engine

Create an opportunity register. Each entry includes:
- ID and title.
- Observed symptom.
- Evidence and evidence quality.
- Affected task, module and tensors.
- Design-intent context.
- Root-cause hypothesis.
- Proposed intervention.
- Expected benefit.
- Risk and likely regressions.
- Compute/data requirements.
- Reversibility.
- Validation and acceptance criteria.
- Confidence and status.

Rank opportunities by evidence strength, expected utility, cost, reversibility and regression risk. Prefer small falsifiable experiments.

## 7. Mathematical intervention library

Select only methods compatible with the model and objective.

### A. Direct parameter transformations
- Scaling or centering under a justified constraint.
- Low-rank updates and factorization.
- Parameter interpolation.
- Structured matrix transformations.
- Spectral filtering or truncation.
- Sparsification and masking.
- Weight normalization or reparameterization where supported.
- Parameter projection onto constraints.

Every edit needs an explicit equation, bounds, target mapping, rollback and functional test.

### B. Gradient-based optimization
- SGD, momentum, Adam, AdamW and compatible optimizers.
- Gradient clipping and accumulation.
- Learning-rate schedules.
- Weight decay and regularization.
- Mixed precision and loss scaling.
- Partial/frozen parameter training.
- Second-order or approximate curvature methods where justified.

### C. Parameter-efficient adaptation
- LoRA and variants.
- QLoRA where architecture/backend supports it.
- Adapters, prompts and other PEFT methods.
- Rank and target-module search.

Discover target modules dynamically. Never assume names.

### D. Model merging
- Linear interpolation.
- SLERP.
- Task arithmetic.
- TIES and DARE.
- Adapter composition.
- Other compatible merge algorithms.

Verify common base lineage, architecture, tensor shapes, vocabulary and parameter semantics.

### E. Compression
- Quantization.
- Structured and unstructured pruning.
- Sparsity.
- Knowledge distillation.
- Weight sharing and factorization.

Measure real deployment gains, not only theoretical parameter reduction.

### F. Training and adaptation
- Supervised fine-tuning.
- Continued pretraining.
- Preference optimization.
- Reinforcement learning.
- Distillation.
- Domain adaptation.
- Curriculum and data rebalancing.

Require licensed, relevant, quality-controlled data and held-out validation.

### G. Architecture changes
- Add/remove/replace modules.
- Change width/depth or routing.
- Modify attention/convolution/recurrent mechanisms.
- Change normalization or positional mechanisms.
- Alter multimodal fusion.

Architecture changes require implementation compatibility, initialization strategy, training plan and explicit approval.

## 8. Architecture-specific adapters

Load or develop an adapter for:
- MLP/FFN.
- CNN/residual.
- RNN/LSTM/GRU.
- Dense Transformer.
- MoE.
- State-space/recurrent.
- ViT/hybrid vision.
- Diffusion/generative.
- Audio/speech.
- Multimodal.
- GNN.
- RL policy/value.
- Scientific/PINN/operator models.
- Unknown/custom.

Each adapter must satisfy `references/universal-adapter-contract-v2.md`. Use `references/intervention-compatibility-matrix.md` only as a screening guide, not as permission to mutate. Family-specific adapter templates are in `adapters/`. For unknown architectures follow `references/custom-architecture-workflow.md`. Report maturity per capability (L0-L6); do not claim universal support based on a family name.

No adapter may claim universal compatibility based on architecture family alone.

## 9. Experiment lifecycle

For each experiment:
1. Freeze source checkpoint and hash.
2. Define hypothesis and success criteria.
3. Create isolated run directory.
4. Save config, code revision, data version and environment.
5. Estimate compute and memory.
6. Run a small smoke test.
7. Execute with controlled seed where possible.
8. Log loss, gradients, metrics, resource use and checkpoints.
9. Stop on divergence, NaNs, unexpected tensor changes or severe regressions.
10. Validate on target and protected suites.
11. Accept, reject or request more evidence.
12. Preserve full reproducibility record.

## 10. Evaluation and acceptance

Compare baseline and candidate on identical data and settings.

Report:
- Per-task metrics and absolute/relative deltas.
- Repeated-run variance or confidence intervals where appropriate.
- Protected regressions.
- Robustness and safety.
- Latency, throughput, memory, energy and size.
- Export/runtime parity.
- New failure modes.

A candidate is not an improvement unless predeclared criteria are met. If criteria are missing, propose them before expensive training.

## 11. Deployment and monitoring

Validate checkpoint loading, processor/tokenizer, preprocessing, output heads, inference settings, conversion parity, latency and memory.

Create model card, provenance manifest, known limitations, rollback route and post-deployment monitoring plan.

## 12. Tool strategy

Use deterministic tools for numerical work:
- PyTorch/JAX/TensorFlow as appropriate.
- Safetensors and format-specific readers.
- NumPy/SciPy.
- PEFT and compatible training libraries.
- ONNX and runtime tools where supported.
- GGUF/llama.cpp only for compatible models.
- Architecture-specific profiling tools.

Inspect installed versions and actual API compatibility. Do not invent commands or APIs.

## 13. Final response contract

State:
1. Model and architecture identified.
2. Files and evidence inspected.
3. Baseline and constraints.
4. Opportunities discovered, ranked.
5. Facts vs hypotheses.
6. Experiments run and changes made.
7. Metrics and regressions.
8. Unknowns and limitations.
9. Output paths and reproduction commands.

Never claim completion without an artifact and verification.


## 14. Expanded model-family coverage (v2)
Consult `references/model-family-catalogue.md` before classifying a model. The taxonomy spans sequence/attention, SSM/recurrent, sparse/routed, vision/video/3D, generative, audio/speech/music, multimodal/embodied, graph/geometric, RL/control, time-series/tabular, scientific/continuous-time, spiking/neuromorphic, classical ML and parameter-format variants. The catalogue is intentionally extensible. Unknown models use the custom-architecture workflow.

## 15. Non-neural parameter objects
Do not force trees, boosting ensembles, SVMs, kernels, statistical state-space models or retrieval indexes into a tensor-only abstraction. Their learned state may be split thresholds, topology, support vectors, coefficients, exemplars, embeddings, calibration objects or index structures. Use family-specific parameter adapters and validation.

## 16. Adapter maturity and claims
Every adapter reports L0 Catalogue, L1 Read-only, L2 Structural, L3 Diagnostic, L4 Transform, L5 Training or L6 Export maturity separately for each capability. A catalogue entry is not an implementation. A successful load is not proof of correct behavior. A successful export is not proof of numerical or task parity.


## Open-world architecture and adapter extensibility (v3)
Treat every architecture catalogue as non-exhaustive. New, custom, hybrid and future model families must enter the Unknown Model Discovery Protocol rather than being rejected or forced into a known family. Use the Adapter Plugin SDK and manifest schema to create a new adapter or compose multiple adapters. Select operations by verified capabilities and preconditions, not model-family labels. Keep unknown capabilities blocked. Require conformance tests before promoting an adapter capability. Read `references/future-architecture-compatibility.md`, `references/adapter-plugin-sdk.md`, `references/unknown-model-discovery-protocol.md`, and `references/capability-based-operation-routing.md`.


## Open-world architecture and adapter extensibility (v3)
Treat every architecture catalogue as non-exhaustive. New, custom, hybrid and future model families must enter the Unknown Model Discovery Protocol rather than being rejected or forced into a known family. Use the Adapter Plugin SDK and manifest schema to create a new adapter or compose multiple adapters. Select operations by verified capabilities and preconditions, not model-family labels. Keep unknown capabilities blocked. Require conformance tests before promoting an adapter capability. Read `references/future-architecture-compatibility.md`, `references/adapter-plugin-sdk.md`, `references/unknown-model-discovery-protocol.md`, and `references/capability-based-operation-routing.md`.


## Weight-Space Programming (WSP) agentic execution protocol

For direct, gradient-free parameter programming and lower-capability orchestrator models, follow `wsp_protocol/README.md` and its playbooks. This protocol makes the agent an experiment director rather than a raw-float generator.

### Required execution architecture

`Agent LLM <-> typed Python/MCP orchestrator <-> architecture adapter <-> target model/checkpoint`

The LLM receives compact structural maps, evidence summaries, experiment ledgers and patch manifests. Deterministic tools perform all tensor arithmetic, activation capture, search, evaluation, patch application and integrity checks. Do not pass full weight arrays into chat unless a small, explicitly bounded example is necessary.

### Required WSP stages

1. Artifact authorization, immutable hash and format inventory.
2. Architecture discovery and observed computation graph reconstruction.
3. Circuit map construction at model, module, tensor, runtime and behavior levels.
4. Reproducible behavioral baseline and dataset split.
5. Activation and parameter diagnostics with controls.
6. Competing, falsifiable circuit hypotheses.
7. Least-invasive intervention selection.
8. Bounded gradient-free search over a compact parameterization when appropriate.
9. Multi-objective validation and regression testing.
10. Reversible sparse/low-rank/structured patch creation.
11. Transactional apply-to-copy, reload, hash verification and rollback.
12. Final report with uncertainty, evidence and exact reproduction steps.

### Keep intervention categories distinct

- Activation ablation is a temporary causal probe, not a permanent weight edit.
- Parameter ablation and direct parameter deltas are weight-space interventions.
- Random search, coordinate search, simulated annealing, evolutionary search and Bayesian optimization are candidate-selection procedures; they do not by themselves prove a discovered circuit.
- Gradient descent/fine-tuning is training-based optimization, not gradient-free WSP.
- Prompt, decoding, tokenizer, KV-cache and runtime fixes are system-level corrections, not weight edits.

### Universal architecture rule

Use the capability contract and unknown-architecture protocol. Never assume Transformer blocks, attention heads, MLP neurons, a particular tensor naming scheme, PyTorch hooks, or a supported loader. For hybrid systems, compose adapters and track shared parameters. New architectures must pass conformance tests before mutation operations are enabled.

### Mandatory safeguards

- Never edit the original checkpoint in place.
- Never zero “top neurons” solely because a summary ranks them highly.
- Never call a PCA component a defect direction without controlled causal evidence.
- Never tune repeatedly on final holdout data.
- Never claim a benchmark improvement without sample size, uncertainty, controls and reproducibility.
- Never claim research replication from secondary sources alone.
- Require human approval before production promotion.

### Tool implementation status

The WSP tool API and MCP server documents define contracts, not pre-existing live services. Reference scripts are intentionally narrow: they support only the explicitly documented checkpoint shapes/formats and must not be described as universal adapters. Extend through tested format- and architecture-specific plugins.
