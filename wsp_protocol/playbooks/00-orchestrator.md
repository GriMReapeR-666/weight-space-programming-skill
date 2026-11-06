# Universal WSP orchestrator playbook

## Mission

Given an authorized target model and a behavioral objective, discover its actual architecture, build a faithful structural map, diagnose a measurable failure/opportunity, formulate testable hypotheses, run bounded interventions, and deliver a validated reversible patch or a no-change report.

## Non-negotiable operating rules

1. Never mutate the original checkpoint. Work on a verified copy or immutable base plus patch overlay.
2. Never infer architecture from a marketing name alone. Inspect config, graph, tensor names/shapes, code, runtime and forward behavior.
3. Never assume `TransformerLens` supports a model. Use it only when its model implementation explicitly supports the target; otherwise use native framework hooks or an adapter.
4. Never infer a faulty neuron from weight magnitude, histogram shape, PCA direction or activation variance alone.
5. Separate activation ablation from parameter editing. Activation ablation is a causal probe; it does not itself establish a safe permanent weight patch.
6. Do not use test/holdout data to repeatedly tune candidate patches. Split diagnosis, search/validation and final holdout.
7. Reject improvements that are within evaluation noise, regress critical capabilities, exploit prompt artifacts, or fail to reproduce.
8. Require human approval before committing a patch to a production or shared model.
9. Never fabricate tool outputs, benchmark scores, paper findings or adapter support.
10. If the architecture is unknown, report unknowns and build a new adapter scaffold; do not force a false match.

## Phase A — authorization and asset inventory

- Confirm the user owns or is authorized to modify the target checkpoint.
- Identify format: PyTorch state dict, safetensors, GGUF, ONNX, TensorRT, Flax, JAX, Orbax, MLX, Core ML, TFLite, custom binary, distributed/sharded checkpoint, adapter-only artifact.
- Record hashes, file sizes, shard index, config, tokenizer/processor, vocabulary, quantization, dtype, tied weights, shared tensors, aliases and provenance.
- Record software versions, device, memory, deterministic settings, kernels and inference backend.
- Preserve license and model-card constraints. Do not bypass access controls or remove provenance.

## Phase B — architecture and computation graph discovery

Build a canonical graph with:
- nodes: modules, operations, parameter tensors, buffers, routing decisions, caches and modality encoders/decoders;
- edges: tensor dataflow, residual paths, attention, recurrence, skip connections, cross-modal links and parameter sharing;
- metadata: tensor shape/dtype/layout, fan-in/out, groups, heads, experts, receptive field, state dimensions, positional scheme, normalization, quantization and execution order.

Discover both declared architecture and observed runtime graph. Detect fused kernels, weight tying, dynamic control flow, MoE routing, recurrent state, cache state, custom ops and multimodal branches.

Produce a circuit map at several resolutions:
1. model/family overview;
2. stage/block/module topology;
3. tensor and parameter groups;
4. selected activation/feature summaries;
5. candidate circuit/direction evidence.

Never label a direction as a semantic circuit without causal tests and controls.

## Phase C — behavioral baseline

Define failure behavior operationally. Build:
- representative positive examples;
- negative/control examples;
- adversarial and boundary examples;
- task-specific quality metrics;
- safety/format/latency/memory metrics where relevant;
- repeated runs and random seeds for stochastic generation.

Record prompt templates, decoding parameters, tokenizer version, stop conditions, context length, cache settings and exact inference runtime. Reproduce the issue before inspecting weights.

## Phase D — diagnostic evidence

Use multiple independent signals:
- module output and input activation distributions;
- activation norms, variance, covariance, sparsity, saturation and dead-unit frequency;
- token/step-level traces and repetition/entropy measures;
- attention/routing statistics where meaningful;
- attribution or influence estimates;
- layer-wise perturbation sensitivity;
- activation patching, ablation and counterfactual comparisons;
- weight statistics and singular/eigenspectrum diagnostics;
- quantization error and numerical stability.

A zero-entropy output, high variance, outlier norm or repeated token is a symptom. It is not, by itself, proof of a particular weight defect.

## Phase E — hypothesis register

Each hypothesis must specify:
- observed failure and evidence;
- suspected component and why;
- predicted behavior if correct;
- intervention family;
- expected benefit and possible collateral damage;
- competing explanations;
- minimal experiment and stopping rule;
- confidence and evidence gaps.

Rank hypotheses by expected value, not by how plausible their story sounds.

## Phase F — intervention selection

Choose the least invasive compatible method:
1. inference configuration or prompt/runtime correction;
2. activation-space diagnostic intervention;
3. parameter-efficient delta or adapter;
4. sparse parameter patch;
5. low-rank or subspace patch;
6. structured pruning/sparsification;
7. quantization or dequantization change;
8. gradient-free parameter search;
9. conventional fine-tuning if direct editing is unsuitable.

A method is not WSP merely because it changes a model. Clearly label the method actually used.

## Phase G — bounded search and verification

- Generate candidates from a declared parameterization, not unconstrained full tensor rewriting.
- Search with fixed budgets, deterministic seeds where possible, early stopping and a separate validation set.
- Evaluate all objectives and regressions.
- Compare to unmodified baseline, random/no-op controls and relevant alternative patches.
- Estimate uncertainty across examples/seeds. Use paired comparisons when appropriate.
- Re-run the best candidate from a clean base to prove reproducibility.
- Validate serialization, reload, inference equivalence outside the intended behavior, and deployment runtime.

## Phase H — patch promotion

Create a patch artifact containing exact tensor keys, shapes, dtypes, coordinate/slice ranges, before/after hashes, delta encoding, base checkpoint hash, adapter version and application code. Apply transactionally to a copy. Verify post-application hashes and loadability. Keep rollback artifact.

## Phase I — final report

Return:
- architecture identification and confidence;
- supported/unsupported components;
- baseline and evaluation protocol;
- circuit map references;
- hypotheses tested and rejected;
- intervention type (activation probe vs direct weight patch vs optimizer training);
- patch coordinates/summary and hashes;
- score deltas with uncertainty and regressions;
- reproducibility instructions;
- known limitations and rollback command;
- recommendation: promote, continue research, or no change.
