# Universal adapter contract (v2)

An adapter is a tested bridge between the universal workflow and one concrete implementation. It must not infer support merely from a model name.

## Required identity
- `family`, `architecture`, `implementation`, `version/revision`, `framework`, `checkpoint_format`.
- Source/config evidence and confidence; unresolved facts listed explicitly.
- Input/output modality, task, preprocessing/processor and postprocessing.

## Required inspection interface
- `load_metadata(path)` — safe metadata/config inspection; no arbitrary code execution by default.
- `enumerate_parameters()` — stable names, shapes, dtypes, storage formats, aliases/ties, trainable flags.
- `enumerate_buffers_and_state()` — non-parameter state, quantization scales, running stats, caches and other buffers.
- `map_parameter_to_component(name)` — module/functional role and graph location.
- `trace_io(example)` — validated input/output shapes and execution graph where available.
- `baseline(evaluation_spec)` — deterministic task metrics and resource metrics.

## Required capability declarations
For each capability return `supported`, `unsupported`, or `conditional`, plus exact conditions:
- statistics / spectra / activation capture / gradient capture / sensitivity;
- full fine-tuning / partial fine-tuning / PEFT;
- pruning / quantization / low-rank factorization / distillation;
- merge / interpolation / adapter composition;
- export / conversion / runtime parity.

## Required mutation interface
- `plan_transform(spec)` returns exact target mapping, equations/algorithm, preconditions, expected side effects and rollback plan.
- `apply_transform(spec, destination)` writes only to a new destination, never overwrites source.
- `validate_transform(before, after, evaluation_spec)` runs structural, numerical, behavioral and deployment checks.
- `export(destination, target_runtime)` verifies reload and output parity.

## Universal safety invariants
- Original artifact is immutable; hash before and after.
- Preserve tied/shared parameter relationships unless the transformation explicitly handles them.
- Distinguish parameters, buffers, optimizer state, quantization metadata and non-tensor model objects.
- Never apply matrix assumptions to convolution kernels, embeddings, normalization vectors, tree nodes, packed quantized blocks or opaque state without a declared mapping.
- Check dtype/device/layout, finite values, shape constraints and model-specific invariants.
- Every edit has an isolated candidate, rollback and task-specific acceptance criteria.
- A conversion that loads successfully is not necessarily functionally equivalent.

## Adapter maturity levels
- L0 Catalogue: family is listed; no executable support.
- L1 Read-only: metadata and parameters can be enumerated.
- L2 Structural: graph mapping and smoke inference pass.
- L3 Diagnostic: baseline, activation/gradient/sensitivity tools validated.
- L4 Transform: one or more mutations implemented with tests.
- L5 Training: fine-tuning or adaptation path validated.
- L6 Export: target runtime conversion and parity validated.

Report maturity per capability, not one blanket support badge. Include test fixtures and exact tested versions.
