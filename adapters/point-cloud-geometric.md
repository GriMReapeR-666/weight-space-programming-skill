# Adapter specification: Point clouds, voxels, meshes and geometric/equivariant networks

## Scope
This adapter covers discovery and engineering for **Point clouds, voxels, meshes and geometric/equivariant networks**. It is a specification/template until implemented and tested against named model classes and versions.

## Identify before loading or editing
- Concrete architecture, implementation, version and source revision.
- Checkpoint/container format; parameter objects versus buffers/state; aliases and shards.
- Task objective, input representation, preprocessing and output interpretation.
- Hardware/runtime requirements and any custom kernels or compiled artifacts.

## Family-specific inspection
Map each parameter to its mathematical role and computational component. Record shapes, layout, dtype, constraints, state, sharing, and invariances. Inspect the family-specific invariants before choosing any operation.

## Candidate diagnostics
Start with task failures and baseline metrics. Add activations, gradients, sensitivity, ablations and parameter statistics only where the implementation exposes valid hooks. Avoid assuming every parameter is a dense matrix.

## Candidate interventions
Declare each operation as supported, unsupported or conditional for the exact implementation. Require explicit preconditions, a reversible candidate, and a task-level evaluation. Do not inherit generic methods from another family merely because tensor shapes look similar.

## Required tests
- Metadata and parameter inventory fixture.
- Load/serialize round-trip and source-hash preservation.
- Input/output shape and smoke-inference tests.
- Family-invariant tests (for example routing, recurrence, equivariance, chronology, invertibility, action constraints or physical constraints, as applicable).
- Baseline-versus-candidate task evaluation and deployment parity.

## Maturity declaration
Report L0-L6 per capability using `references/universal-adapter-contract-v2.md`. State exact tested versions and known limitations. Until tests pass, this file is a design template, not executable support.
