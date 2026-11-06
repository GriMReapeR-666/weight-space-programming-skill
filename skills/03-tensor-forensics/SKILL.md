---
name: 03-tensor-forensics
description: Inspect tensor shapes, dtypes, distributions, norms and integrity.
---
# 03 Tensor Forensics

## Objective
Inspect tensor shapes, dtypes, distributions, norms and integrity.

## Required procedure
1. Inspect the actual model implementation and task.
2. Establish preconditions and compatibility.
3. Record source evidence and unknowns.
4. Propose the smallest testable operation.
5. Implement with deterministic, documented tools.
6. Add unit/smoke tests and expected outputs.
7. Record resource needs, risks and rollback.
8. Validate against baseline and protected tests.
9. Report facts, hypotheses, results and limitations separately.

## Required output
Produce a concise technical report, machine-readable metadata where appropriate, reproducible commands, tests, and a list of unresolved issues.

## Safety
Never assume compatibility from a model-family label. Never modify the only copy of a checkpoint. Never claim improvement without measured evidence.
