# Capability-based operation routing

Route work by declared capabilities and preconditions, not by architecture names. An operation must specify: target component types; required capabilities; prohibited conditions; mathematical assumptions; invariants; expected quality/performance trade-offs; compute and memory estimates; staging and rollback; and validation gates.

Examples:
- Low-rank factorization requires compatible tensor structure and a meaningful rank/error criterion; it is not a generic operation for arbitrary state or quantized packed formats.
- Pruning requires a supported sparsity representation and a runtime that benefits from it; zeroing weights alone may not accelerate inference.
- Quantization requires calibration or a validated quantization-aware path, supported operators, and output-quality checks.
- Weight merging requires compatible architecture, parameter correspondence, tokenizer/processor conventions and training provenance; matching tensor shapes alone is insufficient.
- Adapter insertion requires a supported insertion point and forward-path semantics.
- Distillation requires a teacher/student interface and a task-appropriate objective; it is not a direct weight edit.
- Classical estimators may support feature selection, tree pruning, calibration or hyperparameter optimization rather than neural weight-space operations.

If preconditions are unknown, the operation is blocked pending a diagnostic. If no meaningful weight-space operation exists, recommend an appropriate model-level or data-level optimization instead.
