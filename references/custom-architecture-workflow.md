# Custom and emerging architecture adapter workflow

Use this when a model is absent from the family catalogue or has novel components.

1. **Contain**: work on a copy in a restricted environment. Do not execute untrusted checkpoint code or custom operators without review.
2. **Collect evidence**: config, model card, paper, source revision, dependency lock, checkpoint metadata and runtime logs.
3. **Describe graph**: nodes/modules, tensor shapes, parameter/buffer objects, data flow, recurrent/stateful edges, conditional paths and custom kernels.
4. **Describe semantics**: what each parameter controls, constraints, invariances, normalization, symmetries and objective coupling.
5. **Establish a baseline**: representative inputs, deterministic settings where possible, task metrics and resource profile.
6. **Build read-only adapter**: metadata, parameter inventory, component mapping, forward smoke test.
7. **Create tests**: shape tests, serialization round-trip, deterministic smoke test, gradient test where relevant, invariance tests and golden outputs.
8. **Add one capability at a time**: diagnostic first, then a narrowly scoped transform, then training/export.
9. **Validate**: compare baseline/candidate on task, stress, out-of-distribution and deployment suites.
10. **Publish support claim**: maturity level, tested versions, supported methods, limitations and reproducible fixtures.

If semantics cannot be established, classify as `unknown/custom-unresolved`; do not perform weight-space mutation. It is acceptable to provide read-only inventory and request implementation details.
