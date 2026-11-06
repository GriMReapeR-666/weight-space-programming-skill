# Future-architecture compatibility

## Contract
The framework is open-world: the family catalogue is a discovery aid, never an allow-list. A model is supported when a compatible adapter implements the required capabilities and passes the conformance suite. Unknown architectures enter a safe discovery-and-scaffold workflow; they are not silently treated as Transformers.

## Required lifecycle
1. Identify artifact format, framework, model code, config, tokenizer/processor, checkpoint shards, quantization, custom ops and execution constraints.
2. Inspect source/config and construct a component graph. Record uncertainty and evidence for every inferred component.
3. Match existing adapters by capabilities, not name alone. A model may compose several adapters.
4. If no match exists, generate a new adapter scaffold from the plugin SDK and run read-only probes.
5. Implement only capabilities supported by evidence. Mark unsupported methods explicitly.
6. Run adapter conformance, numerical parity, behavioral baseline, safety and regression tests.
7. Register the adapter with version, framework/model-code compatibility, tested environments and provenance.

## No universal assumptions
Never assume all models have layers, attention, a tokenizer, floating-point dense matrices, gradients, a single checkpoint, or a forward method with the same signature. Discover these properties. Support graph-based, procedural, recurrent, continuous-time, sparse, quantized, distributed, hybrid and non-neural parameterized systems through declared capabilities.

## Compatibility levels
- L0 Identify: format/config/code recognized; no execution.
- L1 Inspect: parameter/component inventory and provenance.
- L2 Evaluate: reproducible inference/evaluation and baseline.
- L3 Train: verified training or fine-tuning path.
- L4 Transform: verified architecture-aware weight transformation.
- L5 Deploy: export, runtime parity and deployment validation.
A level is earned per model + adapter + environment tuple, not globally.

## Future-proofing policy
When a new family appears, add a plugin and tests; do not rewrite the core. Keep adapter API versioned and backward-compatible where possible. Unknown plugin code runs in an isolated process with read-only access by default. Never auto-execute model-provided code or custom kernels without explicit trust and sandbox review.
