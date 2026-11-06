# Adapter Plugin SDK (normative)

An adapter is a versioned plugin implementing `describe()`, `detect(artifact)`, `inspect(model, context)`, `capabilities()`, `baseline(model, context)`, and optional operation handlers. Keep the core orchestration independent of PyTorch, JAX, TensorFlow, ONNX, MLX, GGUF, custom runtimes and future frameworks.

## Minimum metadata
- `adapter_id`: reverse-DNS or stable namespaced identifier.
- `api_version`: semantic version of this SDK contract.
- `adapter_version`: plugin version.
- `supported_artifacts`: formats and constraints.
- `frameworks`: frameworks/runtime versions tested.
- `architecture_signatures`: positive and negative detection evidence.
- `capabilities`: explicit capability declarations with status `supported`, `partial`, `unsupported`, or `unknown`.
- `operations`: operation ID, preconditions, parameterization, invariants, rollback strategy and validation.
- `safety`: trust level, code execution needs, network needs, write scopes.
- `tests`: conformance and regression test references.

## Required behavior
- Detection returns confidence, evidence and competing matches; never only a guessed family label.
- Inspection is read-only and returns a normalized component graph plus raw framework-specific metadata.
- Every mutation is a proposal first, then a staged operation on a copy.
- Every operation declares expected tensor/parameter shapes, dtype/device behavior, invariants, numerical tolerance and rollback.
- Unsupported operations fail closed with actionable reasons.
- No operation may mutate the source checkpoint in place.
- Deterministic operations record seeds, software versions, input hashes and output hashes.

## Capability vocabulary
`load`, `inspect_config`, `inspect_graph`, `enumerate_parameters`, `enumerate_buffers`, `trace_forward`, `run_inference`, `extract_activations`, `compute_gradients`, `train`, `fine_tune`, `quantize`, `dequantize`, `prune`, `sparsify`, `factorize`, `merge`, `distill`, `convert`, `export`, `distributed_shards`, `custom_kernels`, `runtime_parity`, `behavioral_eval`, `safety_eval`.

Capabilities can be extended using namespaced IDs. A capability declaration does not prove correctness; it must link to tests and a tested environment.

## Adapter composition
Use multiple adapters when a system contains heterogeneous subsystems, such as a vision encoder + language decoder + audio encoder + diffusion decoder. Define component boundaries, shared parameters, routing, preprocessing/postprocessing and end-to-end tests. Do not apply a component-local transformation if it breaks cross-component alignment.

## Reference pseudocode
```python
class Adapter:
    api_version = "1.0"
    def describe(self) -> dict: ...
    def detect(self, artifact) -> dict: ...  # confidence + evidence
    def inspect(self, model, context) -> dict: ...  # normalized graph
    def capabilities(self) -> dict: ...
    def baseline(self, model, context) -> dict: ...
    def plan_operation(self, operation_id, model, request) -> dict: ...
    def apply_staged(self, plan, source, destination) -> dict: ...
    def validate(self, source, destination, plan, context) -> dict: ...
```
The pseudocode is a contract illustration, not a drop-in implementation. Each plugin must provide a real implementation and tests.
