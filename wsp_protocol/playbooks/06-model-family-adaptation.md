# Model-family adaptation and unknown architectures

## Universal adapter responsibilities

Every adapter must expose capabilities rather than pretending every model has transformer layers or neurons.

Required adapter operations:
- `identify_artifact`
- `inspect_config`
- `enumerate_parameters`
- `describe_graph`
- `map_parameter_to_module`
- `capture_runtime_trace`
- `capture_activations` (if supported)
- `run_intervention` (only declared intervention types)
- `evaluate`
- `export_patch`
- `validate_patch`
- `apply_patch_to_copy`
- `rollback`
- `self_test`

Each operation returns structured data with provenance, warnings, limitations and explicit support status.

## Capability declaration

Examples:
- `has_attention`
- `has_recurrence`
- `has_expert_routing`
- `has_convolution`
- `has_normalization`
- `has_shared_parameters`
- `has_dynamic_graph`
- `has_tokenizer`
- `has_multimodal_processor`
- `supports_activation_hooks`
- `supports_parameter_patch`
- `supports_low_rank_delta`
- `supports_quantized_patch`
- `supports_deterministic_eval`

Operations are enabled only when required capabilities and conformance tests are present.

## Unknown architecture protocol

1. Identify framework and serialization.
2. Inspect config and model source where available.
3. Enumerate tensors and buffers.
4. Trace one minimal forward pass with representative inputs.
5. Infer graph topology from observed calls and tensor shapes.
6. Identify custom ops, fused modules, shared weights and hidden state.
7. Create an unknown-family adapter manifest.
8. Implement read-only inspection first.
9. Add activation tracing and evaluation.
10. Add one intervention at a time.
11. Run conformance and regression tests.
12. Mark adapter as experimental until validated on multiple checkpoints.

## Hybrid models

Compose adapters by graph subcomponent. Example: a vision encoder + language decoder + cross-attention + audio pathway. Track cross-adapter tensor boundaries and shared parameters. Avoid duplicate patching of tied/shared weights.

## Never assume

- every model is a Transformer;
- every tensor called `weight` is a dense matrix;
- every hidden dimension is a neuron;
- every attention head can be independently edited;
- every checkpoint is fully materialized in one file;
- every model can be loaded with `transformers`;
- every model exposes activations through a generic hook;
- every quantized format supports in-place editing.
