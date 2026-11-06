# Tool API contract for agentic WSP

The agent should call deterministic tools. Tool names below are conceptual API contracts; implement them in the local orchestrator/MCP server and do not pretend they exist until implemented.

## Read-only tools

- `inspect_checkpoint(path, format_hint?)` → artifact metadata, hashes, config and shard inventory.
- `inspect_architecture(model_ref)` → graph/module/tensor topology and adapter capabilities.
- `summarize_tensor(model_ref, tensor_key, slice?, summary_profile?)` → numeric summary and exact artifact reference.
- `build_circuit_map(model_ref, examples_ref, target_modules?)` → graph plus activation evidence.
- `capture_activations(model_ref, examples_ref, module_selectors, limits)` → bounded activation summaries and optional binary captures.
- `evaluate_baseline(model_ref, dataset_ref, protocol)` → metrics, raw outputs and run metadata.
- `trace_generation(model_ref, prompt_ref, decode_config)` → token, logits/entropy/repetition and runtime trace.
- `get_patch_status(patch_id)` → patch lifecycle state.

## Diagnostic tools

- `ablate_activation(model_ref, examples_ref, module_selector, intervention, controls)` → causal probe report.
- `ablate_parameters(model_ref, examples_ref, tensor_selector, mask_spec, controls)` → disposable parameter probe.
- `compare_directions(model_ref, examples_ref, direction_ref, matched_controls)` → dose-response and control comparison.
- `sensitivity_scan(model_ref, examples_ref, component_selector, budget)` → ranked sensitivity estimates.

## Search and patch tools

- `propose_candidate(parameterization, search_space, seed)` → candidate spec only.
- `evaluate_candidate(model_ref, candidate_ref, dataset_ref, budget)` → metrics.
- `optimize_candidates(search_config)` → bounded search ledger and finalists.
- `create_patch(base_hash, candidate_ref, metadata)` → immutable patch artifact.
- `validate_patch(patch_ref, target_model_ref)` → compatibility and safety report.
- `apply_patch_to_copy(patch_ref, output_path)` → transactional output copy.
- `rollback_patch(patched_ref, base_ref)` → restore verified base.

## Tool safety

- Every tool validates path allowlists, size limits, resource budgets and user authorization.
- Read-only tools never mutate.
- Patch tools never overwrite source artifacts.
- Candidate execution has time, memory and GPU budgets.
- Tool outputs include `status`, `warnings`, `evidence_refs`, `model_hash`, `tool_version`.
- Any error returns a structured error; no fabricated fallback result.
