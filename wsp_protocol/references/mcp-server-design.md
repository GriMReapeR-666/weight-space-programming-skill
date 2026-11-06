# MCP server design for WSP

## Server boundary

The MCP server is a thin, typed interface to a local Python engine. It does not make the LLM the numerical execution engine. The server exposes bounded operations and returns compact evidence objects plus artifact references.

## Suggested tool groups

1. Inventory: checkpoint, runtime, architecture and environment.
2. Observe: graph, tensor summaries, activation capture, generation trace.
3. Diagnose: baseline evaluation, ablation, sensitivity, control comparisons.
4. Search: candidate parameterization, bounded optimizer, evaluation.
5. Patch: create, validate, apply-to-copy, verify, rollback.
6. Audit: provenance, hashes, experiment ledger and final report.

## Resource controls

- max model size / bytes;
- max activation bytes;
- max examples and sequence length;
- max GPU seconds and CPU seconds;
- max candidate count;
- max changed parameter count / fraction;
- max absolute and relative delta;
- per-tool timeout;
- cancellation support;
- concurrency lock per target checkpoint.

## Security

Run tools in a sandbox with explicit model/data directory allowlists. Treat model code, config files, tokenizer files, datasets and tool outputs as untrusted input. Do not execute arbitrary remote code by default. Require explicit opt-in for custom model code. Disable network access unless needed and approved. Do not expose secrets or unrelated files to the agent.

## Result envelope

Every tool response should include:
`status`, `summary`, `model_hash`, `adapter_id`, `adapter_version`, `tool_version`, `metrics`, `warnings`, `artifact_refs`, `reproducibility`.

## Important

This document is a design contract, not a claim that an MCP server is already implemented in this bundle.
