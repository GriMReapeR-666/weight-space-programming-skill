# Unknown model discovery protocol

## Phase A — artifact forensics (read-only)
Inventory files, hashes, sizes, shards, manifests, config, metadata, tokenizer/processor assets, model code, licenses, quantization metadata and custom operators. Identify whether the artifact is a neural checkpoint, executable graph, classical estimator, optimizer state, adapter, or composite package.

## Phase B — safe source inspection
Read configuration and source without importing untrusted code. Identify entry points, module construction, parameter registration, custom ops, forward signatures, preprocessing and output semantics. If execution is necessary, use an isolated, network-disabled environment with resource limits and no write access to originals.

## Phase C — structural probes
Build a component graph; enumerate parameters and buffers; record shapes, dtypes, layouts, aliases, tied/shared parameters, sparsity, quantization, sharding and parameter ownership. For graph or procedural models, capture operators and state transitions rather than forcing a layer abstraction.

## Phase D — behavioral probes
Only after trust review, run tiny bounded inputs. Record input/output contracts, deterministic behavior, statefulness, gradients, numerical stability, memory and latency. Use synthetic inputs first. Do not infer semantic quality from random inputs.

## Phase E — adapter selection or scaffold
Match by capabilities and structural signatures. If no adequate adapter exists, create a plugin scaffold. Compose adapters for hybrid models. Preserve unknown components as opaque nodes with explicit boundaries.

## Phase F — conformance and promotion
Run shape, dtype, device, serialization, reload, numerical parity, determinism, gradient, distributed and behavioral tests as applicable. Promote capability levels individually. Keep unverified capabilities unknown/unsupported.

## Output
Produce `discovery.json`, component graph, evidence log, uncertainty register, adapter match report, unsupported capability list, and a no-mutation baseline. Never modify weights during discovery.
