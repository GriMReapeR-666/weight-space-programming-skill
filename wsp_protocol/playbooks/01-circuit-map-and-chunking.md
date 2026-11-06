# Circuit map, spatial abstraction and context-safe chunking

## Purpose

Convert large numerical structures into compact, machine-generated evidence that an agent can reason over without reading raw tensor dumps.

## Required layers of abstraction

**L0 — Artifact map:** files, shards, formats, model config, tokenizer/processor, quantization and hashes.

**L1 — Graph map:** module hierarchy, computation graph, tensor connections, shared/tied weights, residual and modality paths.

**L2 — Parameter map:** tensor key, shape, dtype, layout, parameter count, fan-in/out, norm summaries, sparsity, quantization metadata and shard location.

**L3 — Runtime map:** observed module calls, input/output shapes, invocation counts, routing decisions, cache/state shape, activation summaries.

**L4 — Behavioral map:** examples, token/time traces, failures, metrics, relevant modules and controlled interventions.

**L5 — Hypothesis map:** candidate directions/components with evidence, controls, confidence and next experiment.

## Chunking rules

- Chunk by semantic module or adapter-defined structural group, not arbitrary flat token counts.
- Preserve tensor identity, coordinate convention, shape, dtype, layout and source hash in every chunk.
- Use deterministic chunk IDs and stable ordering.
- Store numeric arrays in binary artifacts; send summaries and references to the LLM.
- Include quantiles, norms, sparsity, finite-value counts, top-k coordinates and low-resolution projections only when useful.
- Never imply that PCA axes or high-magnitude coordinates are intrinsically meaningful.
- Do not downsample away rare but important structures without preserving a way to inspect exact source coordinates.
- Make chunks addressable by `artifact_hash + tensor_key + slice + transform_version`.

## Suggested summary fields

`tensor_key`, `shape`, `dtype`, `layout`, `numel`, `finite_fraction`, `min`, `max`, `mean`, `std`, `l1_norm`, `l2_norm`, `zero_fraction`, `quantiles`, `top_abs_coordinates`, `singular_value_summary`, `activation_summary_ref`, `graph_neighborhood`, `source_hash`.

## PCA and subspace caveats

PCA can summarize covariance structure, but:
- components depend on centering/scaling and sampled data;
- signs are arbitrary;
- degenerate subspaces can rotate;
- weight-space PCA and activation-space PCA answer different questions;
- a principal direction is not automatically a defect direction.
Always compare with random directions, matched-norm controls, alternative samples and direct interventions.

## Circuit map output

The map must be JSON/graph data plus a concise Markdown summary. It must point back to exact tensor/module locations and raw evidence. The LLM should propose queries against the map, not invent coordinates.
