# Surgical patch format, transaction, integrity and rollback

## Patch principles

A patch is a declarative delta against a specific immutable base. It is not a rewritten checkpoint and it must never silently apply to a different model revision.

## Patch manifest requirements

- schema version;
- patch ID and creation timestamp;
- base model identifier and cryptographic hash;
- tensor key and exact shape/dtype;
- operation (`add`, `scale`, `mask`, `replace_slice`, `low_rank_add`, adapter coefficient);
- coordinate/slice specification;
- delta payload URI and hash;
- numeric bounds and expected before/after summaries;
- adapter name/version and tool versions;
- authoring agent and human approval state;
- evaluation report reference;
- rollback reference.

## Transactional application

1. Verify base hash and tensor metadata.
2. Verify patch payload hash and schema.
3. Load into a disposable output path.
4. Apply to a copy using explicit dtype/device behavior.
5. Check finite values, shape invariants, tied/shared weight invariants and allowed bounds.
6. Save to a temporary file or directory.
7. Reload the result with the target runtime.
8. Verify post-application hash and expected changed-tensor set.
9. Atomically promote the output copy only after all checks pass.
10. Preserve base and rollback artifact.

If any step fails, discard the candidate and report failure. Never partially overwrite the source.

## Quantized models

Do not assume a quantized tensor can be edited like a floating-point tensor. Depending on format, a patch may require dequantization/requantization, editing a supported adapter overlay, or conversion to a training-capable checkpoint. Record quantization error and validate the exact deployed runtime. GGUF is generally an inference/distribution format; use an appropriate source checkpoint for training when available.

## Rollback

Rollback means selecting the untouched base artifact or removing the overlay, not attempting to reverse an untracked floating-point operation. Verify the restored hash.
