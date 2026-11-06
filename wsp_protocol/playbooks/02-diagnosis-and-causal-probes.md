# Diagnosis, activation tracing and causal probes

## Reproduce first

A model that loops may be affected by prompt formatting, stop-token handling, tokenizer mismatch, sampling, KV-cache bugs, context truncation, quantization, speculative decoding, or runtime kernels. Reproduce under a known-good reference runtime before attributing it to weights.

## Activation capture

Instrument module inputs/outputs using the native framework where possible. Capture only selected layers and examples. Use streaming aggregates or bounded samples; avoid retaining all activations in RAM.

For each selected module record:
- call index and sequence/time position;
- input/output shape and dtype;
- norm and finite-value summary;
- per-token/per-step summary if relevant;
- routing/expert choice or attention metadata if available;
- association to the exact input example and model hash.

Do not assume that a module is called once. Recurrent, autoregressive, diffusion and iterative models may call components many times.

## Ablation types

1. **Activation ablation:** replace selected intermediate activation with zero, mean, noise, or a matched-norm control.
2. **Activation scaling:** multiply a selected activation or subspace projection by a bounded coefficient.
3. **Activation patching:** replace an activation from a clean/control run at a corresponding location.
4. **Parameter ablation:** temporarily zero or mask a parameter slice in a disposable copy.
5. **Parameter scaling:** scale a selected parameter group or low-rank component.
6. **Parameter delta:** apply an explicitly stored additive or structured delta.

Report these separately. Activation ablation is often the safer first causal probe.

## Causal interpretation

For each intervention compare:
- target failure examples;
- clean examples;
- matched random component/direction;
- sham/no-op intervention;
- multiple seeds or samples;
- dose-response curve;
- restoration/reversal where possible.

A candidate is more credible if the predicted behavior changes selectively, the effect is reproducible, and matched controls do not produce the same result.

## Repetition diagnostics for autoregressive language models

Track:
- repeated n-gram rates at multiple n;
- token entropy and distinct-token ratio;
- cycle length and recurrence of hidden states/logits;
- EOS/stop-token probability and actual stop handling;
- per-step logits, selected token and decoding parameters;
- attention/cache growth and position IDs.

Do not equate low token entropy with a faulty layer. Low entropy can be appropriate for deterministic output.
