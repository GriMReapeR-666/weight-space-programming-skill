# Claims and evidence discipline

The supplied background narrative mentions a purported MIT CSAIL study, named frontier model capabilities, particular Qwen repetition-loop surgery, exact modified-weight counts, benchmark deltas, and a hand-authored Atari policy. This bundle does not independently establish those claims.

Before using any such claim in a research report:
1. locate the primary paper or official technical report;
2. verify author, institution, date and version;
3. inspect the exact model checkpoint and code used;
4. reproduce the stated evaluation protocol;
5. distinguish direct weight edits from activation interventions, adapters, fine-tuning and decoding/runtime changes;
6. report failed reproductions and deviations.

Do not use a secondary blog, Reddit post, video title, social post or encyclopedia entry as sole proof of a research result. Links supplied by a user are leads, not validation.

General technical caution: direct weight edits can change behavior without backpropagation, but causal localization and reliable improvement are empirical tasks. No universal guarantee exists that an LLM can infer semantics from raw weights alone.
