# Evaluation, statistical confidence and acceptance gates

## Dataset separation

- **Diagnostic split:** inspect failures and develop hypotheses.
- **Search split:** choose candidate parameters.
- **Validation split:** compare finalists and tune stopping decisions.
- **Final holdout:** one-time estimate after selection. Keep it isolated.

Prevent contamination by prompt templates, near-duplicates, synthetic variants and benchmark answer leakage.

## Evaluation dimensions

Measure task quality plus relevant side effects:
- accuracy, exact match, pass@k, calibration or task-specific loss;
- instruction following and format validity;
- robustness and out-of-distribution behavior;
- refusal/safety behavior where relevant;
- latency, throughput, memory, energy and context scaling;
- generation repetition, truncation and completion rate;
- multimodal alignment/grounding where relevant;
- numerical stability and loadability.

## Statistics

Use paired comparisons when examples are shared. Report sample size, seeds, confidence intervals or bootstrap intervals, variance, and effect size. For stochastic models, repeat runs. Avoid treating one successful sample as a benchmark result.

## Acceptance gates

A patch is promotable only if:
- baseline is reproducible;
- target behavior improves beyond expected noise;
- critical non-target capabilities remain within predeclared regression limits;
- final holdout is not used for iterative tuning;
- candidate is reproducible from clean base;
- artifact loads and runs in deployment runtime;
- provenance, hashes and rollback are complete;
- required human approval is obtained.

If not, report no-go or inconclusive. A no-change result is a valid engineering outcome.
