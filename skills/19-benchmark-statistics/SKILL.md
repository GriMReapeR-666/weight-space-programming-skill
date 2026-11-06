---
name: 19-benchmark-statistics
description: Rigorous statistical evaluation, perplexity measurement, hypothesis testing, and confidence intervals for weight-space changes.
---

# 19 Benchmark Statistics & Evaluation Rigor

## Purpose
In weight-space programming, claiming an improvement or claiming "zero regression" requires empirical proof. This skill enforces statistical testing, confidence interval reporting, protected benchmark gates, and perplexity verification for every model mutation.

---

## Evaluation Metrics & Protocols

### 1. Perplexity (PPL) Verification
For language and sequence models, perplexity over evaluation tokens $X = (x_1, \dots, x_N)$:
$$\text{PPL}(X) = \exp\left( -\frac{1}{N} \sum_{i=1}^N \log P_\theta(x_i | x_{<i}) \right)$$
- **Acceptance Threshold**: Any weight edit $\theta \to \theta'$ must satisfy:
  $$\Delta \text{PPL} = \text{PPL}_{\theta'} - \text{PPL}_\theta \le +0.10$$
  Any larger degradation triggers an immediate automatic rollback.

### 2. Paired Hypothesis Testing (Wilcoxon Signed-Rank / t-test)
When comparing baseline accuracy $M_{base}$ and modified accuracy $M_{mod}$ across $N$ test items:
1. Compute pairwise differences $d_i = M_{mod}(x_i) - M_{base}(x_i)$.
2. Calculate $p$-value under null hypothesis $H_0: \mathbb{E}[d] \le 0$.
3. Reject $H_0$ only if $p < 0.05$ with $95\%$ bootstrap confidence interval excluding zero.

```python
import numpy as np
from scipy import stats

def evaluate_significance(scores_base, scores_mod):
    t_stat, p_val = stats.ttest_rel(scores_mod, scores_base)
    diff = np.array(scores_mod) - np.array(scores_base)
    ci_low, ci_high = np.percentile(
        [np.mean(np.random.choice(diff, size=len(diff), replace=True)) for _ in range(1000)],
        [2.5, 97.5]
    )
    return {
        "mean_delta": float(np.mean(diff)),
        "p_value": float(p_val),
        "ci_95": [float(ci_low), float(ci_high)],
        "statistically_significant": bool(p_val < 0.05 and ci_low > 0)
    }
```

### 3. Protected Invariant Benchmark Gate
Always evaluate against non-negotiable benchmark buckets:
- **Core Reasoning**: GSM8K / ARC subsets.
- **Safety / Refusal Alignment**: Harmful request refusal rate must remain $\ge 99\%$ of baseline.
- **Syntactic Validity**: JSON / code output parser success rate.

---

## Tool Protocol
Record all validation results in `experiment_ledger.py`:
```bash
python wsp_protocol/tools/experiment_ledger.py log \
    --event benchmark_eval \
    --model models/patched_model.pt \
    --notes "Evaluation: Delta PPL = +0.02, accuracy +3.4% (p=0.012)"
```
