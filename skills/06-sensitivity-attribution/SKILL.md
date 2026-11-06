---
name: 06-sensitivity-attribution
description: Causal ablation probes, Taylor expansion sensitivity, Fisher Information estimation, and identifying high-impact layers.
---

# 06 Sensitivity Attribution & Causal Probing

## Purpose
Modifying arbitrary weights blindly causes catastrophic forgetting. Sensitivity attribution identifies *which* layers, attention heads, or MLP neurons govern a specific capability, and which parameters can be safely edited without degrading core competencies.

---

## Methodologies

### 1. Causal Ablation Interventions
Ablate targeted units during execution and measure target metric drop $\Delta M$:
- **Zero Ablation**: Set activation $a_i \leftarrow 0$.
- **Mean Replacement**: Set $a_i \leftarrow \mu_i$ (preserves activation baseline bias).
- **Gaussian Noise**: Set $a_i \leftarrow \mathcal{N}(\mu_i, \sigma_i^2)$ (disrupts information content).
- **Freeze / Identity**: Replace layer transformation with a skip-connection passthrough.

### 2. First-Order Taylor Expansion Importance
Estimate parameter saliency $I(W)$ with respect to loss $\mathcal{L}$:
$$I(w_{ij}) = \left| w_{ij} \cdot \frac{\partial \mathcal{L}}{\partial w_{ij}} \right|$$
This estimates the loss deviation if weight $w_{ij}$ were zeroed, without requiring second derivatives.

### 3. Empirical Fisher Information Matrix (Diagonal)
Approximates the local curvature of the loss surface:
$$F_{ii} = \mathbb{E}_{x \sim \mathcal{D}} \left[ \left( \frac{\partial \log p(y|x, \theta)}{\partial \theta_i} \right)^2 \right]$$
High Fisher diagonal values indicate critical parameters that must not be altered during merges or edits.

---

## Tool Execution

Run `ablation_runner.py` to systematically probe model layers:
```bash
# Test zero-ablation across intermediate layers
python wsp_protocol/tools/ablation_runner.py \
    --model models/target.pt \
    --mode zero \
    --layers model.layers.10,model.layers.11,model.layers.12 \
    --output ablation_results.json
```

Use `patch_delta.py` to non-destructively test a targeted sensitivity edit:
```bash
python wsp_protocol/tools/patch_delta.py \
    --checkpoint models/target.pt \
    --patch patch_spec.json \
    --output models/probed_model.pt
```

---

## Deliverables
- A sensitivity ranking table sorting layers by degradation impact:
  ```csv
  layer,param_count,ablation_mode,metric_baseline,metric_ablated,relative_drop
  model.layers.0,17500000,zero,0.92,0.91,0.0108
  model.layers.16,17500000,zero,0.92,0.34,0.6304
  ```
- Ledger recording of high-sensitivity bottlenecks before any weight-space patching.
