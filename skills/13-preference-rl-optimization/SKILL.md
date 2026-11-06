---
name: 13-preference-rl-optimization
description: Preference optimization in weight space, Direct Preference Optimization (DPO), KTO, and reward model weight fusion.
---

# 13 Preference & RL Optimization in Weight Space

## Purpose
Aligning models to human preferences or safety specifications often involves Reinforcement Learning from Human Feedback (RLHF) or Direct Preference Optimization (DPO). In weight-space programming, we analyze preference deltas as directional alignment vectors $\tau_{align}$ that can be scaled, negated, or merged across models.

---

## Formulations & Weight-Space Principles

### 1. DPO Implicit Reward & Alignment Vectors
In DPO, the optimization directly increases the log-ratio of chosen ($y_w$) versus rejected ($y_l$) completions:
$$\mathcal{L}_{DPO}(\theta; \theta_{ref}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

The resulting delta:
$$\tau_{dpo} = \theta_{dpo} - \theta_{ref}$$
acts as a pure **alignment vector**.

### 2. Alignment Vector Scaling & Negation
- **Strengthen Alignment**: $\theta_{new} = \theta_{base} + \gamma \cdot \tau_{dpo}$ with $\gamma > 1.0$.
- **Ablate / De-bias**: $\theta_{debiased} = \theta_{base} - \gamma \cdot \tau_{bias}$.
- **Cross-Model Alignment Transfer**: Transfer $\tau_{dpo}$ from an aligned model $M_A$ to a sibling base model $M_B$ of identical architecture:
  $$\theta_{B, aligned} = \theta_B + \alpha \cdot (\theta_{A, dpo} - \theta_A)$$

---

## Tool Execution

Merge alignment vectors into unaligned bases using SLERP or TIES:
```bash
python -m weight_space_programming_skill merge slerp \
    models/base.safetensors models/dpo_aligned.safetensors \
    --alpha 0.6 \
    --output models/merged_aligned.safetensors
```

Log alignment parameters:
```bash
python wsp_protocol/tools/experiment_ledger.py log \
    --event alignment_merge \
    --model models/merged_aligned.safetensors \
    --notes "Merged DPO alignment vector with alpha=0.6"
```
