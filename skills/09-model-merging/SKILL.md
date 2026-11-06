---
name: 09-model-merging
description: Weight-space model fusion using Linear Task Arithmetic, SLERP, TIES-Merging, and DARE algorithms.
---

# 09 Model Merging & Weight-Space Fusion

## Purpose
Model merging combines the capabilities of two or more fine-tuned checkpoints derived from a shared base model without running expensive retraining or data curation. This skill details the mathematical formulas, algorithms, trade-offs, and tool commands for Linear Task Arithmetic, SLERP, TIES-Merging, and DARE.

---

## Merging Algorithms & Formulations

### 1. Task Arithmetic (Linear Interpolation)
Given base model $W_{base}$ and fine-tuned model $W_{ft}$, compute task vector $\tau = W_{ft} - W_{base}$:
$$W_{merged} = W_{base} + \alpha \cdot \tau = (1 - \alpha) W_{base} + \alpha W_{ft}$$

### 2. Spherical Linear Interpolation (SLERP)
Linear interpolation fails on high-dimensional unit hyperspheres because it reduces vector magnitude in intermediate blends. SLERP interpolates along the geodesic curve preserving vector norm:
$$\Omega = \arccos\left(\frac{W_0 \cdot W_1}{\|W_0\| \|W_1\|}\right)$$
$$\text{SLERP}(W_0, W_1; \alpha) = \frac{\sin((1-\alpha)\Omega)}{\sin\Omega} W_0 + \frac{\sin(\alpha\Omega)}{\sin\Omega} W_1$$

### 3. TIES-Merging (Trimming, Electing Sign, Disjoint Merge)
Resolves destructive parameter interference across multiple fine-tuned models:
1. **Trim (Top-K Sparsification)**: Retain only the top $p\%$ largest magnitude delta parameters in each task vector; set remaining to 0.
2. **Elect Sign**: Compute majority consensus sign vector $\gamma = \text{sgn}\left(\sum_k \tau_k\right)$.
3. **Disjoint Merge**: Discard values that disagree with the consensus sign, then average the remaining aligned parameters.

### 4. DARE (Drop And REscale)
Randomly zeroes out delta parameters with probability $p$ and rescales the survivors by $\frac{1}{1-p}$ to maintain expectation:
$$\tau_{dare} = \frac{1}{1-p} \cdot (\tau \odot M), \quad M_{ij} \sim \text{Bernoulli}(1-p)$$

---

## Tool Execution

All algorithms are implemented in `merge_models.py` and accessible via the unified CLI:

```bash
# Linear Task Arithmetic
python -m weight_space_programming_skill merge linear base.pt ft.pt \
    --alpha 0.6 --output merged_linear.pt

# SLERP Interpolation (Best for combining two generalist LLMs)
python -m weight_space_programming_skill merge slerp base.safetensors ft.safetensors \
    --alpha 0.5 --output merged_slerp.safetensors

# TIES Merging (Multi-model conflict resolution)
python -m weight_space_programming_skill merge ties base.pt ft1.pt ft2.pt \
    --density 0.4 --output merged_ties.pt

# DARE Merging (Extreme sparsity with noise suppression)
python -m weight_space_programming_skill merge dare base.pt ft.pt \
    --density 0.5 --alpha 0.7 --output merged_dare.pt
```

---

## Failure Modes & Diagnostic Checks
- **Cosine Similarity Check**: If $\cos(W_0, W_1) \approx 0$ or negative, models do not share a common loss basin; merging will produce incoherent gibberish.
- **Layer-wise Alpha Scheduling**: Apply lower $\alpha$ (0.1–0.3) to early feature extraction layers, and higher $\alpha$ (0.6–0.8) to higher reasoning layers.
- **Always Validate Output Norms**: Verify that merged layer norms are within $\pm 10\%$ of base model layer norms.
