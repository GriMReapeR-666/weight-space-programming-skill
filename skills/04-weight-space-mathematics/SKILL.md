---
name: 04-weight-space-mathematics
description: Mathematical operations directly in weight space: SVD low-rank decomposition, singular spectrum analysis, orthogonality, and null-space projection.
---

# 04 Weight-Space Mathematics

## Purpose
Weight-space engineering treats parameter tensors as linear maps between vector spaces. This skill provides mathematical protocols to perform direct algebraic transformations on weight matrices without lossy full-parameter gradient retraining: Singular Value Decomposition (SVD), low-rank matrix approximation, subspace alignment, null-space projection, and spectral truncation.

---

## Core Mathematical Operations

### 1. Truncated SVD Low-Rank Approximation
Any weight matrix $W \in \mathbb{R}^{m \times n}$ factorizes into:
$$W = U \Sigma V^T = \sum_{i=1}^{\min(m,n)} \sigma_i u_i v_i^T$$
A rank-$r$ approximation minimizes Frobenius reconstruction error (Eckart–Young–Mirsky theorem):
$$W_r = U_{:, :r} \Sigma_{:r, :r} V_{:, :r}^T$$
Decomposed into LoRA-style matrices $A, B$ where $W_r \approx B \cdot A$:
$$B = U_{:, :r} \sqrt{\Sigma_{:r, :r}}, \quad A = \sqrt{\Sigma_{:r, :r}} V_{:, :r}^T$$

```python
import torch

def decompose_svd(W: torch.Tensor, rank: int):
    # W shape: (out_features, in_features)
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]
    
    # LoRA factor format: B (out x r), A (r x in)
    sqrt_S = torch.diag(torch.sqrt(S_r))
    B = U_r @ sqrt_S
    A = sqrt_S @ Vh_r
    return B, A
```

### 2. Spectral Energy & Stable Rank
Evaluate the effective dimensionality of a weight matrix:
- **Energy Preservation**:
  $$\text{Energy}(r) = \frac{\sum_{i=1}^r \sigma_i^2}{\sum_{i=1}^{\min(m,n)} \sigma_i^2}$$
- **Stable Rank**:
  $$\text{srank}(W) = \frac{\|W\|_F^2}{\|W\|_2^2} = \frac{\sum_i \sigma_i^2}{\sigma_1^2}$$
  A low stable rank indicates significant compression or low-rank edit opportunities.

### 3. Null-Space Projection (Safe Edits)
To modify $W$ to alter behavior on a subspace $X_{edit}$ without interfering with protected knowledge $X_{keep}$:
1. Compute the column space covariance $C = X_{keep}^T X_{keep}$.
2. Compute the null-space projector $P_{\perp} = I - C (C^T C)^{-1} C^T$.
3. Any perturbation $\Delta W = \Delta \tilde{W} P_{\perp}$ satisfies $\Delta W X_{keep} = 0$, guaranteeing zero collateral degradation.

---

## Tool Integration & CLI

Use `lora_inject.py` to create and merge low-rank updates directly derived from matrix math:
```bash
# Decompose and create LoRA scaffold
python wsp_protocol/tools/lora_inject.py scaffold models/target.pt \
    --layer model.layers.0.self_attn.q_proj \
    --rank 16 \
    --output lora_init.pt
```

---

## Diagnostic Checklist
- [ ] Check condition number $\kappa(W) = \frac{\sigma_1}{\sigma_k}$. If $\kappa > 10^5$, apply regularization before inverting.
- [ ] Verify reconstruction error: $\|W - BA\|_F / \|W\|_F < \epsilon$.
- [ ] Check for singular value collapse (vanishing values beyond rank $k$).
