---
name: 16-gnn-scientific-engineering
description: Graph Neural Networks, message passing layers, equivariant molecular networks, and physics-informed neural operator weights.
---

# 16 GNN & Scientific Neural Network Engineering

## Purpose
Scientific models (molecular property predictors, AlphaFold, Equivariant GNNs, Fourier Neural Operators) enforce strict physical symmetries: SE(3) rotational equivariance, permutation invariance, or energy conservation. Modifying weights without honoring these physical constraints destroys conservation laws. This skill defines protocols for weight engineering on scientific networks.

---

## Physical Constraints & Principles

### 1. SE(3) Equivariant Networks (EGNN, SchNet, DimeNet)
- Node coordinates $x_i$ and edge distances $d_{ij} = \|x_i - x_j\|$ are transformed by radial neural networks.
- Spherical harmonic filter weights must not be perturbed arbitrarily; any modification must preserve irrep tensor product decompositions.
- Verify energy conservation: $\sum_i F_i = -\nabla_{x_i} E = 0$ (net force vanishes in isolated systems).

### 2. Message Passing Neural Networks (MPNN)
- Aggregation functions (`sum`, `mean`, `max`) are parameter-free.
- Message functions $\phi_e(h_i, h_j, e_{ij})$ and update functions $\phi_v(h_i, m_i)$ are standard MLPs.
- Parameter operations (merging, LoRA, pruning) should target update MLPs first, leaving edge embedding layers untouched.

### 3. Physics-Informed Neural Networks (PINN / Neural Operators)
- In FNO (Fourier Neural Operators), weights operate in the frequency domain: `spectral_weights` $\in \mathbb{C}^{d_{in} \times d_{out} \times k_{max}}$.
- High-frequency cutoff modes can be zero-pruned to act as low-pass physical filters.

---

## Tool Execution
Always run forensic verification before modifying scientific checkpoints:
```bash
python wsp_protocol/tools/tensor_summary.py models/scientific_model.pt --top-k 10
```
Record physical benchmark checks in `experiment_ledger.py`:
```bash
python wsp_protocol/tools/experiment_ledger.py log \
    --event scientific_audit \
    --model models/scientific_model.pt \
    --notes "Verified SE(3) symmetry invariance after 10% weight pruning"
```
