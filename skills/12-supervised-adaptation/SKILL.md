---
name: 12-supervised-adaptation
description: Supervised micro-fine-tuning, task vector generation, cross-entropy objective isolation, and atomic checkpoint saving.
---

# 12 Supervised Adaptation & Task Vector Synthesis

## Purpose
Supervised adaptation within weight-space engineering focuses on creating clean task vectors $\tau = \theta_{adapted} - \theta_{base}$. Rather than heavy end-to-end retraining, this skill prescribes rapid micro-fine-tuning loops, early stopping based on validation curvature, and isolating parameter deltas into exportable patches.

---

## Technical Protocol

### 1. Minimal Step / Low-Learning-Rate Fine-Tuning
Over-training destroys the linear arithmetic properties of weight space. Keep adaptations within the linear basin of attraction:
- **Optimizer**: AdamW with high weight decay ($\lambda = 0.05$ to $0.10$).
- **Learning Rate**: $1\times 10^{-5}$ to $5\times 10^{-5}$ for LoRA adapters; $\le 5\times 10^{-6}$ for full-rank adjustments.
- **Steps**: 50–200 steps with warm-up.

### 2. Task Vector Extraction
Once the adaptation is complete, compute and isolate the delta tensor map:
$$\Delta W_i = W_i^{(adapted)} - W_i^{(base)}$$

```python
import torch

def compute_task_vector(base_sd, adapted_sd, threshold=1e-5):
    task_vector = {}
    for key in base_sd.keys():
        if key in adapted_sd:
            delta = adapted_sd[key] - base_sd[key]
            # Zero out negligible floating point noise
            delta = torch.where(delta.abs() < threshold, torch.zeros_like(delta), delta)
            if torch.any(delta != 0):
                task_vector[key] = delta
    return task_vector
```

### 3. Packaging into Patch Files
Save the isolated task vector for deployment via `patch_delta.py`:
```bash
python wsp_protocol/tools/patch_delta.py \
    --checkpoint models/base.pt \
    --patch patch_spec.json \
    --output models/patched_adapted.pt
```

---

## Acceptance Gate
- Base capability check: Zero regression on the protected anchor dataset ($< 1\%$ change in cross-entropy).
- Clean logging in `experiment_ledger.py`.
