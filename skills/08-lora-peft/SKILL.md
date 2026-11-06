---
name: 08-lora-peft
description: Parameter-Efficient Fine-Tuning, Low-Rank Adaptation (LoRA), scaffold generation, rank selection, and weight folding.
---

# 08 LoRA & Parameter-Efficient Fine-Tuning (PEFT)

## Purpose
Low-Rank Adaptation (LoRA) freezes the pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$ and injects trainable rank decomposition matrices $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$ with rank $r \ll \min(d, k)$. This skill defines the exact process for inspecting layers, scaffolding LoRA matrices, scaling updates, and folding them back into base weights without inference latency penalties.

---

## LoRA Mathematics & Scaling Rule

During training, the modified forward pass is:
$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} (B A) x$$

Where:
- $W_0$: Base weights (frozen).
- $A \sim \mathcal{N}(0, \sigma^2)$: Initialized with Gaussian random values.
- $B = 0$: Initialized to zero so that $\Delta W = 0$ at initialization ($h = W_0 x$).
- $\alpha$: LoRA scaling hyperparameter (typically $\alpha = 2 \times r$ or $\alpha = r$).
- $\frac{\alpha}{r}$: Ensures constant learning rate dynamics when rank $r$ is varied.

### Weight Folding (Zero-Latency Inference)
Once trained or synthesized, fold $\Delta W$ directly into the base weights:
$$W_{merged} = W_0 + \frac{\alpha}{r} (B A)$$

---

## Tool Protocol & CLI Commands

### 1. Inspect Injectable Linear Layers
Scan the target model to discover linear layers suitable for LoRA adaptation:
```bash
python wsp_protocol/tools/lora_inject.py inspect models/target.safetensors
# Or using unified CLI:
python -m weight_space_programming_skill lora inspect models/target.safetensors
```

### 2. Scaffold LoRA Tensors
Generate $A$ and $B$ parameter scaffolding targeting specific projections (e.g. `q_proj`, `v_proj`):
```bash
python wsp_protocol/tools/lora_inject.py scaffold models/target.pt \
    --layer model.layers.0.self_attn.q_proj \
    --rank 8 \
    --alpha 16.0 \
    --output lora_layer0_q.pt
```

### 3. Fold LoRA into Base Checkpoint
Merge the adapter weights permanently back into the base weights:
```bash
python wsp_protocol/tools/lora_inject.py merge models/target.pt \
    --lora-file lora_layer0_q.pt \
    --output models/merged_target.pt
```

---

## Diagnostic Checklist & Verification
- [ ] **Rank Selection Guidance**:
  - $r=4$ or $r=8$: Sufficient for style tuning, classification heads, or instruction adherence.
  - $r=16$ or $r=32$: Recommended for complex reasoning, code generation, or domain transfer.
- [ ] **NaN Verification**: Run `_validate_no_nans()` after folding to guarantee numerical integrity.
- [ ] **Shape Compatibility**: Ensure $B.\text{shape}[1] == A.\text{shape}[0] == r$, and $(B.\text{shape}[0], A.\text{shape}[1]) == W_0.\text{shape}$.
