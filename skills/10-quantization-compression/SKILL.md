---
name: 10-quantization-compression
description: Post-Training Quantization (PTQ), GGUF k-quants, weight pruning, scale/zero-point computation, and quantization error profiling.
---

# 10 Quantization & Weight-Space Compression

## Purpose
Deploying large checkpoints on resource-constrained hardware requires compressing floating-point weights (FP32/FP16/BF16) into low-bit representations (INT8, INT4, GGUF k-quants, ternary 1.58-bit). This skill provides mathematical formulas, calibration methods, and tools for error-bounded quantization and structured pruning.

---

## Quantization Mechanics & Formulas

### 1. Uniform Affine (Asymmetric) Quantization
Maps continuous values $[x_{min}, x_{max}]$ into $b$-bit integer integers $[0, 2^b - 1]$:
$$S = \frac{x_{max} - x_{min}}{2^b - 1}, \quad Z = \text{round}\left(-\frac{x_{min}}{S}\right)$$
$$q = \text{clamp}\left(\text{round}\left(\frac{x}{S}\right) + Z, 0, 2^b - 1\right)$$
Dequantization:
$$\hat{x} = S \cdot (q - Z)$$

### 2. Symmetric Quantization
Zero-point $Z = 0$, mapping $[-x_{max}, x_{max}]$ to $[-2^{b-1}, 2^{b-1} - 1]$:
$$S = \frac{\max(|x|)}{2^{b-1} - 1}, \quad q = \text{clamp}\left(\text{round}\left(\frac{x}{S}\right), -2^{b-1}, 2^{b-1} - 1\right)$$

### 3. GGUF Quantization Schemes
The llama.cpp / GGUF ecosystem uses block-based quantization (`Q4_0`, `Q4_K_M`, `Q8_0`, `IQ4_NL`):
- `Q4_0`: 32 weights per block sharing one FP16 scale factor (4.5 bits/weight).
- `Q4_K_M`: Mixed block precision where attention weights use 4-bit and MLP matrices use variable scale/min factors.
- `Q8_0`: 8-bit precision, serving as the industry standard reference baseline.

---

## Tool Protocol & CLI Commands

### 1. Inspect GGUF Quantization Layout
Analyze metadata, quantization type per tensor, and block structures:
```bash
python wsp_protocol/tools/gguf_reader.py inspect models/model.gguf
# Or using unified CLI:
python -m weight_space_programming_skill inspect models/model.gguf
```

### 2. Compute Quantization Error Profile
Evaluate reconstruction degradation $\|W - \hat{W}\|_F / \|W\|_F$ across layers:
```python
import torch

def quantization_error(W: torch.Tensor, bits=8):
    scale = (W.abs().max() / (2**(bits - 1) - 1)).clamp(min=1e-8)
    q = torch.clamp(torch.round(W / scale), -(2**(bits - 1)), 2**(bits - 1) - 1)
    dequant = q * scale
    rel_error = torch.linalg.norm(W - dequant) / (torch.linalg.norm(W) + 1e-8)
    return float(rel_error)
```

---

## Pruning Guidelines
- **Magnitude Pruning**: Zero the lowest $p\%$ weights by absolute value $|w_{ij}|$. Safe up to $15–20\%$ sparsity without calibration.
- **Semi-Structured (2:4 Sparsity)**: For every 4 contiguous values along rows, exactly 2 must be zero. Supported by NVIDIA Ampere/Hopper Tensor Cores for $2\times$ throughput speedup.
