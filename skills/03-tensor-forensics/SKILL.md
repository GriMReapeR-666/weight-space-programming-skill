---
name: 03-tensor-forensics
description: Deep statistical auditing of weight tensors for NaNs, Infs, extreme outliers, zero-sparsity, dead channels, and dynamic ranges.
---

# 03 Tensor Forensics & Anomaly Detection

## Purpose
Before modifying, pruning, merging, or fine-tuning weights, the agent must perform a forensic audit to establish an objective baseline of weight health. This identifies pre-existing corrupted layers, catastrophic outlier channels, dead/zeroed neurons, or numerical instabilities ($NaN$, $\pm\infty$).

---

## Forensic Metrics & Mathematical Criteria

For each tensor parameter matrix $W \in \mathbb{R}^{m \times n}$:

1. **Finite Ratio**:
   $$\text{finite\_ratio} = \frac{1}{mn}\sum_{i,j} \mathbb{I}(\text{isfinite}(W_{ij}))$$
   *Requirement*: Must strictly equal $1.0000$. Any value $< 1.0$ indicates numerical breakdown.

2. **Frobenius / L2 Norm**:
   $$\|W\|_F = \sqrt{\sum_{i,j} W_{ij}^2}$$

3. **Kurtosis & Outlier Detection**:
   Locate individual weights $|W_{ij}| > 5 \cdot \sigma_W + \mu_W$. Heavy-tailed outlier channels (frequent in LLMs like LLM.int8()) must be catalogued before quantization or merging.

4. **Zero-Sparsity**:
   $$\text{sparsity} = \frac{1}{mn}\sum_{i,j} \mathbb{I}(W_{ij} == 0)$$

---

## Tool Invocations

### 1. Compute Full Distribution & Quantiles
Execute `tensor_summary.py` on the target checkpoint:
```bash
python wsp_protocol/tools/tensor_summary.py models/target.safetensors --top-k 10 > audit_report.json
```

### 2. Forensic Scan via Python
```python
import torch
from safetensors import safe_open

def audit_checkpoint(path):
    anomalies = []
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            t = f.get_tensor(key).float()
            
            # 1. Finite check
            if not torch.all(torch.isfinite(t)):
                anomalies.append({"tensor": key, "type": "NON_FINITE_VALUES"})
                
            # 2. Dead tensor check
            if (t == 0).float().mean() > 0.90:
                anomalies.append({"tensor": key, "type": "EXCESSIVE_SPARSITY", "sparsity": float((t == 0).float().mean())})
                
            # 3. Dynamic range explosion
            std = float(t.std())
            if std > 10.0 or std < 1e-6:
                anomalies.append({"tensor": key, "type": "ABNORMAL_VARIANCE", "std": std})
                
    return anomalies
```

### 3. Log Audit in Experiment Ledger
```bash
python wsp_protocol/tools/experiment_ledger.py log \
    --event audit \
    --model models/target.safetensors \
    --notes "Forensic scan: 0 NaNs, 2 sparse biases in output heads"
```

---

## Failure Modes & Prescriptions
- **Issue: High outlier values ($|x| > 30$) in specific MLP down-projection columns.**
  *Cause*: Standard activation feature collapse in transformer scaling.
  *Remedy*: Mark these columns as sensitive; preserve at FP16/BF16 during quantization or pruning.
- **Issue: Subnormal numbers (`denormal float`).**
  *Remedy*: Flush to zero using PyTorch tensor operations `torch.where(t.abs() < 1e-30, 0.0, t)`.
