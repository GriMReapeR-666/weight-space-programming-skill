---
name: 05-activation-analysis
description: Forward-hook activation tracing, hidden state monitoring, dead neuron scanning, and internal representation steering.
---

# 05 Activation Analysis & Internal Representation Monitoring

## Purpose
Weight values dictate computational behavior, but observing activations during forward passes reveals how information flows through layers. This skill guides the agent to hook into internal PyTorch modules, track activation magnitudes and kurtosis, detect dead neurons, and identify steering directions.

---

## Technical Protocol

### 1. PyTorch Forward-Hook Mechanism
Non-intrusive runtime instrumentation hooks capture intermediate tensors without modifying module code:

```python
import torch

class ActivationMonitor:
    def __init__(self):
        self.stats = {}
        self.handles = []

    def register(self, model, target_module_names):
        for name, module in model.named_modules():
            if name in target_module_names:
                handle = module.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)

    def _make_hook(self, name):
        def hook(mod, inputs, output):
            # If output is a tuple (e.g. attention outputs), extract first tensor
            out = output[0] if isinstance(output, tuple) else output
            with torch.no_grad():
                detached = out.detach().float()
                self.stats[name] = {
                    "mean": float(detached.mean()),
                    "std": float(detached.std()),
                    "max": float(detached.abs().max()),
                    "sparsity": float((detached == 0).float().mean()),
                    "norm": float(torch.linalg.vector_norm(detached))
                }
        return hook

    def cleanup(self):
        for h in self.handles:
            h.remove()
```

### 2. Dead Neuron Detection
In GeLU/ReLU/SwiGLU feed-forward networks:
- A neuron $i$ is **dead** across dataset $\mathcal{D}$ if activation $a_i(x) \le 0$ (or $< \epsilon$) for all samples $x \in \mathcal{D}$.
- Dead neurons indicate redundant capacity that can be zero-pruned with 0% degradation.

### 3. Steering Vectors & Representation Addition
Compute the mean activation difference between contrasting prompt classes $\mathcal{P}_{pos}$ and $\mathcal{P}_{neg}$:
$$\vec{v}_{steer} = \mathbb{E}_{x \in \mathcal{P}_{pos}}[h_l(x)] - \mathbb{E}_{x' \in \mathcal{P}_{neg}}[h_l(x')]$$
Inject $\vec{v}_{steer}$ at inference time to steer model persona or refusal behavior:
$$h'_l = h_l + \alpha \cdot \vec{v}_{steer}$$

---

## Tool Execution

Use `activation_capture.py` to record layer statistics over reference tokens:
```bash
python wsp_protocol/tools/activation_capture.py \
    --model models/target.pt \
    --layers model.layers.0,model.layers.12,model.layers.23 \
    --output activation_profile.json
```

Log findings to the experiment ledger:
```bash
python wsp_protocol/tools/experiment_ledger.py log \
    --event activation_profile \
    --model models/target.pt \
    --notes "Recorded activation norm across 24 layers; peak variance at layer 16"
```

---

## Acceptance Rubrics
- Captured profiles must contain mean, std, L2 norm, and sparsity for each hooked layer.
- Never retain activations in GPU memory after summary statistics are computed; always detach and move to CPU immediately.
