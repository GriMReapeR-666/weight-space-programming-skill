---
name: 02-architecture-reconstruction
description: Reverse engineer computational graphs, block repeating motifs, tensor symmetries, and generate adapter code.
---

# 02 Architecture Reconstruction

## Purpose
When working with custom, obfuscated, or newly released models that lack public Hugging Face model classes or official documentation, this skill guides the agent to infer the mathematical architecture from raw tensor weights and emit a working adapter skeleton.

---

## Reconstruction Methodology

### 1. Repeating Block & Layer Index Extraction
Model architectures group parameters into regular blocks. Extract repeating numerical indices from parameter keys:
- Pattern: `model.layers.(\d+)\.(.*)` or `encoder.block.(\d+)\.(.*)`
- Layer count $L = \max(\text{layer\_index}) + 1$
- Verify that every layer $l \in [0, L-1]$ contains an identical set of tensor shapes and names.

```python
import re
from collections import defaultdict

def extract_layer_topology(tensor_names):
    layer_map = defaultdict(list)
    for name in tensor_names:
        match = re.search(r"(?:layers?|blocks?|stages?|blk)\.(\d+)", name)
        if match:
            layer_map[int(match.group(1))].append(name)
    return dict(layer_map)
```

### 2. Hidden Dimension & Head Invariance Inferences
From linear weights $W \in \mathbb{R}^{d_{out} \times d_{in}}$:
- **Hidden Dimension $d_{model}$**: Symmetrical dimension shared by Q, K, V projections and residual layers.
- **Attention Heads $N_{head}$**: Derived from $d_{model}$ and inner projection dimension ($d_{head} = d_{model} / N_{head}$, typically 64, 96, or 128).
- **Intermediate / FFN Expansion**: Usually $4 \times d_{model}$ (standard MLP) or $\frac{8}{3} \times d_{model}$ (SwiGLU architectures).
- **Key-Value Heads (GQA/MQA)**: If $\dim(W_q) \neq \dim(W_k)$, compute GQA ratio:
  $$R_{GQA} = \frac{\text{shape}(W_q)[0]}{\text{shape}(W_k)[0]}$$

### 3. Automated Adapter Code Generation
Invoke `scaffold_adapter.py` to synthesize a Python adapter tailored to the inferred architecture:

```bash
# Generate Python adapter file and structural map
python wsp_protocol/tools/scaffold_adapter.py models/target.safetensors \
    --output src/weight_space_programming_skill/adapters/target_adapter.py \
    --map models/target_arch_map.json
```

---

## Step-by-Step Reconstruction Workflow

1. **Scan Tensors**: Collect all parameter names, ranks, and shapes.
2. **Cluster by Symmetry**: Group tensors by matching shapes across layer blocks.
3. **Verify Residual Pathways**: Look for normalization layer weights (`rms_norm`, `layer_norm`) that bookend attention and feed-forward blocks.
4. **Detect Tied Embeddings**: Check if `lm_head.weight` is identical (by tensor hash) or shares shape with `embed_tokens.weight`.
5. **Synthesize Execution Harness**: Test forward-hook capture or mock layer passes using the generated adapter.

---

## Deliverables
- `arch_map.json`: Complete structural map of blocks, layers, and dimensions.
- `<model_name>_adapter.py`: Concrete implementation adhering to the `BaseAdapter` interface in `src/weight_space_programming_skill/adapters/base.py`.
- Architectural classification report in `templates/unknown-architecture-report.md`.
