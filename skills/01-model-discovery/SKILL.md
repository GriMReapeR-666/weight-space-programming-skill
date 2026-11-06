---
name: 01-model-discovery
description: Identify checkpoint formats, header metadata, execution framework, parameter counts, and model purpose.
---

# 01 Model Discovery & Checkpoint Forensics

## Purpose
When presented with an unknown or unlabelled model checkpoint (`.safetensors`, `.gguf`, `.pt`, `.onnx`, `.bin`), this skill directs the agent through an automated, safe discovery sequence to uncover the model's architecture family, parameter footprint, quantization state, tensor layout, and runtime requirements without risking memory explosion or arbitrary code execution.

---

## Pre-Conditions & Safety Invariants
1. **Never use `torch.load(weights_only=False)`** on unverified checkpoints; always pass `weights_only=True` to prevent arbitrary code execution from malicious pickles.
2. **Never load full model weights into RAM** for inspection when metadata-only reading is possible. Use `safetensors.safe_open` or `gguf` header parsers.
3. **Never overwrite original files.** All inspections are strictly read-only.

---

## Step-by-Step Execution Protocol

### Step 1: File Header & Format Identification
Inspect magic bytes, file extensions, and container headers:
- `safetensors`: First 8 bytes define a 64-bit unsigned integer stating JSON header size in bytes.
- `gguf`: Starts with ASCII `GGUF` (bytes `0x47 0x47 0x55 0x46`), followed by version and tensor counts.
- `PyTorch (.pt / .pth)`: Zip container or legacy pickle stream.
- `ONNX`: Protobuf binary format starting with protobuf wire tags.

```bash
# Unified WSP inspection command
python -m weight_space_programming_skill inspect path/to/model.safetensors
python -m weight_space_programming_skill inspect path/to/model.gguf
python -m weight_space_programming_skill inspect path/to/model.pt
```

### Step 2: Automated Architecture & Tensor Inventory
Extract tensor names, shapes, and datatypes, then categorize parameter roles:
```bash
python wsp_protocol/tools/scaffold_adapter.py path/to/model.safetensors --json > arch_inventory.json
```

Use `scaffold_adapter.py` to classify tensor names into component families:
- **Embeddings**: `token_emb`, `embed_tokens`, `wte`
- **Attention Projections**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_weight`
- **Feed-Forward Blocks**: `gate_proj`, `up_proj`, `down_proj`, `fc1`, `fc2`
- **Layer Normalization**: `input_layernorm`, `post_attention_layernorm`, `rms_norm`
- **Routing / MoE**: `gate_weight`, `expert_router`
- **Heads**: `lm_head`, `classification_head`

### Step 3: Parameter & Memory Sizing Analysis
Calculate raw memory footprint and parameter distribution:

```python
import sys
from pathlib import Path
from wsp_protocol.tools.safetensors_adapter import inspect_safetensors

meta = inspect_safetensors("models/target.safetensors")
total_params = sum(t["n_elements"] for t in meta["tensors"])
total_bytes = meta["total_bytes"]

print(f"Total Parameters : {total_params:,} ({total_params/1e6:.2f}M / {total_params/1e9:.2f}B)")
print(f"Memory Footprint : {total_bytes / (1024**3):.2f} GB")
print(f"Tensor Count     : {meta['n_tensors']}")
```

### Step 4: Experiment Logging
Always record the discovered model identity into the append-only ledger:
```bash
python wsp_protocol/tools/experiment_ledger.py log \
    --event discovery \
    --model path/to/model.safetensors \
    --notes "Discovered dense transformer, 421M params, bfloat16, 24 layers"
```

---

## Deliverables & Output Schema
A valid discovery output must produce a JSON report containing:
```json
{
  "checkpoint_path": "models/model.safetensors",
  "format": "safetensors",
  "total_parameters": 421458944,
  "dtype_distribution": {"BF16": 288, "FP32": 4},
  "architecture_guess": "dense_transformer",
  "detected_layers": 24,
  "has_tied_embeddings": true,
  "context_length_hint": 8192
}
```

---

## Diagnostic Checklist & Troubleshooting
- [ ] **Unexpected pickle error on .pt?** Re-run with `torch.load(..., weights_only=True)`. If custom classes are present, isolate inside a sandbox.
- [ ] **Sharded checkpoints?** Check if a `.safetensors.index.json` or `pytorch_model.bin.index.json` exists in the parent folder mapping tensors to shard files.
- [ ] **Non-standard naming?** Consult `references/unknown-model-discovery-protocol.md` for heuristic clustering based on tensor shape symmetries.
