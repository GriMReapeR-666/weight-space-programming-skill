---
name: 14-generative-model-engineering
description: Weight-space operations for diffusion models, UNet / DiT backbones, text encoder merging, and LoRA composition.
---

# 14 Generative Model Engineering (Diffusion & DiT)

## Purpose
Generative models (Stable Diffusion, SDXL, Flux, DiT, Flow Matching) possess distinct component blocks: text encoders (CLIP, T5), cross-attention layers, and image denoisers (UNet / Diffusion Transformer). This skill provides guidelines for weight-space composition, LoRA merging across text/image backbones, and block-weighted blending.

---

## Architecture-Specific Rules

### 1. Component Isolation
Never merge parameters across different component boundaries:
- **Text Encoders**: `text_model.*`, `conditioner.embedders.*` (embed conceptual semantic space).
- **Latent Denoisers**: `model.diffusion_model.*`, `double_blocks.*`, `single_blocks.*` (control image synthesis dynamics).
- **VAE**: `first_stage_model.*` (reconstruct pixels from latents).

### 2. Block-Weighted Merging (UNet Input/Middle/Output)
In diffusion UNets, different blocks control different hierarchical features:
- **Down-Blocks (Input)**: High-level composition, geometry, perspective.
- **Mid-Block**: Core thematic semantics, balance.
- **Up-Blocks (Output)**: Fine texture, lighting, micro-details.

When blending two generative models $A$ and $B$, apply block-specific weight ratios:
$$\alpha_{down} = 0.3, \quad \alpha_{mid} = 0.5, \quad \alpha_{up} = 0.8$$

### 3. Merging via Safetensors Tool
```bash
python wsp_protocol/tools/safetensors_adapter.py inspect models/sd_model.safetensors
```

Inspect and blend generative checkpoints with non-destructive guarantees:
```bash
python -m weight_space_programming_skill merge slerp \
    models/style_a.safetensors models/style_b.safetensors \
    --alpha 0.5 \
    --output models/blended_style.safetensors
```
