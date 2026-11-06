---
name: 15-vision-audio-multimodal
description: Multi-modal weight spaces (ViT, Whisper, Audio Spectrogram Transformers, cross-attention projection layers).
---

# 15 Vision, Audio & Multimodal Weight Engineering

## Purpose
Multimodal architectures bind heterogeneous modalities (vision, audio, text) via shared representation spaces or cross-attention projections. This skill specifies procedures for diagnosing and adapting Vision Transformers (ViT), Whisper speech encoders, multimodal projection layers (MLP / Q-Former), and audio decoders in weight space.

---

## Modality-Specific Mechanics

### 1. Vision Transformers (ViT) & Patch Embeddings
- `patch_embed.proj.weight` carries shape `[dim, in_channels, patch_size, patch_size]` (e.g. `[768, 3, 16, 16]`).
- Do not interpolate patch projection kernels directly with linear algebra unless spatial 2D interpolation is applied.
- Class token (`cls_token`) and positional embeddings (`pos_embed`) must be preserved or bicubically interpolated when scaling input resolutions.

### 2. Audio & Speech Checkpoints (Whisper, Conformer)
- Convolutional frontend: `conv1.weight`, `conv2.weight` (1D convolutions over Mel filterbanks).
- Cross-attention layers in the decoder align acoustic tokens with linguistic vocabularies.
- LoRA updates are most effective when applied exclusively to decoder attention projections (`q_proj`, `v_proj`).

### 3. Cross-Modal Projectors
Multimodal models (LLaVA, MiniCPM-V, CLIP) bridge modalities using a projection layer:
$$h_{text\_space} = W_{proj} \cdot h_{vision}$$
To adapt the model to new visual domains without risking text generation degradation, isolate weight operations exclusively to $W_{proj}$ and the vision encoder.

---

## Tool Workflow
```bash
# Inspect tensor dimensions and identify projection layers
python wsp_protocol/tools/scaffold_adapter.py models/multimodal.safetensors --json > mm_arch.json

# Check parameter stability in projection matrix
python wsp_protocol/tools/tensor_summary.py models/multimodal.safetensors --top-k 5
```
