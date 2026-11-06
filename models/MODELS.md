# models/

> **This directory is gitignored — drop local model files here.**

Model weights are never committed. This folder is the conventional local
drop-zone for checkpoints you want to run the WSP tools against.

## Quick-start

```bash
# Download a model from Hugging Face
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('jaswanthsanjay88/rev-decision-model', 'model.safetensors', local_dir='models/rev-decision-model')
"

# Inspect it
python wsp_protocol/tools/tensor_summary.py models/rev-decision-model/model.safetensors

# Auto-generate an adapter skeleton
python wsp_protocol/tools/scaffold_adapter.py models/rev-decision-model/model.safetensors --json

# LoRA injection plan
python wsp_protocol/tools/lora_inject.py inspect models/rev-decision-model/model.safetensors
```

## Tested models

| Model | HF repo | Format | Size |
|-------|---------|--------|------|
| rev-decision-model | `jaswanthsanjay88/rev-decision-model` | safetensors | ~421M params |

Add your own entries here (not committed — edit locally).
