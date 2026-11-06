# Weight Space Programming Skill

> **A lightweight Antigravity skill bundle for weight-space engineering on AI models.**  
> Inspect, patch, merge, and adapt neural network checkpoints — without touching training code.

---

## What it does

This bundle gives an AI agent (or a human) a full toolkit for operating directly
on model weights:

| Layer | What it covers |
|-------|---------------|
| **Core** | `WeightSpaceProgram` orchestrator + pluggable adapter discovery |
| **Tools** | 10 CLI + importable tools in `wsp_protocol/tools/` |
| **Adapters** | 40+ architecture adapter playbooks in `adapters/` |
| **Skills** | 20 skill modules (LoRA, merge, quantise, steer, probe…) |
| **Protocol** | Agentic WSP execution protocol (`wsp_protocol/`) |

---

## Tools (`wsp_protocol/tools/`)

| Tool | What it does |
|------|-------------|
| `tensor_summary.py` | Per-tensor stats (shape, dtype, mean/std, quantiles, top-k abs) |
| `safetensors_adapter.py` | Inspect / patch / merge `.safetensors` files |
| `scaffold_adapter.py` | Auto-generate adapter skeleton from any checkpoint |
| `lora_inject.py` | LoRA A/B scaffold → merge back into weights |
| `merge_models.py` | Linear / SLERP / TIES / DARE model merging |
| `activation_capture.py` | Forward-hook activation statistics capture |
| `ablation_runner.py` | Causal ablation probe (zero/mean/noise/freeze) |
| `gguf_reader.py` | GGUF header inspect + dequant stats |
| `patch_delta.py` | Apply a weight delta patch file |
| `experiment_ledger.py` | Append-only JSONL experiment log |

---

## Quick Start

```bash
# Install
pip install -e .

# Download a model (stays gitignored in models/)
python scripts/download_model.py jaswanthsanjay88/rev-decision-model

# Inspect weights
python wsp_protocol/tools/tensor_summary.py models/rev-decision-model/model.safetensors

# Auto-generate adapter skeleton
python wsp_protocol/tools/scaffold_adapter.py models/rev-decision-model/model.safetensors --json

# Inject LoRA (dry-run inspect)
python wsp_protocol/tools/lora_inject.py inspect models/rev-decision-model/model.safetensors

# Merge two checkpoints (SLERP)
python wsp_protocol/tools/merge_models.py slerp base.pt tuned.pt --alpha 0.5 --output merged.pt

# Log an experiment
python wsp_protocol/tools/experiment_ledger.py log --event train --model my_model --notes "run 1"

# Python API
python -m weight_space_programming_skill --help
```

---

## Python API

```python
from weight_space_programming_skill import WeightSpaceProgram

wsp = WeightSpaceProgram()
result = wsp.run("describe transformer attention weights")
print(result)
```

---

## Model files are gitignored

Weights never live in this repo. Use the helper:

```bash
python scripts/download_model.py <hf-repo-id>
# → models/<name>/model.safetensors  (gitignored)
```

---

## Tests

```bash
pytest -v        # 8 tests: core + full integration pipeline
```

---

## Structure

```
weight-space-programming-skill/
├── src/weight_space_programming_skill/   # Python package (src layout)
│   ├── adapters/                         # BaseAdapter + GenericAdapter
│   └── src/core.py                       # WeightSpaceProgram orchestrator
├── wsp_protocol/                         # Agentic protocol + tools
│   ├── tools/                            # 10 operational tools
│   └── playbooks/                        # Execution playbooks
├── adapters/                             # 40+ architecture adapter docs
├── skills/                               # 20 skill modules
├── references/                           # Architecture references & SDK
├── schemas/                              # JSON schemas
├── scripts/                              # download_model.py + helpers
├── tests/                                # pytest suite
├── models/                               # gitignored — local checkpoints here
└── SKILL.md                              # Antigravity skill entrypoint
```
