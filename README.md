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

## Python API (WSP v5.0 Runtime)

```python
from weight_space_programming_skill import (
    inspect_model,
    identify_architecture,
    reconstruct_graph,
    inspect_tensor,
    summarize_weights,
    analyze_weight_directions,
    capture_activations,
    trace_generation,
    build_circuit_map,
    run_activation_ablation,
    run_parameter_ablation,
    optimize_patch,
    generate_patch,
    validate_patch,
    apply_patch,
    rollback_patch,
    compare_models,
)

# 1. Model & Architecture Discovery
meta = inspect_model("models/my_model/model.safetensors")
arch = identify_architecture("models/my_model/model.safetensors")
graph = reconstruct_graph("models/my_model/model.safetensors")

# 2. Tensor & Directions Inspection
summary = inspect_tensor("models/my_model/model.safetensors", "layers.0.self_attn.q_proj.weight")
svd_info = analyze_weight_directions("models/my_model/model.safetensors", "layers.0.self_attn.q_proj.weight", top_k=5)

# 3. Transactional Weight Patching & Reversibility
manifest = generate_patch("base.pt", deltas={"layer.0.weight": delta_tensor}, output_patch_path="patch.json")
validate_patch(manifest, "base.pt")
apply_patch(manifest, "base.pt", "patched.pt")
rollback_patch("patched.pt", manifest, "restored.pt")
```

---

## MCP Server (Agent Tool Interface)

Run the built-in Model Context Protocol (MCP) server exposing all engine tools over JSON-RPC 2.0 stdio:

```bash
python -m weight_space_programming_skill.engine.mcp_server
```

---

## Benchmark Results: Verified Model Improvement

We applied the WSP v5.0 optimization engine to [`jaswanthsanjay88/rev-decision-model`](https://huggingface.co/jaswanthsanjay88/rev-decision-model) (421M parameter ModernBERT-large RL decision agent).

Full benchmark documentation is available in [benchmarks/BENCHMARK_REPORT.md](benchmarks/BENCHMARK_REPORT.md):

| Metric | Base Model | WSP Improved Model | Verified Gain |
|---|---|---|---|
| **`scorer.1` Condition Number** | 39,896.6 | **250.0** | **160× numerical stability gain** |
| **`head.layers.0.out_proj` Condition Number** | 7,668.5 | **250.0** | **30× numerical stability gain** |
| **`head.layers.1.out_proj` Condition Number** | 7,459.4 | **250.0** | **30× numerical stability gain** |
| **Autonomous Action Confidence** | 37.5% – 43.8% | **39.3% – 44.9%** | **+1.0% to +1.8% action confidence** |
| **Spurious Deferral Rate** | 55.0% – 62.5% | **52.5% – 58.8%** | **-2.5% to -5.0% premature escalations** |
| **Effective Agent Pipeline Throughput** | ~0.60 decisions/sec | **~0.63 decisions/sec** | **Faster resolution (fewer 3-second human/LLM fallbacks)** |
| **Protected Anchor Benchmark** | 100.0% | 100.0% | **0.0% regression (PASSED)** |

---

## Tests

```bash
pytest -v        # 22 tests: core orchestrator + complete WSP v5.0 engine suite
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
