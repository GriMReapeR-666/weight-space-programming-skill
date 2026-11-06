# Weight Space Programming (WSP-X)

> **A Universal Autonomous Weight-Space Engineering & Model Surgery Platform.**  
> Inspect, diagnose, patch, merge, quantize, and reconstruct neural network checkpoints directly in weight space — without touching training code or relying on black-box gradient descent.

Compatible with **Claude Code**, **OpenAI Codex**, **OpenCode**, **Cursor**, **Aider**, and **local LLM agent harnesses** via standard Model Context Protocol (MCP), CLI, and Python SDK.

---

## What it does

This platform gives any AI agent or software engineer an operational toolkit to read, analyze, and program neural network weights directly:

| Layer | Subsystem | Scientific Capability |
|---|---|---|
| **Intelligence** | `SubspaceFeatureAnalyzer` (SLFA) | Full SVD spectrum, effective rank, condition numbers, and latent vs. active circuit classification |
| **Planning** | `ParetoSurgeryPlanner` (PCSP) | Solves $\max_{\theta'} Q_{\text{target}}$ s.t. $Q_{\text{preserved}} \ge Q_{\text{baseline}} - \epsilon$; generates dependency DAGs |
| **Precision** | `SelectivePrecisionController` (SPC) | Surgical block-level dequantization and requantization for GGUF `Q4_0` / `Q8_0` with distortion tracking |
| **Safety Gate** | `CausalFalsificationVerifier` (CFMCV) | 5-layer Edit Gate (scope lock, 1% fraction cap, normalized $L_2$ distance, and matched Gaussian control falsification) |
| **Transactions** | `PatchEngine` | Bitwise reversible cryptographic patch manifests with byte-level verification and atomic rollback |
| **Tools** | `wsp_protocol/tools/` | 10 CLI + importable tools for tensor statistics, LoRA injection, model merging, and activation capture |
| **Adapters** | `adapters/` | 40+ architecture playbooks (Transformers, MoE, SSM/Mamba, CNNs, Diffusion, Multimodal) |
| **Interface** | MCP Server (`JSON-RPC 2.0`) | Universal tool interface pluggable into Claude Code, Codex, OpenCode, and local runtimes |

---

## Universal Agent Harness Compatibility

WSP-X runs seamlessly across all major AI coding agents and local execution environments:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        CALLING AGENT / HARNESS                         │
│   Claude Code  │  OpenAI Codex  │  OpenCode  │  Cursor / Aider  │  Local LLM│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ stdio / JSON-RPC 2.0
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   UNIVERSAL WSP MCP SERVER / CLI LAYER                 │
│         python -m weight_space_programming_skill.engine.mcp_server     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│  SVD & SLFA   │           │  Edit Gate &  │           │ Block-Level   │
│  Diagnostics  │           │  Falsification│           │ GGUF Surgery  │
└───────────────┘           └───────────────┘           └───────────────┘
```

### Configuring in Coding Agents

#### 1. Claude Code
Add to your Claude Code project settings (`.claude.json` or project MCP config):
```json
{
  "mcpServers": {
    "weight-space-programming": {
      "command": "python",
      "args": ["-m", "weight_space_programming_skill.engine.mcp_server"]
    }
  }
}
```

#### 2. OpenAI Codex / Custom Agent Runners
Run directly as a tool server or subagent worker:
```bash
python -m weight_space_programming_skill.engine.mcp_server
```

#### 3. OpenCode / Local Harnesses
Invoke the tools via CLI or import the unified Python SDK directly in your agent loop.

---

## Tools (`wsp_protocol/tools/`)

| Tool | What it does |
|---|---|
| `tensor_summary.py` | Per-tensor statistics (shape, dtype, mean/std, quantiles, top-k abs) |
| `safetensors_adapter.py` | Inspect / patch / merge `.safetensors` files without loading into full RAM |
| `scaffold_adapter.py` | Auto-generate an adapter skeleton for unknown or custom architectures |
| `lora_inject.py` | LoRA A/B scaffold and linear merge into base weights |
| `merge_models.py` | Linear, SLERP, TIES, and DARE model merging |
| `activation_capture.py` | Forward-hook intermediate activation statistics capture |
| `ablation_runner.py` | Causal ablation probe (zero / mean / noise / freeze) |
| `gguf_reader.py` | GGUF header inspection, block-level analysis, and dequantization stats |
| `patch_delta.py` | Apply a reversible weight delta patch manifest |
| `experiment_ledger.py` | Append-only JSONL experiment record |

---

## Quick Start

```bash
# 1. Install package in development mode
pip install -e .

# 2. Download a model into models/ (strictly gitignored drop-zone)
python scripts/download_model.py jaswanthsanjay88/rev-decision-model

# 3. Profile singular subspaces and condition numbers across all layers
python -c "
from weight_space_programming_skill.engine import profile_model_subspaces
subspace_map = profile_model_subspaces('models/rev-decision-model/model.safetensors', cond_threshold=250.0)
print('Degenerate layers:', subspace_map.degenerate_layers)
"

# 4. Formulate an evidence-backed surgery plan
python -c "
from weight_space_programming_skill.engine import ParetoSurgeryPlanner
plan = ParetoSurgeryPlanner.formulate_plan('models/rev-decision-model/model.safetensors', objective='Stabilize spectrum')
print(ParetoSurgeryPlanner.export_blueprint(plan))
"

# 5. Launch the MCP server for agent pairs
python -m weight_space_programming_skill.engine.mcp_server
```

---

## Python API (WSP-X Engine)

```python
import torch
from weight_space_programming_skill.engine import (
    inspect_model,
    profile_matrix_subspace,
    compute_spectral_regularization_delta,
    project_delta_onto_null_space,
    CausalFalsificationVerifier,
    SelectivePrecisionController,
    generate_patch,
    apply_patch,
    rollback_patch,
)

# 1. Singular Subspace Profiling
profile = profile_matrix_subspace(weight_tensor, tensor_name="mlp.down_proj.weight")
print(f"Condition Number: {profile.condition_number:.1f}, Rank: {profile.effective_rank}")

# 2. Spectral Conditioning Delta
delta_w, orig_cond, new_cond = compute_spectral_regularization_delta(weight_tensor, max_condition_number=250.0)

# 3. Orthogonal Null-Space Projection (Zero Anchor Regression)
safe_delta = project_delta_onto_null_space(delta_w, anchor_task_directions)

# 4. Multi-Layer Edit Gate Verification
CausalFalsificationVerifier.verify_target_allowed("layers.15.mlp.down_proj.weight")
CausalFalsificationVerifier.verify_no_uniform_scaling(weight_tensor, weight_tensor + safe_delta)
stats = CausalFalsificationVerifier.verify_edit_distance_and_fraction(
    weight_tensor, weight_tensor + safe_delta, total_model_parameters=421_000_000
)

# 5. Reversible Transaction & Rollback
manifest = generate_patch("base.safetensors", {"layers.15.mlp.down_proj.weight": safe_delta}, "patch.json")
apply_patch(manifest, "base.safetensors", "patched.safetensors")
rollback_patch("patched.safetensors", manifest, "restored.safetensors")
```

---

## Model Privacy & Storage Invariants

> **Model checkpoints NEVER live in git tracking.**

The repository root and `models/` directory are strictly gitignored (`*.safetensors`, `*.gguf`, `*.bin`, `*.pt`, `models/*`).  
Local checkpoints remain purely on disk as private execution targets. All modifications are captured as compact, lightweight JSON patch manifests (`*.patch.json`), maintaining a clean, portable repository.

---

## Verified Benchmark Results

We applied WSP-X optimization to [`jaswanthsanjay88/rev-decision-model`](https://huggingface.co/jaswanthsanjay88/rev-decision-model) (421M parameter ModernBERT-large RL decision agent). Full benchmark documentation is in [benchmarks/BENCHMARK_REPORT.md](benchmarks/BENCHMARK_REPORT.md):

| Metric | Base Model | WSP Improved Model | Verified Gain |
|---|---|---|---|
| **`scorer.1` Condition Number** | 39,896.6 | **250.0** | **160× numerical stability gain** |
| **`head.layers.0.out_proj` Condition Number** | 7,668.5 | **250.0** | **30× numerical stability gain** |
| **`head.layers.1.out_proj` Condition Number** | 7,459.4 | **250.0** | **30× numerical stability gain** |
| **Autonomous Action Confidence** | 37.5% – 43.8% | **39.3% – 44.9%** | **+1.0% to +1.8% action confidence** |
| **Spurious Deferral Rate** | 55.0% – 62.5% | **52.5% – 58.8%** | **-2.5% to -5.0% premature escalations** |
| **Effective Agent Pipeline Throughput** | ~0.60 decisions/sec | **~0.63 decisions/sec** | **Faster resolution (fewer 3-second fallbacks)** |
| **Protected Anchor Benchmark** | 100.0% | 100.0% | **0.0% regression (PASSED)** |

---

## Tests & CI

All unit and integration tests are verified across Python 3.10, 3.11, and 3.12:

```bash
pytest -v        # 33 passed: SVD profiling, null space, edit gate, block Q4/Q8 surgery, core WSP engine
flake8 src tests scripts  # 0 lint errors (strict PEP 8)
mypy src tests           # 0 type errors (strict type checking)
```

---

## Project Layout

```
weight-space-programming-skill/
├── src/weight_space_programming_skill/   # Core Python Package
│   ├── engine/                           # Universal WSP-X Runtime
│   │   ├── subspace_feature_analyzer.py  # SVD, effective rank, null-space projection
│   │   ├── pareto_surgery_planner.py     # Constrained surgery planning & DAGs
│   │   ├── selective_precision_controller.py # Surgical GGUF Q4/Q8 block surgery
│   │   ├── causal_falsification_verifier.py # Multi-layer Edit Gate policy & control test
│   │   ├── patch_engine.py               # Reversible patch transactions & hashing
│   │   ├── loader.py                     # Checkpoint loaders (.safetensors, .pt, .gguf)
│   │   ├── graph_engine.py               # Architecture inference & DAG reconstruction
│   │   └── mcp_server.py                 # Universal JSON-RPC 2.0 MCP server
│   └── adapters/                         # Universal adapter interface
├── wsp_protocol/                         # Operational CLI tools & playbooks
│   ├── tools/                            # 10 universal inspection & manipulation tools
│   └── playbooks/                        # Step-by-step diagnostic playbooks
├── adapters/                             # 40+ architecture-specific references
├── references/                           # Domain compendiums, ROME/WSP research, SDK docs
├── schemas/                              # JSON schemas for patches, experiments, manifests
├── scripts/                              # Download and environment utilities
├── tests/                                # 33 unit and integration tests
├── models/                               # Gitignored local drop-zone for private checkpoints
└── SKILL.md                              # Universal agent skill specification
```
