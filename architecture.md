# Architecture Overview

The **Weight Space Programming Skill** bundle provides an agentic runtime and programmatic interface for inspecting, patching, merging, and adapting neural network weights directly without retraining.

---

## Directory Layout

```
weight-space-programming-skill/
├── .github/
│   └── workflows/ci.yml               # Automated test & verification pipeline
├── adapters/                          # 40+ architecture-specific weight space playbooks
├── models/                            # Gitignored local drop-zone for model checkpoints
├── references/                        # SDK contracts, schema specs, and architecture references
├── schemas/                           # JSON schemas for manifests, configurations, and plans
├── scripts/
│   └── download_model.py              # Lightweight CLI helper to pull checkpoints into models/
├── skills/                            # 20 operational skill playbooks (LoRA, merge, quant, etc.)
├── src/
│   └── weight_space_programming_skill/ # Core Python package
│       ├── __init__.py                # Package exports (WeightSpaceProgram)
│       ├── __main__.py                # CLI runner (python -m weight_space_programming_skill)
│       ├── cli.py                     # Unified CLI routing for all WSP tools
│       ├── adapters/                  # Pluggable runtime adapters
│       │   ├── __init__.py            # Adapter registry (BaseAdapter, GenericAdapter)
│       │   ├── base.py                # Abstract BaseAdapter contract
│       │   └── generic_adapter.py     # Default universal adapter
│       ├── src/                       # Orchestration engine
│       │   ├── __init__.py            # Re-exports WeightSpaceProgram
│       │   ├── core.py                # Core orchestrator and dynamic adapter loader
│       │   └── importlib.py           # Safe importlib abstraction
│       └── utils/                     # Supporting utilities
│           ├── __init__.py
│           ├── config.py              # Configuration manager
│           ├── env_inspector.py       # Host environment inspection
│           ├── logger.py              # Structured logging
│           ├── schema_validator.py    # Manifest JSON schema validator
│           └── token_metrics.py       # Execution token tracking
├── templates/                         # Templates for experiments, reports, and manifests
├── tests/                             # Pytest test suite
│   ├── test_core.py                   # Core orchestrator & adapter loading tests
│   └── test_integration.py            # End-to-end multi-tool integration pipeline
├── wsp_protocol/                      # Agentic Weight Space Programming protocol
│   ├── playbooks/                     # 7 formal stage-by-stage engineering playbooks
│   ├── references/                    # Tool API contracts & MCP server architecture
│   ├── templates/                     # Hypotheses, system prompts, patch manifests
│   ├── tests/                         # Acceptance gates & verification rubrics
│   └── tools/                         # 10 operational tools for weight engineering
│       ├── ablation_runner.py         # Causal ablation probe
│       ├── activation_capture.py      # PyTorch forward hook statistics
│       ├── experiment_ledger.py       # Append-only JSONL experiment tracker
│       ├── gguf_reader.py             # GGUF inspection & dequantization stats
│       ├── lora_inject.py             # Low-rank adapter injection & merging
│       ├── merge_models.py            # Linear, SLERP, TIES, and DARE merge algorithms
│       ├── patch_delta.py             # Atomic non-destructive weight patcher
│       ├── safetensors_adapter.py     # Safetensors inspection, patching, and merging
│       ├── scaffold_adapter.py        # Automated adapter skeleton generator
│       └── tensor_summary.py          # Detailed tensor parameter statistics
├── .gitignore                         # Strict rules ignoring weights, bins, and logs
├── LICENSE                            # MIT License
├── pyproject.toml                     # Package build specification (PEP 518/621)
├── README.md                          # Project documentation and quick start
├── requirements.txt                   # Dependency definitions
└── SKILL.md                           # Antigravity skill specification entrypoint
```

---

## Architectural Principles

1. **Non-Destructive Operations**: Tools never mutate original weights in-place. All operations work against intermediate buffers and write to target paths with verify-on-reload guarantees.
2. **Standard-Library First**: Minimal external dependencies; uses native Python constructs where possible.
3. **Pluggable Adapter Model**: Dynamically discovers and invokes architecture-specific adapters through `WeightSpaceProgram`.
4. **Lightweight Bundle Footprint**: Model weights, binary checkpoints, and temporary caches are strictly excluded via `.gitignore`, preserving a clean skill-bundle footprint.
