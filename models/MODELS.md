# models/

> **This directory is gitignored — drop local model files here.**

Model weights are never committed to the repository. This folder is the conventional local
drop-zone for checkpoints you want to inspect, profile, patch, or evaluate using WSP tools.

## Quick-start

```bash
# 1. Inspect any local checkpoint
python -m weight_space_programming_skill inspect models/<your-model>/model.safetensors

# 2. Profile condition numbers and singular spectrum
python -m weight_space_programming_skill profile models/<your-model>/model.safetensors --cond-threshold 100.0

# 3. Formulate Pareto-constrained surgery plan
python -m weight_space_programming_skill plan models/<your-model>/model.safetensors --objective "Stabilize MLP"
```
