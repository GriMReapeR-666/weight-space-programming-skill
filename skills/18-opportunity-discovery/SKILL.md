---
name: 18-opportunity-discovery
description: Automated scanning of models for weight optimization opportunities: low-rank factorization, dead channel removal, precision downscaling.
---

# 18 Opportunity Discovery & Optimization Register

## Purpose
Before executing any ad-hoc modification, this skill directs the agent to scan the checkpoint for quantifiable optimization opportunities, calculate estimated parameter and compute savings, and register candidate operations in an opportunity ledger.

---

## Opportunity Categories & Discovery Rubrics

### 1. Low-Rank Matrix Factorization Opportunity
- **Condition**: Linear layer $W \in \mathbb{R}^{m \times n}$ where stable rank $\text{srank}(W) \ll \min(m, n) / 2$.
- **Action**: Replace $W$ with rank-$r$ factorization $B \cdot A$.
- **Savings**: Parameter reduction from $m \cdot n$ to $r \cdot (m + n)$.
- **Confidence Metric**: Reconstruction energy $E(r) \ge 0.95$.

### 2. Dead Channel & Sparsity Pruning Opportunity
- **Condition**: Layer contains $> 20\%$ dead neurons or zero-valued rows/columns.
- **Action**: Slice out unactivated channels and adjust succeeding layer input dimensions.
- **Savings**: Direct proportional FLOP and latency reduction.

### 3. Mixed-Precision Quantization Opportunity
- **Condition**: Non-critical layers with low sensitivity gradient norms and low dynamic range.
- **Action**: Downcast target layer from FP16/BF16 to INT8 or INT4, while keeping outlier-sensitive layers at full precision.

---

## Scanning Protocol & Ledger Recording

Execute an automated audit across all model layers:
```bash
python wsp_protocol/tools/tensor_summary.py models/target.safetensors --top-k 10 > summary.json
```

Filter and populate `templates/opportunity-register.csv`:
```csv
id,component,opportunity_type,current_params,potential_savings,estimated_speedup,risk_level
OPP-01,model.layers.0-5.mlp,low_rank_factorization,50331648,37748736,1.4x,LOW
OPP-02,model.layers.18-23.attn,int8_quantization,25165824,12582912,1.2x,VERY_LOW
OPP-03,lm_head,svd_truncation,32000000,16000000,1.1x,MEDIUM
```

Log opportunity scan completion:
```bash
python wsp_protocol/tools/experiment_ledger.py log \
    --event opportunity_scan \
    --model models/target.safetensors \
    --notes "Discovered 3 high-impact opportunities with estimated 35% memory reduction"
```
