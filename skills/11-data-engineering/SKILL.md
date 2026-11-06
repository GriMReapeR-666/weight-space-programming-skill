---
name: 11-data-engineering
description: Dataset curation, synthetic data generation, tokenization calibration, and format packing for weight adaptations.
---

# 11 Data Engineering & Token Calibration for Weight Edits

## Purpose
Weight-space programming operations (such as null-space projection, activation steering, and micro-fine-tuning) require compact, clean, highly targeted datasets rather than multi-gigabyte training corpuses. This skill specifies calibration dataset formatting, token length alignment, and deduplication.

---

## Protocols

### 1. Contrastive Calibration Pair Assembly
For activation steering vectors ($\vec{v}_{steer}$) or causal ablation probes, assemble paired prompts differing *only* in the target feature:
- Positive: `"Write a helpful response demonstrating clear, concise Python code."`
- Negative: `"Write a confusing, bloated response with unnecessary boilerplate."`
- Maintain token length symmetry to prevent positional encoding bias from polluting the hidden state difference vector.

### 2. Tokenizer Alignment Check
Before feeding calibration datasets to state-dict tools:
```python
def verify_token_alignment(tokenizer, sample_texts, max_seq_len=2048):
    encodings = [tokenizer.encode(t) for t in sample_texts]
    lengths = [len(e) for e in encodings]
    assert max(lengths) <= max_seq_len, f"Sequence exceeds maximum length: {max(lengths)}"
    # Check for unexpected UNK token pollution
    unk_counts = [e.count(tokenizer.unk_token_id) for e in encodings]
    assert sum(unk_counts) == 0, f"Encountered {sum(unk_counts)} UNK tokens in calibration set"
```

### 3. Null-Space Holdout Split (Protected Knowledge)
When applying targeted edits $\Delta W$, maintain a 50-example anchor evaluation set representing invariant core capabilities (e.g. arithmetic, basic grammar, harmless safety refusal). Measure performance before and after every weight edit.
