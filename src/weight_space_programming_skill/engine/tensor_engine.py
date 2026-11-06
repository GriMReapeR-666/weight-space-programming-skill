"""TensorEngine: High-performance, memory-bounded tensor inspection and statistical analysis."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import torch

from .loader import ModelLoader
from .types import ModelMetadata, TensorRole, TensorSummary

# Role pattern mapping
ROLE_RULES = [
    (r"embed|wte|wpe|token_emb", TensorRole.EMBEDDING, "embedding"),
    (r"(?:self_attn|attn)\.q(?:_proj)?", TensorRole.ATTENTION_QUERY, "transformer_attention"),
    (r"(?:self_attn|attn)\.k(?:_proj)?", TensorRole.ATTENTION_KEY, "transformer_attention"),
    (r"(?:self_attn|attn)\.v(?:_proj)?", TensorRole.ATTENTION_VALUE, "transformer_attention"),
    (r"(?:self_attn|attn)\.o(?:_proj)?|out_proj", TensorRole.ATTENTION_OUTPUT, "transformer_attention"),
    (r"(?:self_attn|attn)\.(?:in_proj|qkv)", TensorRole.ATTENTION_QKV_FUSED, "transformer_attention"),
    (r"mlp\.(?:gate|fc1|w1)_proj", TensorRole.FFN_GATE, "transformer_ffn"),
    (r"mlp\.(?:up|fc3)_proj", TensorRole.FFN_UP, "transformer_ffn"),
    (r"mlp\.(?:down|fc2|w2)_proj", TensorRole.FFN_DOWN, "transformer_ffn"),
    (r"norm|ln_[12f]|rms_norm", TensorRole.NORMALIZATION, "normalization"),
    (r"lm_head|output_projection|classifier", TensorRole.OUTPUT_HEAD, "output"),
    (r"expert|router|gate(?:_weight)?", TensorRole.ROUTING_MOE, "moe"),
    (r"conv\d*\.", TensorRole.CONVOLUTION, "cnn"),
    (r"ssm|mamba|recurrent", TensorRole.RECURRENT, "ssm"),
]


def classify_role(name: str) -> tuple[TensorRole, str]:
    """Classify a tensor name into a canonical role and component family."""
    name_l = name.lower()
    for pattern, role, fam in ROLE_RULES:
        if re.search(pattern, name_l):
            return role, fam
    return TensorRole.GENERIC, "generic"


def infer_layer(name: str) -> Optional[int]:
    """Extract integer layer index from tensor name."""
    m = re.search(r"(?:layers?|blocks?|blk)\.(\d+)", name)
    if m:
        return int(m.group(1))
    return None


def inspect_model(path: str) -> ModelMetadata:
    """Inspect model checkpoint format, sizing, and layer topology."""
    return ModelLoader.load_metadata(path)


def inspect_tensor(path: str, tensor_name: str) -> TensorSummary:
    """Inspect a single tensor parameter with full distribution statistics."""
    target_tensor = None
    for name, tensor in ModelLoader.stream_tensors(path):
        if name == tensor_name:
            target_tensor = tensor
            break

    if target_tensor is None:
        raise KeyError(f"Tensor '{tensor_name}' not found in {path}")

    return compute_tensor_summary(tensor_name, target_tensor)


def summarize_weights(path: str, top_k: int = 5) -> List[TensorSummary]:
    """Stream and summarize all tensors in a checkpoint without RAM blowup."""
    summaries = []
    for name, tensor in ModelLoader.stream_tensors(path):
        summaries.append(compute_tensor_summary(name, tensor, top_k=top_k))
    return summaries


def compute_tensor_summary(
    name: str, tensor: Any, top_k: int = 5
) -> TensorSummary:
    """Compute detailed numerical statistics for a single tensor."""
    if not torch.is_tensor(tensor):
        tensor = torch.tensor(tensor)

    t = tensor.detach().float()
    shape = list(t.shape)
    numel = int(t.numel())
    finite_mask = torch.isfinite(t)
    finite_fraction = float(finite_mask.float().mean())

    role, family = classify_role(name)
    layer = infer_layer(name)

    if finite_fraction == 1.0 and numel > 0:
        mean_val = float(t.mean())
        std_val = float(t.std(unbiased=False))
        min_val = float(t.min())
        max_val = float(t.max())
        sparsity = float((t == 0).float().mean())
        l2_norm = float(torch.linalg.vector_norm(t))

        # Quantiles: 0%, 25%, 50%, 75%, 100%
        try:
            q_vals = torch.quantile(t.flatten(), torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])).tolist()
        except Exception:
            q_vals = [min_val, mean_val, max_val]
    else:
        mean_val = std_val = min_val = max_val = sparsity = l2_norm = 0.0
        q_vals = []

    return TensorSummary(
        name=name,
        shape=shape,
        dtype=str(tensor.dtype),
        numel=numel,
        role=role,
        component_family=family,
        layer_index=layer,
        mean=mean_val,
        std=std_val,
        min_val=min_val,
        max_val=max_val,
        sparsity=sparsity,
        l2_norm=l2_norm,
        finite_fraction=finite_fraction,
        quantiles=q_vals,
    )


def analyze_weight_directions(
    matrix: torch.Tensor, top_k: int = 5
) -> Dict[str, Any]:
    """Compute Singular Value Decomposition (SVD), stable rank, and spectral energy."""
    if matrix.ndim != 2:
        raise ValueError(f"SVD requires 2D matrix, got shape {matrix.shape}")

    U, S, Vh = torch.linalg.svd(matrix.float(), full_matrices=False)
    total_energy = float(torch.sum(S ** 2))
    k = min(top_k, len(S))

    top_singular_values = S[:k].tolist()
    top_energy_ratio = float(torch.sum(S[:k] ** 2) / (total_energy + 1e-12))
    stable_rank = float((torch.sum(S ** 2) / (S[0] ** 2)).item())
    condition_number = float((S[0] / (S[-1] + 1e-12)).item())

    return {
        "singular_values": top_singular_values,
        "energy_ratio_top_k": top_energy_ratio,
        "stable_rank": stable_rank,
        "condition_number": condition_number,
        "total_rank": len(S),
    }
