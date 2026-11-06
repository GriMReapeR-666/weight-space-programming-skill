"""GraphEngine: Computational graph reconstruction, data-flow topology, and module connectivity."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from .loader import ModelLoader
from .tensor_engine import infer_layer
from .types import ComputationGraph, GraphNode


def identify_architecture(path: str) -> Dict[str, Any]:
    """Identify model architecture family, layer counts, dimensions, and GQA head ratios."""
    meta = ModelLoader.load_metadata(path)
    tensor_names = []
    shapes = {}

    for name, t in ModelLoader.stream_tensors(path):
        tensor_names.append(name)
        shapes[name] = list(t.shape)

    layer_indices = sorted([int(idx) for idx in {infer_layer(n) for n in tensor_names} if idx is not None])
    hidden_dim = None
    q_heads = None
    kv_heads = None

    # Detect hidden dimension and attention heads
    for name, s in shapes.items():
        if re.search(r"self_attn\.q_proj\.weight|attn\.q\.|attn\.wqkv\.weight", name, re.IGNORECASE):
            hidden_dim = s[1] if len(s) == 2 else None
            out_dim = s[0] // 3 if "wqkv" in name.lower() else s[0]
            # Standard head dim is typically 64 or 128
            for candidate_head_dim in (64, 128, 96):
                if out_dim % candidate_head_dim == 0:
                    q_heads = out_dim // candidate_head_dim
                    break
        elif re.search(r"self_attn\.k_proj\.weight|attn\.k\.", name):
            out_dim = s[0]
            for candidate_head_dim in (64, 128, 96):
                if out_dim % candidate_head_dim == 0:
                    kv_heads = out_dim // candidate_head_dim
                    break

    gqa_ratio = 1.0
    if q_heads and kv_heads:
        gqa_ratio = q_heads / kv_heads

    # Determine architectural family
    fam = meta.architecture_guess
    name_str = " ".join(tensor_names).lower()
    if "expert" in name_str or "router" in name_str:
        fam = "mixture_of_experts"
    elif "conv" in name_str and len(layer_indices) == 0:
        fam = "convolutional_network"
    elif "mamba" in name_str or "ssm" in name_str:
        fam = "state_space_model"
    elif len(layer_indices) > 0:
        fam = "dense_transformer"

    return {
        "family": fam,
        "layer_count": len(layer_indices),
        "layer_indices": layer_indices,
        "hidden_dimension": hidden_dim,
        "query_heads": q_heads,
        "kv_heads": kv_heads,
        "gqa_ratio": gqa_ratio,
        "total_parameters": meta.total_parameters,
        "total_tensors": meta.n_tensors,
    }


def reconstruct_graph(path: str) -> ComputationGraph:
    """Reconstruct module relationships, execution DAG order, and shared parameters."""
    arch = identify_architecture(path)
    nodes: Dict[str, GraphNode] = {}
    execution_order: List[str] = []
    layers: Dict[int, List[str]] = defaultdict(list)
    shapes: Dict[str, List[int]] = {}

    for name, t in ModelLoader.stream_tensors(path):
        shapes[name] = list(t.shape)
        idx = infer_layer(name)
        if idx is not None:
            layers[idx].append(name)

    # 1. Input / Embedding node
    embed_params = [n for n in shapes if re.search(r"embed|wte|token_emb", n)]
    if embed_params:
        nodes["embeddings"] = GraphNode(
            name="embeddings",
            module_type="Embedding",
            layer_index=None,
            input_nodes=["input_tokens"],
            output_nodes=["layer_0"] if 0 in layers else [],
            parameters=embed_params,
            shapes={p: shapes[p] for p in embed_params},
        )
        execution_order.append("embeddings")

    # 2. Sequential Layers
    for idx in sorted(layers.keys()):
        curr_node = f"layer_{idx}"
        prev_node = "embeddings" if idx == 0 else f"layer_{idx-1}"
        next_node = f"layer_{idx+1}" if (idx + 1) in layers else "pre_head_norm"

        nodes[curr_node] = GraphNode(
            name=curr_node,
            module_type="TransformerBlock",
            layer_index=idx,
            input_nodes=[prev_node],
            output_nodes=[next_node],
            parameters=layers[idx],
            shapes={p: shapes[p] for p in layers[idx]},
            is_residual=True,
        )
        execution_order.append(curr_node)

    # 3. Final Norm & Output Head
    head_params = [n for n in shapes if re.search(r"lm_head|output_projection|classifier", n)]
    norm_params = [n for n in shapes if re.search(r"final_layernorm|norm\.weight|ln_f", n)]

    if norm_params:
        last_layer = f"layer_{len(layers)-1}" if layers else "embeddings"
        nodes["pre_head_norm"] = GraphNode(
            name="pre_head_norm",
            module_type="LayerNorm",
            layer_index=None,
            input_nodes=[last_layer],
            output_nodes=["output_head"] if head_params else [],
            parameters=norm_params,
            shapes={p: shapes[p] for p in norm_params},
        )
        execution_order.append("pre_head_norm")

    if head_params:
        input_head = "pre_head_norm" if norm_params else (f"layer_{len(layers)-1}" if layers else "embeddings")
        nodes["output_head"] = GraphNode(
            name="output_head",
            module_type="Linear",
            layer_index=None,
            input_nodes=[input_head],
            output_nodes=["output_logits"],
            parameters=head_params,
            shapes={p: shapes[p] for p in head_params},
        )
        execution_order.append("output_head")

    # Detect tied weights (embeddings == lm_head)
    shared: List[Tuple[str, str]] = []
    if embed_params and head_params:
        for ep in embed_params:
            for hp in head_params:
                if shapes.get(ep) == shapes.get(hp) and ep != hp:
                    shared.append((ep, hp))

    return ComputationGraph(
        nodes=nodes,
        execution_order=execution_order,
        shared_parameters=shared,
        total_layers=len(layers),
        hidden_dim=arch.get("hidden_dimension"),
        output_dim=shapes[head_params[0]][0] if head_params and head_params[0] in shapes else None,
    )
