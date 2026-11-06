"""CircuitMapper: Map behavioral capabilities to specific model layers, heads, and modules."""

from __future__ import annotations

import re
from typing import Callable, List

import torch
import torch.nn as nn

from .types import CircuitMap, CircuitNode


def build_circuit_map(
    model: nn.Module,
    eval_fn: Callable[[nn.Module], float],
    target_modules: List[str],
    behavioral_target: str,
) -> CircuitMap:
    """Systematically probe modules to build a causal circuit map of behavioral influence."""
    baseline_score = eval_fn(model)
    critical_nodes: List[CircuitNode] = []
    edges = []

    for mod_name in target_modules:
        target_mod = None
        for name, module in model.named_modules():
            if name == mod_name:
                target_mod = module
                break

        if target_mod is None:
            continue

        # Extract layer index if present
        m = re.search(r"(?:layers?|blocks?|blk)\.(\d+)", mod_name)
        layer_idx = int(m.group(1)) if m else 0

        # Hook to zero-ablate the target module output
        def zero_ablation_hook(m, inp, out):
            if isinstance(out, tuple):
                return (torch.zeros_like(out[0]),) + out[1:]
            return torch.zeros_like(out)

        handle = target_mod.register_forward_hook(zero_ablation_hook)
        try:
            ablated_score = eval_fn(model)
            impact = abs(baseline_score - ablated_score) / (abs(baseline_score) + 1e-8)
        finally:
            handle.remove()

        # Hypothesize role based on naming and impact
        role = "neutral_pathway"
        if impact > 0.40:
            role = "critical_causal_hub"
        elif impact > 0.15:
            role = "contributing_component"

        node = CircuitNode(
            component=mod_name,
            layer_index=layer_idx,
            importance_score=float(impact),
            ablation_impact=float(baseline_score - ablated_score),
            hypothesized_role=role,
        )
        critical_nodes.append(node)

    # Sort nodes by causal importance descending
    critical_nodes.sort(key=lambda n: n.importance_score, reverse=True)

    # Build sequence edges between adjacent layers
    for i in range(len(critical_nodes) - 1):
        src = critical_nodes[i]
        dst = critical_nodes[i + 1]
        weight = float((src.importance_score + dst.importance_score) / 2.0)
        edges.append((src.component, dst.component, weight))

    summary = (
        f"Circuit map for '{behavioral_target}': "
        f"Identified {len(critical_nodes)} probed components; "
        f"top causal hub: {critical_nodes[0].component if critical_nodes else 'none'} "
        f"with impact score {critical_nodes[0].importance_score:.4f}"
    )

    return CircuitMap(
        target_behavior=behavioral_target,
        critical_nodes=critical_nodes,
        causal_edges=edges,
        summary=summary,
    )
