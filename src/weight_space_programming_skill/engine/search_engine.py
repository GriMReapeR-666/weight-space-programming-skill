"""SearchEngine: Bounded gradient-free parameter optimization in weight-space."""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import torch
import torch.nn as nn

from .types import PatchOperation


def evaluate_candidate(
    model: nn.Module,
    eval_fn: Callable[[nn.Module], float],
    tensor_name: str,
    candidate_delta: torch.Tensor,
) -> float:
    """Evaluate candidate parameter delta without permanent in-place mutation."""
    params = dict(model.named_parameters())
    if tensor_name not in params:
        raise KeyError(f"Parameter '{tensor_name}' not found in model")

    target = params[tensor_name]
    with torch.no_grad():
        target.add_(candidate_delta)
        try:
            score = eval_fn(model)
        finally:
            target.sub_(candidate_delta)

    return score


def optimize_patch(
    model: nn.Module,
    eval_fn: Callable[[nn.Module], float],
    tensor_name: str,
    max_iterations: int = 15,
    step_size: float = 0.01,
    n_samples_per_step: int = 4,
    maximize: bool = True,
) -> Tuple[PatchOperation, float, Dict[str, Any]]:
    """Search for an optimal bounded parameter delta using zero-order coordinate descent."""
    params = dict(model.named_parameters())
    if tensor_name not in params:
        raise KeyError(f"Parameter '{tensor_name}' not found in model")

    target = params[tensor_name]
    shape = list(target.shape)

    # Establish baseline
    best_score = eval_fn(model)
    best_delta = torch.zeros_like(target.data)
    history = [best_score]

    for iteration in range(max_iterations):
        improved = False
        candidates = []

        for _ in range(n_samples_per_step):
            # Directional random unit perturbation
            direction = torch.randn_like(target.data)
            norm = torch.linalg.vector_norm(direction)
            if norm > 0:
                direction.div_(norm)

            for sign in (1.0, -1.0):
                cand_delta = best_delta + (sign * step_size * direction)
                score = evaluate_candidate(model, eval_fn, tensor_name, cand_delta)
                candidates.append((cand_delta, score))

        # Select best candidate
        candidates.sort(key=lambda x: x[1], reverse=maximize)
        top_delta, top_score = candidates[0]

        if (maximize and top_score > best_score) or (not maximize and top_score < best_score):
            best_score = top_score
            best_delta = top_delta
            improved = True

        history.append(best_score)
        if not improved:
            # Decay step size if no progress
            step_size *= 0.70

    patch_op = PatchOperation(
        tensor_name=tensor_name,
        operation="add",
        shape=shape,
        data=best_delta.cpu(),
        alpha=1.0,
    )

    metrics = {
        "iterations": max_iterations,
        "final_step_size": step_size,
        "score_history": history,
        "delta_l2_norm": float(torch.linalg.vector_norm(best_delta)),
    }

    return patch_op, best_score, metrics
