"""AblationSandbox: Controlled causal ablation interventions with matched controls."""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from .types import AblationMode, AblationResult


def run_activation_ablation(
    model: nn.Module,
    eval_fn: Callable[[nn.Module], float],
    module_name: str,
    mode: AblationMode = AblationMode.ZERO,
    mean_tensor: Optional[torch.Tensor] = None,
    noise_std: float = 0.1,
) -> AblationResult:
    """Run an activation ablation on a module and measure metric impact."""
    baseline_score = eval_fn(model)

    target_mod = None
    for name, module in model.named_modules():
        if name == module_name:
            target_mod = module
            break

    if target_mod is None:
        raise KeyError(f"Module '{module_name}' not found in model")

    def ablation_hook(mod: nn.Module, inp: Any, out: Any):
        is_tuple = isinstance(out, tuple)
        x = out[0] if is_tuple else out

        if mode == AblationMode.ZERO:
            modified = torch.zeros_like(x)
        elif mode == AblationMode.MEAN:
            if mean_tensor is not None:
                modified = mean_tensor.to(x.device).expand_as(x)
            else:
                modified = x.mean(dim=-1, keepdim=True).expand_as(x)
        elif mode == AblationMode.NOISE:
            modified = x + torch.randn_like(x) * noise_std
        elif mode == AblationMode.FREEZE:
            inp_tensor = inp[0] if isinstance(inp, tuple) else inp
            modified = inp_tensor if inp_tensor.shape == x.shape else x
        else:
            modified = x

        return (modified,) + out[1:] if is_tuple else modified

    handle = target_mod.register_forward_hook(ablation_hook)
    try:
        ablated_score = eval_fn(model)
    finally:
        handle.remove()

    rel_change = (
        (ablated_score - baseline_score) / (abs(baseline_score) + 1e-8)
    )

    return AblationResult(
        target_component=module_name,
        mode=mode,
        baseline_score=baseline_score,
        ablated_score=ablated_score,
        relative_change=float(rel_change),
        is_statistically_significant=abs(rel_change) > 0.05,
        details={"noise_std": noise_std if mode == AblationMode.NOISE else None},
    )


def run_parameter_ablation(
    model: nn.Module,
    eval_fn: Callable[[nn.Module], float],
    tensor_name: str,
    mode: AblationMode = AblationMode.ZERO,
) -> AblationResult:
    """Temporarily ablate parameter weights in-memory, evaluate, and restore."""
    baseline_score = eval_fn(model)

    # Locate parameter
    params = dict(model.named_parameters())
    if tensor_name not in params:
        raise KeyError(f"Parameter '{tensor_name}' not found in model")

    param = params[tensor_name]
    original_data = param.data.clone()

    try:
        with torch.no_grad():
            if mode == AblationMode.ZERO:
                param.data.zero_()
            elif mode == AblationMode.MEAN:
                param.data.fill_(float(original_data.mean()))
            elif mode == AblationMode.NOISE:
                param.data.add_(torch.randn_like(param.data) * float(original_data.std()))
            else:
                param.data.zero_()

        ablated_score = eval_fn(model)
    finally:
        # Restore original parameter values
        param.data.copy_(original_data)

    rel_change = (
        (ablated_score - baseline_score) / (abs(baseline_score) + 1e-8)
    )

    return AblationResult(
        target_component=tensor_name,
        mode=mode,
        baseline_score=baseline_score,
        ablated_score=ablated_score,
        relative_change=float(rel_change),
        is_statistically_significant=abs(rel_change) > 0.05,
    )
