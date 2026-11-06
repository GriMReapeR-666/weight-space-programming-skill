"""ActivationEngine: Runtime forward-hook tracing, hidden state monitoring, and activation stats."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from .types import ActivationTrace


class ActivationTracer:
    """Non-intrusive runtime forward hook harness for monitoring internal representations."""

    def __init__(self) -> None:
        self.traces: List[ActivationTrace] = []
        self._handles: List[Any] = []

    def attach(self, model: nn.Module, target_layers: List[str]) -> None:
        """Register forward hooks onto targeted module names."""
        self.clear()
        for name, module in model.named_modules():
            if name in target_layers:
                handle = module.register_forward_hook(self._create_hook(name))
                self._handles.append(handle)

    def _create_hook(self, name: str):
        def hook(module: nn.Module, inputs: Any, output: Any):
            out_tensor = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(out_tensor):
                return

            with torch.no_grad():
                detached = out_tensor.detach().float().cpu()
                numel = detached.numel()
                if numel == 0:
                    return

                trace = ActivationTrace(
                    module_name=name,
                    step_index=len(self.traces),
                    mean=float(detached.mean()),
                    std=float(detached.std(unbiased=False)),
                    max_abs=float(detached.abs().max()),
                    l2_norm=float(torch.linalg.vector_norm(detached)),
                    sparsity=float((detached == 0).float().mean()),
                )
                self.traces.append(trace)
        return hook

    def clear(self) -> None:
        """Remove all active PyTorch hooks and reset collected traces."""
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()
        self.traces.clear()

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Return aggregated summary metrics per hooked module."""
        summary: Dict[str, Dict[str, float]] = {}
        for t in self.traces:
            if t.module_name not in summary:
                summary[t.module_name] = {
                    "mean_activation": t.mean,
                    "std_activation": t.std,
                    "max_abs": t.max_abs,
                    "mean_norm": t.l2_norm,
                    "sparsity": t.sparsity,
                }
        return summary


def capture_activations(
    model: nn.Module, inputs: Any, target_layers: List[str]
) -> List[ActivationTrace]:
    """Execute a forward pass with hooked modules and capture activation statistics."""
    tracer = ActivationTracer()
    tracer.attach(model, target_layers)
    try:
        model.eval()
        with torch.no_grad():
            if isinstance(inputs, dict):
                model(**inputs)
            elif isinstance(inputs, (list, tuple)):
                model(*inputs)
            else:
                model(inputs)
    finally:
        traces = list(tracer.traces)
        tracer.clear()
    return traces


def trace_generation(
    model: nn.Module,
    generate_fn: Any,
    prompt: Any,
    target_layers: List[str],
) -> List[ActivationTrace]:
    """Capture layer activations across an entire generation sequence."""
    tracer = ActivationTracer()
    tracer.attach(model, target_layers)
    try:
        model.eval()
        with torch.no_grad():
            generate_fn(prompt)
    finally:
        traces = list(tracer.traces)
        tracer.clear()
    return traces
