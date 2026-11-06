"""EvaluationEngine: Baseline comparison, regression gates, and statistical significance."""

from __future__ import annotations

from typing import Any, Callable, List, Optional

import torch
import torch.nn as nn

from .types import EvaluationResult


def evaluate_metrics(
    model: nn.Module,
    samples: List[Any],
    scoring_fn: Callable[[nn.Module, Any], float],
) -> List[float]:
    """Score model on a list of samples and return per-sample metrics."""
    scores = []
    model.eval()
    with torch.no_grad():
        for s in samples:
            score = scoring_fn(model, s)
            scores.append(float(score))
    return scores


def compare_models(
    base_model: nn.Module,
    candidate_model: nn.Module,
    target_samples: List[Any],
    anchor_samples: List[Any],
    scoring_fn: Callable[[nn.Module, Any], float],
    max_regression_tolerance: float = 0.02,
) -> EvaluationResult:
    """Compare candidate model against base model on target and protected anchor benchmarks."""
    # 1. Target capability evaluation
    base_target_scores = evaluate_metrics(base_model, target_samples, scoring_fn)
    cand_target_scores = evaluate_metrics(candidate_model, target_samples, scoring_fn)

    mean_base_target = sum(base_target_scores) / (len(base_target_scores) + 1e-8)
    mean_cand_target = sum(cand_target_scores) / (len(cand_target_scores) + 1e-8)
    target_delta = mean_cand_target - mean_base_target

    # 2. Protected anchor regression evaluation
    base_anchor_scores = evaluate_metrics(base_model, anchor_samples, scoring_fn)
    cand_anchor_scores = evaluate_metrics(candidate_model, anchor_samples, scoring_fn)

    mean_base_anchor = sum(base_anchor_scores) / (len(base_anchor_scores) + 1e-8)
    mean_cand_anchor = sum(cand_anchor_scores) / (len(cand_anchor_scores) + 1e-8)
    anchor_delta = mean_cand_anchor - mean_base_anchor

    # Regression gate: Anchor performance must not drop by more than tolerance
    passed_gate = anchor_delta >= -abs(max_regression_tolerance)

    # 3. Simple p-value estimate via paired difference
    diffs = [c - b for c, b in zip(cand_target_scores, base_target_scores)]
    p_val: Optional[float] = None
    if len(diffs) > 2:
        try:
            import scipy.stats as stats
            _, p_val = stats.ttest_1samp(diffs, 0.0)
            p_val = float(p_val)
        except Exception:
            p_val = None

    reg_summary = {
        "anchor_base_score": mean_base_anchor,
        "anchor_cand_score": mean_cand_anchor,
        "anchor_delta": anchor_delta,
        "max_regression_tolerance": max_regression_tolerance,
    }

    return EvaluationResult(
        baseline_metric=mean_base_target,
        candidate_metric=mean_cand_target,
        delta=target_delta,
        p_value=p_val,
        passed_regression_gate=passed_gate,
        regression_summary=reg_summary,
    )
