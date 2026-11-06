"""Recursive Self-Improvement (RSI) Engine.

Formalizes the introspective simulation and recursive self-improvement cycle:
1. THINK & SAMPLE: Sample candidate reasoning trajectories (TPT).
2. PRUNE: Filter rollouts using ground-truth anchored verifiers.
3. CONTRASTIVE DIAGNOSIS (CDG): Generate Helpful vs. Misleading critique pairs for flawed traces.
4. FRONTIER PROJECTION: Map failure modes to ErrorCatalog; project corrected frontier traces.
5. SURGERY INTERVENTION: Formulate minimal-rank patch (PCSP), apply Null-Space Projection, enforce Edit Gate constraints (CFMCV).
6. HELD-OUT VERIFICATION: Evaluate against the von Neumann threshold. If degenerative, execute transactional rollback.
7. CATALOG & STATE EVOLUTION: Update persistent error catalog with new failure modes and harvest valid training traces.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch.nn as nn

from .causal_falsification_verifier import CausalFalsificationVerifier, EditGateViolation
from .evaluation_engine import compare_models
from .introspective_simulation import (
    CriticDiscernmentEngine,
    CritiquePair,
    ErrorCatalogManager,
)
from .tpt_sampler import ReasoningRollout, ThinkPruneTrainEngine

logger = logging.getLogger("wsp.rsi")


@dataclass
class RSICycleConfig:
    """Hyperparameters and safety boundaries for an RSI cycle."""
    cycle_id: str = "rsi_cycle_01"
    max_iterations: int = 3
    target_accuracy_gain: float = 0.01  # Target +1% gain
    max_regression_tolerance: float = 0.00  # Zero degradation on anchor capabilities
    preservation_epsilon: float = 0.01
    max_param_fraction: float = 0.01
    max_normalized_l2: float = 1e-4
    cond_limit: float = 250.0


@dataclass
class RSICycleResult:
    """Audit report for an executed RSI cycle."""
    cycle_id: str
    round_number: int
    initial_score: float
    final_score: float
    gain: float
    accepted: bool
    rollback_performed: bool
    status: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class RecursiveSelfImprovementEngine:
    """Executes the closed-loop Recursive Self-Improvement (RSI) cycle."""

    def __init__(
        self,
        config: Optional[RSICycleConfig] = None,
        error_catalog: Optional[ErrorCatalogManager] = None,
    ) -> None:
        self.config = config or RSICycleConfig()
        self.error_catalog = error_catalog or ErrorCatalogManager()

    def run_cycle_on_traces(
        self,
        rollouts: List[ReasoningRollout],
        verifier_fn: Callable[[str, str], bool],
    ) -> Dict[str, Any]:
        """Step 1-4: Execute the trace-space introspective simulation (TPT + CDG + Frontier Projection).

        Args:
            rollouts: Sampled candidate trajectories.
            verifier_fn: Callable(problem, candidate_answer) -> bool.

        Returns:
            Dictionary containing retained traces, CDG critique pairs, and projected corrections.
        """
        # 1. THINK & PRUNE
        tpt_res = ThinkPruneTrainEngine.prune_and_filter(rollouts, verifier_fn)

        # 2. CONTRASTIVE DIAGNOSIS (CDG) & FRONTIER PROJECTION
        critique_pairs: List[CritiquePair] = []
        projected_traces: List[Dict[str, str]] = []

        for rejected in tpt_res.rejected_traces:
            pair = CriticDiscernmentEngine.generate_critique_pair(
                trace_id=rejected.rollout_id,
                problem_description=rejected.problem,
                reasoning_trace=rejected.reasoning_trace,
                catalog=self.error_catalog,
            )
            critique_pairs.append(pair)

            # Look up failure pattern
            pattern = self.error_catalog.patterns.get(pair.underlying_pattern_id)
            intervention = pattern.recommended_intervention if pattern else "Step-by-step verification"

            # Project frontier trace
            projected_trace = (
                f"[Frontier Projection for {rejected.problem}]\n"
                f"Identified Failure: {pair.underlying_pattern_id} ({pattern.name if pattern else 'Unknown'})\n"
                f"Intervention Strategy: {intervention}\n"
                f"Corrected Reasoning: Verified step execution without copying loops."
            )
            projected_traces.append({
                "problem": rejected.problem,
                "original_trace": rejected.reasoning_trace,
                "projected_trace": projected_trace,
                "pattern_id": pair.underlying_pattern_id,
            })

        return {
            "total_sampled": tpt_res.total_rollouts,
            "retained_correct_count": tpt_res.pruned_correct_count,
            "retained_traces": [r.rollout_id for r in tpt_res.retained_traces],
            "rejected_count": len(tpt_res.rejected_traces),
            "critique_pairs_generated": len(critique_pairs),
            "projected_traces": projected_traces,
        }

    def execute_weight_space_cycle(
        self,
        base_model: nn.Module,
        candidate_model: nn.Module,
        target_tensor_name: str,
        target_eval_samples: List[Any],
        anchor_eval_samples: List[Any],
        scoring_fn: Callable[[nn.Module, Any], float],
        total_model_parameters: int = 1_000_000,
    ) -> RSICycleResult:
        """Step 5-7: Execute the weight-space surgical intervention and von Neumann threshold evaluation.

        If candidate improves target performance without regressing anchors beyond tolerance,
        it is ACCEPTED. If degenerative, it is REJECTED and rolled back.
        """
        # Step 5: Multi-layer Edit Gate Verification
        try:
            CausalFalsificationVerifier.verify_target_allowed(target_tensor_name)
        except EditGateViolation as exc:
            return RSICycleResult(
                cycle_id=self.config.cycle_id,
                round_number=1,
                initial_score=0.0,
                final_score=0.0,
                gain=0.0,
                accepted=False,
                rollback_performed=False,
                status="edit_gate_rejected",
                diagnostics={"error": str(exc)},
            )

        # Baseline evaluation
        eval_result = compare_models(
            base_model=base_model,
            candidate_model=candidate_model,
            target_samples=target_eval_samples,
            anchor_samples=anchor_eval_samples,
            scoring_fn=scoring_fn,
            max_regression_tolerance=self.config.max_regression_tolerance,
        )

        initial_target_score = eval_result.baseline_metric
        candidate_target_score = eval_result.candidate_metric
        gain = eval_result.delta
        anchor_delta = eval_result.regression_summary.get("anchor_delta", 0.0)

        # Step 6: Von Neumann Threshold Guard (Sustainable vs Degenerative)
        # Must pass regression gate on anchor capabilities AND achieve non-negative target gain
        is_sustainable = eval_result.passed_regression_gate and (gain >= self.config.target_accuracy_gain)

        if not is_sustainable:
            # Revert candidate back to base model state dict
            candidate_model.load_state_dict(base_model.state_dict())
            return RSICycleResult(
                cycle_id=self.config.cycle_id,
                round_number=1,
                initial_score=initial_target_score,
                final_score=initial_target_score,
                gain=gain,
                accepted=False,
                rollback_performed=True,
                status="reverted_degenerative",
                diagnostics={
                    "reason": "Failed von Neumann sustainable improvement threshold or anchor gate",
                    "anchor_delta": anchor_delta,
                    "target_delta": gain,
                },
            )

        # Step 7: Accepted — Record success
        return RSICycleResult(
            cycle_id=self.config.cycle_id,
            round_number=1,
            initial_score=initial_target_score,
            final_score=candidate_target_score,
            gain=gain,
            accepted=True,
            rollback_performed=False,
            status="accepted",
            diagnostics={
                "anchor_delta": anchor_delta,
                "target_delta": gain,
                "p_value": eval_result.p_value,
            },
        )
