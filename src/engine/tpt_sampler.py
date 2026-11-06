"""Think, Prune, Train (TPT) & Test-Time Recursive Sampler.

Implements ground-truth anchored trajectory filtering and self-distillation (Stanford, ICLR 2025 Workshop):
1. THINK: Samples k reasoning candidate rollouts per task.
2. PRUNE: Rejects hallucinations by verifying each candidate against ground truth (unit tests, execution, exact match).
3. TRAIN / HARVEST: Formats valid traces into supervised fine-tuning pairs and error catalog updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class ReasoningRollout:
    rollout_id: str
    problem: str
    reasoning_trace: str
    candidate_answer: str
    is_verifiably_correct: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TPTBatchResult:
    total_rollouts: int
    pruned_correct_count: int
    retained_traces: List[ReasoningRollout] = field(default_factory=list)
    rejected_traces: List[ReasoningRollout] = field(default_factory=list)
    success_rate: float = 0.0


class ThinkPruneTrainEngine:
    """Executes the Think-Prune-Train ground-truth anchored sampling loop."""

    @staticmethod
    def prune_and_filter(
        rollouts: List[ReasoningRollout],
        verifier_fn: Callable[[str, str], bool],
    ) -> TPTBatchResult:
        """Evaluates rollouts against an external verifier and separates correct from incorrect traces.

        Args:
            rollouts: List of sampled reasoning trajectories.
            verifier_fn: Callable(problem, candidate_answer) -> bool.

        Returns:
            TPTBatchResult summarizing accepted vs. rejected trajectories.
        """
        retained: List[ReasoningRollout] = []
        rejected: List[ReasoningRollout] = []

        for r in rollouts:
            is_valid = verifier_fn(r.problem, r.candidate_answer)
            r.is_verifiably_correct = is_valid
            if is_valid:
                retained.append(r)
            else:
                rejected.append(r)

        total = len(rollouts)
        success_rate = (len(retained) / total) if total > 0 else 0.0

        return TPTBatchResult(
            total_rollouts=total,
            pruned_correct_count=len(retained),
            retained_traces=retained,
            rejected_traces=rejected,
            success_rate=success_rate,
        )

    @staticmethod
    def export_sft_dataset(
        batch: TPTBatchResult,
    ) -> List[Dict[str, str]]:
        """Formats accepted rollouts into supervised fine-tuning pairs."""
        return [
            {
                "instruction": r.problem,
                "input": "",
                "output": f"<think>\n{r.reasoning_trace}\n</think>\n{r.candidate_answer}",
            }
            for r in batch.retained_traces
        ]
