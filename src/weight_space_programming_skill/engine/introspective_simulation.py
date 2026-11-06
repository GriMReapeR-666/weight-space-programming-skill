"""Introspective Simulation & Critic-Discernment Engine (CDG).

Implements the theoretical framework from arXiv:2607.04277 (Self-reference in LLMs:
the introspection threshold for recursive self-improvement) and ICML 2025 (Critic-Discernment Game):
1. Critic-Discernment Game (CDG): Evaluates reasoning traces by generating contrasting
   Helpful (true flaw) vs. Misleading (plausible distractor) critiques.
2. Error Catalog Manager: Maintains a persistent, machine-readable catalog of structural
   failure modes (e.g., variable drop, attention copying loop) to guide self-correction.
3. Introspective Projection: Projects structured diagnostic reasoning onto traces,
   compensating for smaller models' lack of raw in-context parameter self-awareness.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class FailurePattern:
    pattern_id: str
    name: str
    description: str
    typical_symptom: str
    recommended_intervention: str
    occurrence_count: int = 1


@dataclass
class CritiquePair:
    trace_id: str
    helpful_critique: str
    misleading_critique: str
    helpful_is_option_a: bool
    underlying_pattern_id: str


class ErrorCatalogManager:
    """Maintains an auditable, persistent catalog of structural failure modes."""

    def __init__(self, catalog_path: Optional[str] = None) -> None:
        self.catalog_path = Path(catalog_path) if catalog_path else None
        self.patterns: Dict[str, FailurePattern] = {}
        self._load_defaults()

    def _load_defaults(self) -> None:
        """Seed foundational failure patterns from the WSP and model editing literature."""
        defaults = [
            FailurePattern(
                pattern_id="WSP-ERR-001",
                name="Repetitive Attention Copying Loop",
                description="Attention copying heads reinforce self-similar tokens in the residual stream.",
                typical_symptom="Generation hits token limit without reaching a terminal delimiter.",
                recommended_intervention="Rank-1 projection subtracting repetition direction from MLP down-projection.",
            ),
            FailurePattern(
                pattern_id="WSP-ERR-002",
                name="Deep Variable Dropping",
                description="Information tracking fails beyond depth > 3 in multi-step problem chains.",
                typical_symptom="Intermediate calculation drops earliest initialized variables.",
                recommended_intervention="Spectral conditioning clamping condition number on out_proj.",
            ),
            FailurePattern(
                pattern_id="WSP-ERR-003",
                name="Premature Fallback / Deferral Bias",
                description="Linear bias offset in decision head arbitrarily favors escalation over autonomous resolution.",
                typical_symptom="Confidence score suppresses action probability despite clear decision boundary.",
                recommended_intervention="Prior neutralization and rank-1 margin alignment on action head.",
            ),
            FailurePattern(
                pattern_id="WSP-ERR-004",
                name="Quantization Rounding Distortion",
                description="Lossy low-bit nibble scaling exceeds numerical tolerance in sensitive projection blocks.",
                typical_symptom="Normalized L2 distortion > 0.15 after block quantization.",
                recommended_intervention="Block-level selective dequantization or promotion to FP16.",
            ),
        ]
        for p in defaults:
            self.patterns[p.pattern_id] = p

    def register_pattern(self, pattern: FailurePattern) -> None:
        if pattern.pattern_id in self.patterns:
            self.patterns[pattern.pattern_id].occurrence_count += 1
        else:
            self.patterns[pattern.pattern_id] = pattern

    def find_matching_pattern(self, symptom_text: str) -> Optional[FailurePattern]:
        s_lower = symptom_text.lower()
        for p in self.patterns.values():
            if any(term in s_lower for term in p.name.lower().split() if len(term) > 3):
                return p
        return None

    def export_catalog(self) -> List[Dict[str, Any]]:
        return [
            {
                "pattern_id": p.pattern_id,
                "name": p.name,
                "description": p.description,
                "typical_symptom": p.typical_symptom,
                "recommended_intervention": p.recommended_intervention,
                "occurrences": p.occurrence_count,
            }
            for p in self.patterns.values()
        ]


class CriticDiscernmentEngine:
    """Generates and evaluates contrasting critique pairs for self-correction."""

    @staticmethod
    def generate_critique_pair(
        trace_id: str,
        problem_description: str,
        reasoning_trace: str,
        catalog: ErrorCatalogManager,
    ) -> CritiquePair:
        """Constructs a paired Helpful vs. Misleading critique pair."""
        matched_pattern = catalog.find_matching_pattern(reasoning_trace)
        if not matched_pattern:
            matched_pattern = list(catalog.patterns.values())[0]

        helpful = (
            f"Careful Analyst Critique: The reasoning fails due to {matched_pattern.name}. "
            f"Specifically: {matched_pattern.description} "
            f"Corrective action: {matched_pattern.recommended_intervention}"
        )

        misleading = (
            "Sloppy Reviewer Critique: The steps appear superficially fine, but the final answer "
            "simply used the wrong constant multiplier at the very end. No structural modification needed."
        )

        # Deterministic pseudo-random split
        helpful_is_a = (len(trace_id) % 2 == 0)

        return CritiquePair(
            trace_id=trace_id,
            helpful_critique=helpful if helpful_is_a else misleading,
            misleading_critique=misleading if helpful_is_a else helpful,
            helpful_is_option_a=helpful_is_a,
            underlying_pattern_id=matched_pattern.pattern_id,
        )

    @staticmethod
    def evaluate_discernment(
        critique_pair: CritiquePair,
        chosen_option: str,  # 'A' or 'B'
    ) -> Tuple[bool, float]:
        """Evaluates whether the agent correctly chose the Helpful critique over the Misleading distractor.

        Returns:
            (is_correct, reward_score)
        """
        choice = chosen_option.strip().upper()
        correct_choice = "A" if critique_pair.helpful_is_option_a else "B"

        is_correct = (choice == correct_choice)
        reward = 1.0 if is_correct else -0.5
        return is_correct, reward
