"""Causal Falsification & Matched-Control Verifier (CFMCV).

Implements the multi-layered Edit Gate policy system:
1. Scope restrictions: locks input embeddings, final RMSNorm, and output heads.
2. Fraction-changed cap: enforces 1% threshold on parameter count.
3. Normalized L2 distance check: prevents disabling edits (||Delta||_2 / N < threshold).
4. Whole-tensor uniform scaling detection: rejects scalar knob modifications.
5. Matched Gaussian Control Verifier: tests candidate deltas against randomized
   orthogonal perturbations with matched Frobenius norm (||Delta_rand||_F == ||Delta||_F)
   to falsify spurious correlation hypotheses.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Tuple

import torch


class EditGateViolation(Exception):
    """Raised when a candidate edit violates structural or numerical safety invariants."""
    pass


class CausalFalsificationVerifier:
    """Multi-layer safety gate and causal control verifier."""

    # Permanent forbidden tensor patterns
    FORBIDDEN_PATTERNS = [
        r"embed",
        r"wte",
        r"wpe",
        r"lm_head",
        r"output\.weight",
        r"final_norm",
        r"norm\.weight$",
        r"ln_f",
    ]

    # Safe limits
    MAX_FRACTION_MODIFIED = 0.01  # 1% cap
    MAX_NORMALIZED_L2_DISTANCE = 1e-4  # Disabling edit threshold per Gupta et al. (2024)

    @classmethod
    def verify_target_allowed(cls, tensor_name: str) -> None:
        """Verifies that the target tensor is not in the forbidden scope."""
        name_lower = tensor_name.lower()
        for pat in cls.FORBIDDEN_PATTERNS:
            if re.search(pat, name_lower):
                raise EditGateViolation(
                    f"Edit target '{tensor_name}' matches forbidden scope pattern '{pat}'. "
                    f"Modifications to input embeddings, final norms, and output heads are prohibited."
                )

    @classmethod
    def verify_no_uniform_scaling(cls, original: torch.Tensor, candidate: torch.Tensor) -> None:
        """Rejects edits that represent uniform scalar multiplication of an entire tensor."""
        orig_fp = original.float()
        cand_fp = candidate.float()

        # Check if candidate / original is virtually constant across non-zero elements
        mask = torch.abs(orig_fp) > 1e-7
        if torch.sum(mask) == 0:
            return

        ratios = cand_fp[mask] / orig_fp[mask]
        ratio_std = float(torch.std(ratios).item())
        ratio_mean = float(torch.mean(ratios).item())

        # If variance across ratios is near 0 and mean != 1, it's uniform scaling
        if ratio_std < 1e-5 and abs(ratio_mean - 1.0) > 1e-6:
            raise EditGateViolation(
                f"Candidate edit represents uniform whole-tensor scaling (ratio={ratio_mean:.6f}). "
                f"Whole-tensor scalar adjustments are forbidden by the edit gate."
            )

    @classmethod
    def verify_edit_distance_and_fraction(
        cls,
        original: torch.Tensor,
        candidate: torch.Tensor,
        total_model_parameters: int,
    ) -> Dict[str, float]:
        """Enforces fraction cap (<= 1%) and normalized L2 distance (<= 1e-4)."""
        delta = candidate.float() - original.float()

        # Count modified elements
        modified_mask = torch.abs(delta) > 1e-9
        num_modified = int(torch.sum(modified_mask).item())

        fraction_changed = num_modified / max(1, total_model_parameters)
        if fraction_changed > cls.MAX_FRACTION_MODIFIED:
            raise EditGateViolation(
                f"Edit modified {num_modified} parameters ({fraction_changed:.4%}), "
                f"exceeding the safety cap of {cls.MAX_FRACTION_MODIFIED:.1%}."
            )

        # Normalized L2 distance: ||Delta||_2 / N
        frobenius_norm = float(torch.linalg.norm(delta).item())
        tensor_elements = original.numel()
        norm_l2_dist = frobenius_norm / max(1, tensor_elements)

        if norm_l2_dist > cls.MAX_NORMALIZED_L2_DISTANCE:
            raise EditGateViolation(
                f"Normalized L2 distance {norm_l2_dist:.2e} exceeds the safe threshold "
                f"{cls.MAX_NORMALIZED_L2_DISTANCE:.2e} (potential catastrophic forgetting)."
            )

        return {
            "num_modified": num_modified,
            "fraction_model_modified": fraction_changed,
            "normalized_l2_distance": norm_l2_dist,
            "frobenius_norm": frobenius_norm,
        }

    @classmethod
    def generate_matched_control_delta(cls, real_delta: torch.Tensor) -> torch.Tensor:
        """Generates a randomized Gaussian perturbation with identical Frobenius norm ||Delta||_F."""
        real_fp = real_delta.float()
        target_norm = torch.linalg.norm(real_fp)

        # Sample random Gaussian noise
        random_delta = torch.randn_like(real_fp)
        random_norm = torch.linalg.norm(random_delta)

        # Scale random delta to match exact Frobenius norm
        matched_control = (random_delta / (random_norm + 1e-12)) * target_norm
        return matched_control.to(real_delta.dtype)

    @classmethod
    def run_falsification_check(
        cls,
        eval_fn,
        original_model: Any,
        real_delta_fn,
        control_delta_fn,
    ) -> Tuple[bool, Dict[str, float]]:
        """Tests if the real intervention significantly outperforms the matched random control.

        Args:
            eval_fn: Callable taking a model and returning a performance metric score (float).
            original_model: Base model.
            real_delta_fn: Context manager or hook applying the real surgical delta.
            control_delta_fn: Context manager or hook applying the matched control delta.

        Returns:
            (is_causally_valid, metrics_dict)
        """
        baseline_score = eval_fn(original_model)

        with real_delta_fn():
            real_score = eval_fn(original_model)

        with control_delta_fn():
            control_score = eval_fn(original_model)

        real_gain = real_score - baseline_score
        control_gain = control_score - baseline_score

        # If control achieves comparable or better gain, the hypothesis is FALSIFIED
        is_valid = real_gain > (control_gain + 1e-4)

        return is_valid, {
            "baseline_score": baseline_score,
            "real_score": real_score,
            "control_score": control_score,
            "real_gain": real_gain,
            "control_gain": control_gain,
        }
