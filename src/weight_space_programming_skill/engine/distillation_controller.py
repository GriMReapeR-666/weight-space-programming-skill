"""Weight-Space Distillation & Fine-Tuning Controller.

Integrates Weight-Space Programming (WSP) with conventional fine-tuning and
knowledge distillation:
1. Target layer identification via spectral profiling (high condition number = candidate for LoRA/distillation).
2. Distillation loss formulation with temperature scaling:
   L_total = alpha * L_CE(student, targets) + (1 - alpha) * T^2 * L_KD(student, teacher)
3. Null-space protected distillation weight updates:
   Delta W_safe = Delta W_distill * (I - Q Q^T)
4. Preservation gating to prevent catastrophic forgetting during distillation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch
import torch.nn.functional as F

from .causal_falsification_verifier import CausalFalsificationVerifier
from .subspace_feature_analyzer import profile_model_subspaces, project_delta_onto_null_space


@dataclass
class DistillationLayerPlan:
    """Strategy assigned to an individual model layer during knowledge distillation."""
    layer_name: str
    strategy: str  # "lora", "null_space_projected", "activation_match", "freeze"
    recommended_rank: int
    condition_number: float
    temperature: float
    kl_weight: float


@dataclass
class DistillationRecommendation:
    """Comprehensive blueprint for weight-space guided knowledge distillation."""
    student_checkpoint: str
    total_layers_analyzed: int
    target_layers_to_train: List[str]
    protected_layers_to_freeze: List[str]
    layer_plans: Dict[str, DistillationLayerPlan] = field(default_factory=dict)
    default_temperature: float = 2.0
    alpha_ce: float = 0.5


class DistillationController:
    """Orchestrates weight-space guided distillation and fine-tuning."""

    @staticmethod
    def plan_distillation_strategy(
        student_checkpoint: str,
        cond_limit: float = 250.0,
        lora_rank_default: int = 8,
    ) -> DistillationRecommendation:
        """Inspects student parameter manifolds to select optimal distillation targets.

        Layers with elevated condition numbers or manifold bottlenecks are prioritized
        for rank-restricted LoRA/distillation; embeddings and output heads are locked.
        """
        subspace_map = profile_model_subspaces(student_checkpoint, cond_threshold=cond_limit)

        targets_to_train: List[str] = []
        frozen_layers: List[str] = []
        layer_plans: Dict[str, DistillationLayerPlan] = {}

        for name, profile in subspace_map.profiles.items():
            # Check if layer is forbidden from modification by Edit Gate policy
            is_forbidden = False
            try:
                CausalFalsificationVerifier.verify_target_allowed(name)
            except Exception:
                is_forbidden = True

            if is_forbidden:
                frozen_layers.append(name)
                layer_plans[name] = DistillationLayerPlan(
                    layer_name=name,
                    strategy="freeze",
                    recommended_rank=0,
                    condition_number=profile.condition_number,
                    temperature=1.0,
                    kl_weight=0.0,
                )
            elif profile.is_degenerate or profile.condition_number > 150.0:
                # Degenerate manifolds need low-rank LoRA stabilization
                targets_to_train.append(name)
                layer_plans[name] = DistillationLayerPlan(
                    layer_name=name,
                    strategy="lora",
                    recommended_rank=lora_rank_default,
                    condition_number=profile.condition_number,
                    temperature=2.5,
                    kl_weight=0.7,
                )
            else:
                # Well-conditioned layers can use null-space protected distillation
                targets_to_train.append(name)
                layer_plans[name] = DistillationLayerPlan(
                    layer_name=name,
                    strategy="null_space_projected",
                    recommended_rank=lora_rank_default * 2,
                    condition_number=profile.condition_number,
                    temperature=2.0,
                    kl_weight=0.5,
                )

        return DistillationRecommendation(
            student_checkpoint=student_checkpoint,
            total_layers_analyzed=len(subspace_map.profiles),
            target_layers_to_train=targets_to_train,
            protected_layers_to_freeze=frozen_layers,
            layer_plans=layer_plans,
        )

    @staticmethod
    def compute_kd_loss(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 2.0,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """Computes standard knowledge distillation loss with temperature scaling:

        L = alpha * L_CE(student, labels) + (1 - alpha) * T^2 * KL(soft_student || soft_teacher)
        """
        ce_loss = F.cross_entropy(student_logits, labels)

        soft_student = F.log_softmax(student_logits / temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)

        kl_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)

        return alpha * ce_loss + (1.0 - alpha) * kl_loss

    @staticmethod
    def apply_protected_distillation_update(
        original_weight: torch.Tensor,
        distilled_delta: torch.Tensor,
        protected_basis: torch.Tensor,
    ) -> torch.Tensor:
        """Projects distillation gradient updates onto null space to safeguard anchor capabilities."""
        safe_delta = project_delta_onto_null_space(distilled_delta, protected_basis)
        return original_weight + safe_delta
