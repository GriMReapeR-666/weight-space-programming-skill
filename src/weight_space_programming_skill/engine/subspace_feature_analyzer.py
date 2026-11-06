"""Subspace Latency & Feature Analyzer (SLFA).

Analyzes singular subspaces, condition numbers, and feature directions of weight matrices
and correlates activation representations with behavioral probes to distinguish between:
1. Latent capabilities (represented along secondary singular vectors but suppressed)
2. Active circuits (well-conditioned primary singular directions)
3. Degenerate noise directions (high condition numbers kappa(W) > threshold)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch

from .loader import ModelLoader
from .types import CapabilitySubspaceMap, SubspaceProfile


def profile_matrix_subspace(
    weight: torch.Tensor,
    tensor_name: str,
    cond_threshold: float = 500.0,
    probe_direction: Optional[torch.Tensor] = None,
) -> SubspaceProfile:
    """Computes singular value spectrum, effective rank, and directional alignment.

    Args:
        weight: 2D weight matrix [out_features, in_features].
        tensor_name: Name identifier for the tensor.
        cond_threshold: Condition number threshold above which the matrix is considered degenerate.
        probe_direction: Optional vector [in_features] or [out_features] to test directional alignment.

    Returns:
        SubspaceProfile containing condition number, effective rank, and alignment.
    """
    w_fp32 = weight.detach().float()
    if w_fp32.ndim != 2:
        raise ValueError(f"profile_matrix_subspace expects a 2D matrix, got shape {list(weight.shape)}")

    # Full singular value decomposition
    U, S, Vh = torch.linalg.svd(w_fp32, full_matrices=False)

    s_max = float(S[0].item())
    s_min = float(S[-1].item()) if S[-1].item() > 1e-12 else 1e-12
    cond = float(s_max / s_min)

    # Singular value variance ratio for dominant mode
    total_var = float(torch.sum(S ** 2).item())
    dom_var_ratio = float((S[0] ** 2).item() / (total_var + 1e-12))

    # Effective rank (Roy & Vetterli entropy-based formulation: exp(H(p)))
    norm_s = S / (torch.sum(S) + 1e-12)
    entropy = -float(torch.sum(norm_s * torch.log(norm_s + 1e-12)).item())
    effective_rank = max(1, int(round(math.exp(entropy))))

    top_5_s = [round(float(s.item()), 4) for s in S[:5]]
    is_degenerate = cond > cond_threshold or effective_rank < max(1, int(min(w_fp32.shape) * 0.05))

    # Optional directional alignment score (cosine similarity between probe and top singular direction)
    alignment_score: Optional[float] = None
    if probe_direction is not None:
        p = probe_direction.detach().float().flatten()
        if p.shape[0] == Vh.shape[1]:
            # Align with input singular subspace (right singular vector)
            v0 = Vh[0]
            cos_sim = torch.dot(p, v0) / (torch.norm(p) * torch.norm(v0) + 1e-12)
            alignment_score = float(torch.abs(cos_sim).item())
        elif p.shape[0] == U.shape[0]:
            # Align with output singular subspace (left singular vector)
            u0 = U[:, 0]
            cos_sim = torch.dot(p, u0) / (torch.norm(p) * torch.norm(u0) + 1e-12)
            alignment_score = float(torch.abs(cos_sim).item())

    return SubspaceProfile(
        tensor_name=tensor_name,
        condition_number=cond,
        effective_rank=effective_rank,
        singular_values_top_5=top_5_s,
        is_degenerate=is_degenerate,
        dominant_direction_variance_ratio=dom_var_ratio,
        probe_alignment_score=alignment_score,
    )


def profile_model_subspaces(
    checkpoint_path: str,
    cond_threshold: float = 500.0,
    max_tensors: Optional[int] = None,
) -> CapabilitySubspaceMap:
    """Profiles all 2D linear weight matrices in a model checkpoint.

    Args:
        checkpoint_path: Path to .safetensors, .pt, or .bin checkpoint.
        cond_threshold: Condition number threshold for degeneracy flags.
        max_tensors: Optional limit on the number of matrices to profile.

    Returns:
        CapabilitySubspaceMap summarizing healthy, latent, and degenerate subspaces.
    """
    profiles: Dict[str, SubspaceProfile] = {}
    degenerate_layers: List[str] = []
    latent_capabilities: List[str] = []
    active_circuits: List[str] = []

    count = 0
    for name, tensor in ModelLoader.stream_tensors(checkpoint_path):
        # We only profile 2D linear and projection weights
        if tensor.ndim == 2 and not name.endswith(".bias"):
            prof = profile_matrix_subspace(
                weight=tensor,
                tensor_name=name,
                cond_threshold=cond_threshold,
            )
            profiles[name] = prof
            if prof.is_degenerate:
                degenerate_layers.append(name)
            elif prof.dominant_direction_variance_ratio < 0.15:
                # Rich multi-dimensional representations (active general circuit)
                active_circuits.append(name)
            else:
                # Low-rank or strongly peaked singular structure
                latent_capabilities.append(name)

            count += 1
            if max_tensors and count >= max_tensors:
                break

    return CapabilitySubspaceMap(
        model_path=checkpoint_path,
        total_matrices_profiled=len(profiles),
        degenerate_layers=degenerate_layers,
        profiles=profiles,
        latent_capabilities=latent_capabilities,
        active_circuits=active_circuits,
    )


def compute_spectral_regularization_delta(
    weight: torch.Tensor,
    max_condition_number: float = 250.0,
) -> Tuple[torch.Tensor, float, float]:
    """Computes a regularizing weight delta Delta W that clamps condition number without altering primary modes.

    Args:
        weight: 2D weight matrix [M, N].
        max_condition_number: Maximum allowed ratio sigma_max / sigma_min.

    Returns:
        (delta_w, original_condition_number, new_condition_number)
    """
    orig_dtype = weight.dtype
    w_fp32 = weight.detach().float()
    U, S, Vh = torch.linalg.svd(w_fp32, full_matrices=False)

    orig_cond = float(S[0].item() / (S[-1].item() + 1e-12))
    if orig_cond <= max_condition_number:
        return torch.zeros_like(weight), orig_cond, orig_cond

    min_allowed_sigma = S[0].item() / max_condition_number
    S_new = torch.clamp(S, min=min_allowed_sigma)
    new_cond = float(S_new[0].item() / S_new[-1].item())

    # Reconstruct regularized matrix
    W_reg = (U @ torch.diag(S_new) @ Vh).to(orig_dtype)
    delta_w = W_reg - weight
    return delta_w, orig_cond, new_cond


def project_delta_onto_null_space(
    candidate_delta: torch.Tensor,
    anchor_directions: torch.Tensor,
) -> torch.Tensor:
    """Projects candidate weight updates Delta W onto the orthogonal complement (null space) of anchor directions.

    Ensures Delta W_safe * V_anchor = 0, mathematically preventing regression on preserved anchor tasks.

    Args:
        candidate_delta: Proposed weight modification Delta W [out_dim, in_dim].
        anchor_directions: Matrix of critical preserved input directions V_anchor [in_dim, K].

    Returns:
        Safe projected weight delta Delta W_safe [out_dim, in_dim].
    """
    orig_dtype = candidate_delta.dtype
    dW = candidate_delta.detach().float()
    V = anchor_directions.detach().float()

    in_dim = dW.shape[1]
    if V.shape[0] != in_dim:
        raise ValueError(f"Anchor directions dim 0 ({V.shape[0]}) must match delta in_dim ({in_dim})")

    # Orthonormalize anchor basis Q
    Q, _ = torch.linalg.qr(V)  # [in_dim, K]
    # Projection operator onto anchor subspace: P = Q @ Q.T
    # Safe delta: dW_safe = dW @ (I - P)
    P = Q @ Q.T
    eye = torch.eye(in_dim, dtype=torch.float32, device=dW.device)
    dW_safe = dW @ (eye - P)

    return dW_safe.to(orig_dtype)
