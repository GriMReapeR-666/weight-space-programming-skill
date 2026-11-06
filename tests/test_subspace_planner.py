"""Tests for Subspace Feature Analyzer and Pareto Surgery Planner."""

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from engine import (  # noqa: E402
    CapabilitySubspaceMap,
    ParetoSurgeryPlanner,
    SubspaceProfile,
    compute_spectral_regularization_delta,
    profile_matrix_subspace,
    project_delta_onto_null_space,
)


def test_profile_matrix_subspace_healthy():
    # Orthogonal matrix has optimal condition number 1.0
    q, _ = torch.linalg.qr(torch.randn(32, 32))
    profile = profile_matrix_subspace(q, tensor_name="linear.weight", cond_threshold=50.0)

    assert profile.tensor_name == "linear.weight"
    assert profile.condition_number <= 2.0
    assert profile.effective_rank > 20
    assert len(profile.singular_values_top_5) <= 5
    assert not profile.is_degenerate


def test_profile_matrix_subspace_degenerate():
    # Near-rank-1 matrix with extreme condition number
    u = torch.randn(64, 1)
    v = torch.randn(1, 64)
    w = u @ v + 1e-6 * torch.randn(64, 64)

    profile = profile_matrix_subspace(w, tensor_name="bottleneck.weight", cond_threshold=100.0)
    assert profile.is_degenerate
    assert profile.condition_number > 100.0


def test_compute_spectral_regularization_delta():
    # Construct ill-conditioned matrix
    u = torch.randn(32, 1)
    v = torch.randn(1, 32)
    w = u @ v + 1e-5 * torch.randn(32, 32)

    delta_w, orig_cond, new_cond = compute_spectral_regularization_delta(w, max_condition_number=50.0)
    assert orig_cond > 50.0
    assert new_cond <= 50.0 + 1e-3
    assert delta_w.shape == w.shape


def test_project_delta_onto_null_space_orthogonal():
    in_dim = 16
    out_dim = 8
    # Proposed modification
    delta_w = torch.randn(out_dim, in_dim)
    # Critical anchor directions that must not be perturbed
    anchor_dirs = torch.randn(in_dim, 2)

    safe_delta = project_delta_onto_null_space(delta_w, anchor_dirs)
    assert safe_delta.shape == (out_dim, in_dim)

    # Mathematical invariance verification: safe_delta @ anchor_dirs should be ~ 0
    interference = safe_delta @ anchor_dirs
    assert torch.max(torch.abs(interference)).item() < 1e-4


def test_pareto_surgery_planner_formulation():
    mock_map = CapabilitySubspaceMap(
        model_path="dummy.safetensors",
        total_matrices_profiled=2,
        degenerate_layers=["layer.1.weight"],
        profiles={
            "layer.1.weight": SubspaceProfile(
                tensor_name="layer.1.weight",
                condition_number=12000.0,
                effective_rank=1,
                singular_values_top_5=[10.0, 0.001],
                is_degenerate=True,
                dominant_direction_variance_ratio=0.99,
            )
        },
    )

    plan = ParetoSurgeryPlanner.formulate_plan(
        checkpoint_path="dummy.safetensors",
        objective="Calibrate decision confidence and stabilize spectrum",
        preservation_epsilon=0.01,
        subspace_map=mock_map,
    )

    assert len(plan.actions) == 1
    action = plan.actions[0]
    assert action.target_tensor == "layer.1.weight"
    assert action.operation_type == "spectral_clamp"

    blueprint = ParetoSurgeryPlanner.export_blueprint(plan)
    assert blueprint["plan_id"] == plan.plan_id
    assert blueprint["total_actions"] == 1
