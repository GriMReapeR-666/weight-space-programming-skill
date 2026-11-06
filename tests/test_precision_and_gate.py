"""Tests for SelectivePrecisionController and CausalFalsificationVerifier."""

import pytest

torch = pytest.importorskip("torch")

from engine import (  # noqa: E402
    CausalFalsificationVerifier,
    EditGateViolation,
    SelectivePrecisionController,
)


def test_q8_0_block_roundtrip():
    # 32 random values
    original = torch.randn(32) * 5.0
    quant_bytes, scale = SelectivePrecisionController.requantize_q8_0_block(original)
    assert len(quant_bytes) == 32
    assert scale > 0

    dequant = SelectivePrecisionController.dequantize_q8_0_block(quant_bytes, scale)
    assert dequant.shape == (32,)

    # Max reconstruction error for Q8_0 is bounded by scale / 2
    max_err = float(torch.max(torch.abs(original - dequant)).item())
    assert max_err <= scale + 1e-4


def test_q4_0_block_roundtrip():
    # 32 random values
    original = torch.randn(32) * 3.0
    nibble_bytes, scale = SelectivePrecisionController.requantize_q4_0_block(original)
    assert len(nibble_bytes) == 16
    assert scale > 0

    dequant = SelectivePrecisionController.dequantize_q4_0_block(nibble_bytes, scale)
    assert dequant.shape == (32,)

    # Q4_0 distortion check
    distortion = SelectivePrecisionController.compute_normalized_l2_distortion(original, dequant)
    assert distortion < 0.2


def test_edit_gate_forbidden_scope():
    # Should block embeddings and lm_head
    with pytest.raises(EditGateViolation, match="forbidden scope"):
        CausalFalsificationVerifier.verify_target_allowed("model.embed_tokens.weight")

    with pytest.raises(EditGateViolation, match="forbidden scope"):
        CausalFalsificationVerifier.verify_target_allowed("lm_head.weight")

    with pytest.raises(EditGateViolation, match="forbidden scope"):
        CausalFalsificationVerifier.verify_target_allowed("model.norm.weight")

    # Allowed layer
    CausalFalsificationVerifier.verify_target_allowed("model.layers.15.mlp.down_proj.weight")


def test_edit_gate_uniform_scaling_rejection():
    orig = torch.randn(64, 64)
    # Scaled uniformly by 1.05 (a global knob)
    scaled = orig * 1.05

    with pytest.raises(EditGateViolation, match="uniform whole-tensor scaling"):
        CausalFalsificationVerifier.verify_no_uniform_scaling(orig, scaled)

    # Sparse localized edit is allowed
    sparse_edit = orig.clone()
    sparse_edit[0, :5] += 0.1
    CausalFalsificationVerifier.verify_no_uniform_scaling(orig, sparse_edit)


def test_edit_gate_fraction_and_norm_l2():
    total_params = 1_000_000
    orig = torch.zeros(100, 100)  # 10,000 params

    # Touching 20,000 parameters (> 1% of total) -> should raise
    cand_large = orig.clone()
    cand_large[:] = 1.0
    with pytest.raises(EditGateViolation, match="exceeding the safety cap"):
        CausalFalsificationVerifier.verify_edit_distance_and_fraction(orig, cand_large, total_model_parameters=500_000)

    # Small surgical edit within limits
    cand_small = orig.clone()
    cand_small[:10, :10] = 0.001
    stats = CausalFalsificationVerifier.verify_edit_distance_and_fraction(orig, cand_small, total_model_parameters=total_params)
    assert stats["num_modified"] == 100
    assert stats["fraction_model_modified"] <= 0.01
    assert stats["normalized_l2_distance"] <= 1e-4


def test_matched_control_delta_frobenius_norm():
    delta = torch.randn(64, 64) * 0.01
    matched = CausalFalsificationVerifier.generate_matched_control_delta(delta)

    norm_real = float(torch.linalg.norm(delta).item())
    norm_matched = float(torch.linalg.norm(matched).item())

    assert abs(norm_real - norm_matched) < 1e-5
    assert matched.shape == delta.shape
