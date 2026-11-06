"""Unit and integration tests for MechanisticDiagnostician and Remediation Engine."""

import json
import os
import tempfile
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from engine import (  # noqa: E402
    DefectCategory,
    DiagnosticReport,
    MCPServer,
    MechanisticDiagnostician,
)


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------
def _create_synthetic_model(ill_conditioned: bool = False) -> str:
    """Creates a temporary safetensors/pt checkpoint with known spectral properties."""
    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)

    # Large embedding table to establish sufficient total parameter budget
    embed = torch.randn(1000, 64)

    if ill_conditioned:
        # Construct rank-1 dominant matrix with condition number > 1000
        u = torch.randn(32, 1)
        v = torch.randn(1, 32)
        down_proj = u @ v + 1e-5 * torch.randn(32, 32)
        gate_proj, _ = torch.linalg.qr(torch.randn(32, 32))
    else:
        # Orthogonal matrices with condition number ~ 1.0
        down_proj, _ = torch.linalg.qr(torch.randn(32, 32))
        gate_proj, _ = torch.linalg.qr(torch.randn(32, 32))

    tensors = {
        "model.embed_tokens.weight": embed,
        "model.layers.0.mlp.down_proj.weight": down_proj,
        "model.layers.0.mlp.gate_proj.weight": gate_proj,
        "model.norm.weight": torch.ones(32),
    }
    torch.save(tensors, path)
    return path


# ---------------------------------------------------------------------------
# Triage Tests
# ---------------------------------------------------------------------------
def test_triage_anomalies_repetition():
    diagnostician = MechanisticDiagnostician()
    matches = diagnostician.triage_anomalies("generation suffering from repetition loop and copying runaway")
    assert len(matches) > 0
    top = matches[0]
    assert top.defect_id == "DEFECT-REP-01"
    assert top.category == DefectCategory.INDUCTION_LOOP_COLLAPSE


def test_triage_anomalies_spectral():
    diagnostician = MechanisticDiagnostician()
    matches = diagnostician.triage_anomalies("severe numerical instability and erratic confidence")
    assert len(matches) > 0
    top = matches[0]
    assert top.defect_id == "DEFECT-SPEC-02"
    assert top.category == DefectCategory.SPECTRAL_ILL_CONDITIONING


def test_triage_anomalies_quantization():
    diagnostician = MechanisticDiagnostician()
    matches = diagnostician.triage_anomalies("severe quantization collapse under int4 format")
    assert len(matches) > 0
    top = matches[0]
    assert top.defect_id == "DEFECT-OUT-03"
    assert top.category == DefectCategory.OUTLIER_CHANNEL_KURTOSIS


def test_triage_anomalies_logit_bias():
    diagnostician = MechanisticDiagnostician()
    matches = diagnostician.triage_anomalies("excessive hesitation and spurious deferral to secondary systems")
    assert len(matches) > 0
    top = matches[0]
    assert top.defect_id == "DEFECT-BIAS-04"
    assert top.category == DefectCategory.LOGIT_CALIBRATION_DRIFT


def test_triage_anomalies_subspace():
    diagnostician = MechanisticDiagnostician()
    matches = diagnostician.triage_anomalies("regression on held-out anchor tasks and catastrophic interference")
    assert len(matches) > 0
    top = matches[0]
    assert top.defect_id == "DEFECT-INTERF-05"
    assert top.category == DefectCategory.SUBSPACE_INTERFERENCE


def test_triage_anomalies_no_match():
    diagnostician = MechanisticDiagnostician()
    matches = diagnostician.triage_anomalies("unrelated query about weather or gardening")
    assert matches == []


# ---------------------------------------------------------------------------
# Inspection & Diagnostics Tests
# ---------------------------------------------------------------------------
def test_inspect_healthy_checkpoint():
    path = _create_synthetic_model(ill_conditioned=False)
    try:
        diagnostician = MechanisticDiagnostician()
        report = diagnostician.inspect_checkpoint(path, cond_limit=200.0)

        assert isinstance(report, DiagnosticReport)
        assert report.structural_integrity_score >= 90.0
        assert report.status_summary == "Structurally Optimal"
        assert len(report.remediation_plan) == 0
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_inspect_ill_conditioned_checkpoint():
    path = _create_synthetic_model(ill_conditioned=True)
    try:
        diagnostician = MechanisticDiagnostician()
        report = diagnostician.inspect_checkpoint(
            path,
            observed_behaviors=["numerical instability", "repetition loop"],
            cond_limit=150.0,
        )

        assert isinstance(report, DiagnosticReport)
        assert report.structural_integrity_score < 90.0
        defect_ids = [f.defect.defect_id for f in report.findings]
        assert "DEFECT-SPEC-02" in defect_ids or "DEFECT-REP-01" in defect_ids

        # Remediation plan formulation
        assert len(report.remediation_plan) > 0
        actions = [a.operation_type for a in report.remediation_plan]
        assert "spectral_clamp" in actions or "null_space_projection" in actions

        # Summary formatting
        md_summary = diagnostician.generate_diagnostic_summary(report)
        assert "Mechanistic Diagnostic Report" in md_summary
        assert "DEFECT-" in md_summary
        assert "Safety & Gating Constraints" in md_summary
    finally:
        if os.path.exists(path):
            os.remove(path)


# ---------------------------------------------------------------------------
# Remediation Execution Tests
# ---------------------------------------------------------------------------
def test_execute_remediation():
    in_path = _create_synthetic_model(ill_conditioned=True)
    out_fd, out_path = tempfile.mkstemp(suffix=".pt")
    os.close(out_fd)

    try:
        diagnostician = MechanisticDiagnostician()
        report = diagnostician.inspect_checkpoint(in_path, cond_limit=150.0)

        res = diagnostician.execute_remediation(report, output_checkpoint_path=out_path)
        assert res["status"] in ("remediation_complete", "no_remediation_needed")

        if res["status"] == "remediation_complete":
            assert os.path.exists(out_path)
            # Ensure remediated file is loadable
            loaded = torch.load(out_path, weights_only=True)
            assert "model.layers.0.mlp.down_proj.weight" in loaded
            assert "model.embed_tokens.weight" in loaded
    finally:
        if os.path.exists(in_path):
            os.remove(in_path)
        if os.path.exists(out_path):
            os.remove(out_path)


# ---------------------------------------------------------------------------
# MCP Tool Handlers Tests
# ---------------------------------------------------------------------------
def test_mcp_triage_defect_anomalies():
    server = MCPServer()
    req = {
        "jsonrpc": "2.0",
        "id": 101,
        "method": "tools/call",
        "params": {
            "name": "triage_defect_anomalies",
            "arguments": {
                "query": "repetition loop endlessly generating tokens",
            },
        },
    }
    raw = server.process_message(json.dumps(req))
    assert raw is not None
    resp = json.loads(raw)
    assert resp["id"] == 101
    assert "result" in resp
    content = resp["result"]["content"][0]["text"]
    assert "DEFECT-REP-01" in content
    assert "Residual Stream Induction Loop Collapse" in content


def test_mcp_diagnose_defects():
    server = MCPServer()
    path = _create_synthetic_model(ill_conditioned=False)
    try:
        req = {
            "jsonrpc": "2.0",
            "id": 102,
            "method": "tools/call",
            "params": {
                "name": "diagnose_defects",
                "arguments": {
                    "checkpoint_path": path,
                    "observed_behaviors": ["routine baseline scan"],
                    "cond_limit": 200.0,
                },
            },
        }
        raw = server.process_message(json.dumps(req))
        assert raw is not None
        resp = json.loads(raw)
        assert resp["id"] == 102
        assert "result" in resp
        content = resp["result"]["content"][0]["text"]
        assert "structural_integrity_score" in content
        assert "Structurally Optimal" in content
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_mcp_execute_remediation():
    server = MCPServer()
    in_path = _create_synthetic_model(ill_conditioned=True)
    out_fd, out_path = tempfile.mkstemp(suffix=".pt")
    os.close(out_fd)

    try:
        req = {
            "jsonrpc": "2.0",
            "id": 103,
            "method": "tools/call",
            "params": {
                "name": "execute_remediation",
                "arguments": {
                    "checkpoint_path": in_path,
                    "output_checkpoint_path": out_path,
                    "cond_limit": 150.0,
                },
            },
        }
        raw = server.process_message(json.dumps(req))
        assert raw is not None
        resp = json.loads(raw)
        assert resp["id"] == 103
        assert "result" in resp
        content = resp["result"]["content"][0]["text"]
        assert "remediation" in content.lower()
    finally:
        if os.path.exists(in_path):
            os.remove(in_path)
        if os.path.exists(out_path):
            os.remove(out_path)
