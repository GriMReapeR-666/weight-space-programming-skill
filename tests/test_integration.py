"""Integration test: end-to-end pipeline on a tiny randomly-initialised model."""
import os, sys, tempfile, json
import pytest

TOOLS = os.path.join(os.path.dirname(__file__), "..", "wsp_protocol", "tools")
sys.path.insert(0, os.path.abspath(TOOLS))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_tiny_model():
    torch = pytest.importorskip("torch")
    nn = torch.nn
    model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))
    return model, model.state_dict()


def _save_pt(sd):
    torch = pytest.importorskip("torch")
    f = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    f.close()
    torch.save(sd, f.name)
    return f.name


# ---------------------------------------------------------------------------
# 1. WeightSpaceProgram core still works
# ---------------------------------------------------------------------------
def test_core_generic_adapter():
    from weight_space_programming_skill import WeightSpaceProgram
    prog = WeightSpaceProgram()
    result = prog.run("hello")
    assert "[generic] hello" in result


# ---------------------------------------------------------------------------
# 2. scaffold_adapter: load tensor inventory + build arch map
# ---------------------------------------------------------------------------
def test_scaffold_adapter():
    torch = pytest.importorskip("torch")
    _, sd = _make_tiny_model()
    pt = _save_pt(sd)
    try:
        from scaffold_adapter import load_tensor_inventory, build_arch_map
        inv = load_tensor_inventory(pt)
        assert len(inv) > 0
        arch = build_arch_map(inv)
        assert arch["n_tensors"] > 0
        assert arch["total_parameters"] > 0  # key is total_parameters
    finally:
        os.unlink(pt)


# ---------------------------------------------------------------------------
# 3. lora_inject: scaffold LoRA matrices + merge back
# ---------------------------------------------------------------------------
def test_lora_inject():
    torch = pytest.importorskip("torch")
    _, sd = _make_tiny_model()
    pt = _save_pt(sd)
    try:
        from lora_inject import inspect_linear_layers, _load_state_dict, create_lora_matrices, merge_lora
        loaded = _load_state_dict(pt)
        layers = inspect_linear_layers(loaded)
        assert len(layers) >= 2  # two Linear layers

        W = sd["0.weight"]
        A, B = create_lora_matrices(W.shape[0], W.shape[1], rank=2)
        merged = merge_lora(dict(sd), "0.weight", A, B, alpha=1.0, rank=2)
        assert merged["0.weight"].shape == W.shape
    finally:
        os.unlink(pt)


# ---------------------------------------------------------------------------
# 4. experiment_ledger: append + read
# ---------------------------------------------------------------------------
def test_experiment_ledger():
    from experiment_ledger import log_event, read_ledger
    f = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w")
    f.close()
    try:
        log_event(f.name, "train", model="tiny", notes="integration")
        log_event(f.name, "eval",  model="tiny", result={"accuracy": 0.99})
        records = read_ledger(f.name)
        assert len(records) == 2
        assert records[0]["event_type"] == "train"
        assert records[1]["result"]["accuracy"] == 0.99
    finally:
        os.unlink(f.name)


# ---------------------------------------------------------------------------
# 5. merge_models: linear merge of two tiny state_dicts
# ---------------------------------------------------------------------------
def test_merge_models_linear():
    torch = pytest.importorskip("torch")
    _, sd1 = _make_tiny_model()
    _, sd2 = _make_tiny_model()
    pt1 = _save_pt(sd1)
    pt2 = _save_pt(sd2)
    out = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    out.close()
    try:
        from merge_models import merge_linear, _load, _save
        a = _load(pt1)
        b = _load(pt2)
        merged, _warns = merge_linear(a, b, alpha=0.5)
        _save(merged, out.name)
        reloaded = _load(out.name)
        for k in sd1:
            assert reloaded[k].shape == sd1[k].shape
    finally:
        for p in (pt1, pt2, out.name):
            try: os.unlink(p)
            except FileNotFoundError: pass


# ---------------------------------------------------------------------------
# 6. tensor_summary: run via subprocess (it's a CLI script with only main())
# ---------------------------------------------------------------------------
def test_tensor_summary_cli():
    import subprocess
    torch = pytest.importorskip("torch")
    _, sd = _make_tiny_model()
    pt = _save_pt(sd)
    try:
        tool = os.path.join(TOOLS, "tensor_summary.py")
        result = subprocess.run(
            [sys.executable, tool, pt],
            capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, result.stderr
        rows = json.loads(result.stdout)
        assert len(rows) == len(sd)
        for row in rows:
            assert "key" in row and "shape" in row and "dtype" in row
    finally:
        os.unlink(pt)
