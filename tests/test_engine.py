"""Unit and integration tests for the WSP v5.0 Runtime Engine."""

import os
import tempfile
import pytest

from weight_space_programming_skill.engine import (
    AblationMode,
    PatchOperation,
    analyze_weight_directions,
    apply_patch,
    build_circuit_map,
    capture_activations,
    compare_models,
    generate_patch,
    identify_architecture,
    inspect_model,
    inspect_tensor,
    optimize_patch,
    reconstruct_graph,
    rollback_patch,
    run_activation_ablation,
    run_parameter_ablation,
    summarize_weights,
    validate_patch,
)

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = pytest.importorskip("torch")  # type: ignore
    nn = torch.nn  # type: ignore


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------
class TinyTransformerBlock(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(dim, dim)
        self.self_attn.k_proj = nn.Linear(dim, dim)
        self.self_attn.v_proj = nn.Linear(dim, dim)
        self.self_attn.o_proj = nn.Linear(dim, dim)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(dim, dim * 2)
        self.mlp.down_proj = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        q = self.self_attn.q_proj(x)
        k = self.self_attn.k_proj(x)
        v = self.self_attn.v_proj(x)
        attn_out = self.self_attn.o_proj(q + k + v)
        x = self.norm(residual + attn_out)
        ffn_out = self.mlp.down_proj(torch.relu(self.mlp.gate_proj(x)))
        return x + ffn_out


class TinyTransformerModel(nn.Module):
    def __init__(self, dim=16, n_layers=2):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, dim)
        self.layers = nn.ModuleList([TinyTransformerBlock(dim) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, 32, bias=False)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return self.lm_head(x)


@pytest.fixture
def tiny_checkpoint():
    model = TinyTransformerModel(dim=16, n_layers=2)
    sd = model.state_dict()
    f = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    f.close()
    torch.save(sd, f.name)
    yield f.name, model
    try:
        os.unlink(f.name)
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Milestone 1: Model Inspection & Tensor Engine
# ---------------------------------------------------------------------------
def test_inspect_model(tiny_checkpoint):
    path, _ = tiny_checkpoint
    meta = inspect_model(path)
    assert meta.n_tensors > 0
    assert meta.total_parameters > 0
    assert meta.layer_count == 2
    assert meta.format.value == "pytorch"


def test_inspect_tensor(tiny_checkpoint):
    path, _ = tiny_checkpoint
    summary = inspect_tensor(path, "layers.0.self_attn.q_proj.weight")
    assert summary.shape == [16, 16]
    assert summary.finite_fraction == 1.0
    assert summary.numel == 256
    assert summary.layer_index == 0


def test_summarize_weights(tiny_checkpoint):
    path, _ = tiny_checkpoint
    summaries = summarize_weights(path)
    assert len(summaries) >= 10
    names = [s.name for s in summaries]
    assert "embed_tokens.weight" in names


def test_analyze_weight_directions():
    matrix = torch.randn(32, 16)
    analysis = analyze_weight_directions(matrix, top_k=4)
    assert len(analysis["singular_values"]) == 4
    assert 0.0 < analysis["energy_ratio_top_k"] <= 1.0
    assert analysis["stable_rank"] > 0
    assert analysis["condition_number"] >= 1.0


# ---------------------------------------------------------------------------
# Milestone 2: Architecture & Graph Reconstruction
# ---------------------------------------------------------------------------
def test_identify_architecture(tiny_checkpoint):
    path, _ = tiny_checkpoint
    arch = identify_architecture(path)
    assert arch["layer_count"] == 2
    assert arch["hidden_dimension"] == 16
    assert arch["family"] == "dense_transformer"


def test_reconstruct_graph(tiny_checkpoint):
    path, _ = tiny_checkpoint
    graph = reconstruct_graph(path)
    assert "embeddings" in graph.nodes
    assert "layer_0" in graph.nodes
    assert "layer_1" in graph.nodes
    assert "output_head" in graph.nodes
    assert graph.total_layers == 2
    assert graph.execution_order[0] == "embeddings"


# ---------------------------------------------------------------------------
# Milestone 3: Activation Tracing & Circuit Mapping
# ---------------------------------------------------------------------------
def test_capture_activations(tiny_checkpoint):
    _, model = tiny_checkpoint
    inputs = torch.randint(0, 32, (1, 8))
    target_layers = ["layers.0.self_attn.q_proj", "layers.1.mlp.down_proj"]
    traces = capture_activations(model, inputs, target_layers)
    assert len(traces) == 2
    for tr in traces:
        assert tr.max_abs >= 0.0
        assert tr.l2_norm > 0.0


def test_build_circuit_map(tiny_checkpoint):
    _, model = tiny_checkpoint
    inputs = torch.randint(0, 32, (1, 8))

    def eval_fn(m):
        out = m(inputs)
        return float(out.detach().mean())

    targets = ["layers.0.self_attn.q_proj", "layers.1.mlp.down_proj"]
    circuit = build_circuit_map(model, eval_fn, targets, behavioral_target="mean_logit")
    assert len(circuit.critical_nodes) == 2
    assert circuit.target_behavior == "mean_logit"


# ---------------------------------------------------------------------------
# Milestone 4: Ablation Sandbox
# ---------------------------------------------------------------------------
def test_run_activation_ablation(tiny_checkpoint):
    _, model = tiny_checkpoint
    inputs = torch.randint(0, 32, (1, 8))

    def eval_fn(m):
        return float(m(inputs).detach().abs().mean())

    res = run_activation_ablation(
        model, eval_fn, "layers.0.self_attn.o_proj", mode=AblationMode.ZERO
    )
    assert res.baseline_score > 0
    assert res.target_component == "layers.0.self_attn.o_proj"


def test_run_parameter_ablation(tiny_checkpoint):
    _, model = tiny_checkpoint
    inputs = torch.randint(0, 32, (1, 8))

    def eval_fn(m):
        return float(m(inputs).mean())

    res = run_parameter_ablation(
        model, eval_fn, "layers.0.self_attn.q_proj.weight", mode=AblationMode.ZERO
    )
    assert res.target_component == "layers.0.self_attn.q_proj.weight"


# ---------------------------------------------------------------------------
# Milestone 5: Optimization & Search Engine
# ---------------------------------------------------------------------------
def test_evaluate_candidate_and_optimize(tiny_checkpoint):
    _, model = tiny_checkpoint
    inputs = torch.randint(0, 32, (1, 8))

    def eval_fn(m):
        # Objective: push logits mean closer to 10.0
        val = float(m(inputs).mean())
        return -abs(10.0 - val)

    patch_op, score, metrics = optimize_patch(
        model,
        eval_fn,
        "layers.0.self_attn.q_proj.weight",
        max_iterations=5,
        step_size=0.05,
    )
    assert patch_op.tensor_name == "layers.0.self_attn.q_proj.weight"
    assert len(metrics["score_history"]) >= 5


# ---------------------------------------------------------------------------
# Milestone 6: Patch Engine (Generate, Validate, Apply, Rollback)
# ---------------------------------------------------------------------------
def test_patch_engine_lifecycle(tiny_checkpoint):
    path, _ = tiny_checkpoint
    delta = torch.ones(16, 16) * 0.1
    op = PatchOperation(
        tensor_name="layers.0.self_attn.q_proj.weight",
        operation="add",
        shape=[16, 16],
        data=delta,
        alpha=1.0,
    )
    manifest = generate_patch(path, [op], description="Test patch")
    assert validate_patch(manifest, path) is True

    # Apply patch to new file
    patched_file = tempfile.NamedTemporaryFile(suffix=".pt", delete=False).name
    restored_file = tempfile.NamedTemporaryFile(suffix=".pt", delete=False).name
    try:
        apply_patch(manifest, path, patched_file)
        assert os.path.exists(patched_file)

        # Rollback patch to verify reversibility
        rollback_patch(patched_file, manifest, restored_file)
        assert os.path.exists(restored_file)

        # Restored model weights must match original exactly
        from weight_space_programming_skill.engine.patch_engine import verify_model_equivalence
        assert verify_model_equivalence(path, restored_file) is True
    finally:
        for p in (patched_file, restored_file):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------------------
# Milestone 7: Evaluation & Regression Gates
# ---------------------------------------------------------------------------
def test_compare_models():
    base = nn.Linear(8, 2)
    cand = nn.Linear(8, 2)
    with torch.no_grad():
        cand.weight.copy_(base.weight + 0.01)
        cand.bias.copy_(base.bias)

    target_samples = [torch.randn(1, 8) for _ in range(5)]
    anchor_samples = [torch.randn(1, 8) for _ in range(5)]

    def score_fn(m, x):
        return float(m(x).mean())

    result = compare_models(
        base, cand, target_samples, anchor_samples, score_fn, max_regression_tolerance=0.50
    )
    assert result.passed_regression_gate is True
    assert "anchor_delta" in result.regression_summary


# ---------------------------------------------------------------------------
# Milestone 9: MCP Server & Agent Protocol
# ---------------------------------------------------------------------------
def test_mcp_server_protocol(tiny_checkpoint):
    path, _ = tiny_checkpoint
    from weight_space_programming_skill.engine import MCPServer
    import json

    server = MCPServer()

    # 1. Initialize
    init_req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    init_res = json.loads(server.process_message(init_req))
    assert init_res["result"]["serverInfo"]["name"] == "weight-space-programming-engine"
    assert "tools" in init_res["result"]["capabilities"]

    # 2. Tools list
    list_req = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    list_res = json.loads(server.process_message(list_req))
    tool_names = [t["name"] for t in list_res["result"]["tools"]]
    assert "inspect_model" in tool_names
    assert "identify_architecture" in tool_names
    assert "reconstruct_graph" in tool_names

    # 3. Call tool
    call_req = json.dumps({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "inspect_model",
            "arguments": {"checkpoint_path": path},
        },
    })
    call_res = json.loads(server.process_message(call_req))
    assert "content" in call_res["result"]
    payload = json.loads(call_res["result"]["content"][0]["text"])
    assert payload["total_parameters"] > 0
    assert payload["format"] in ("pytorch", "pt")
