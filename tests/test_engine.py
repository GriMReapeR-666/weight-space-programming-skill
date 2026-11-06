"""Unit and integration tests for the WSP v5.0 Runtime Engine."""

import os
import tempfile
import pytest

from engine import (
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
        from engine.patch_engine import verify_model_equivalence
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
    from engine import MCPServer
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


def test_mcp_server_wsp_x_tools(tiny_checkpoint):
    path, _ = tiny_checkpoint
    from engine import MCPServer
    import json

    server = MCPServer()

    # 1. profile_subspaces
    req = json.dumps({
        "jsonrpc": "2.0",
        "id": 10,
        "method": "tools/call",
        "params": {
            "name": "profile_subspaces",
            "arguments": {"checkpoint_path": path, "cond_threshold": 50.0},
        },
    })
    res = json.loads(server.process_message(req))
    assert "result" in res
    prof = json.loads(res["result"]["content"][0]["text"])
    assert "profiles" in prof

    # 2. plan_surgery
    req = json.dumps({
        "jsonrpc": "2.0",
        "id": 11,
        "method": "tools/call",
        "params": {
            "name": "plan_surgery",
            "arguments": {"checkpoint_path": path, "objective": "Stabilize numerical rank"},
        },
    })
    res = json.loads(server.process_message(req))
    plan = json.loads(res["result"]["content"][0]["text"])
    assert "plan_id" in plan
    assert "actions" in plan

    # 3. verify_edit_gate (allowed target vs forbidden target)
    req_ok = json.dumps({
        "jsonrpc": "2.0",
        "id": 12,
        "method": "tools/call",
        "params": {
            "name": "verify_edit_gate",
            "arguments": {"tensor_name": "layers.0.mlp.down_proj.weight"},
        },
    })
    res_ok = json.loads(server.process_message(req_ok))
    gate_ok = json.loads(res_ok["result"]["content"][0]["text"])
    assert gate_ok["passed"] is True

    req_forbidden = json.dumps({
        "jsonrpc": "2.0",
        "id": 13,
        "method": "tools/call",
        "params": {
            "name": "verify_edit_gate",
            "arguments": {"tensor_name": "lm_head.weight"},
        },
    })
    res_forbidden = json.loads(server.process_message(req_forbidden))
    assert res_forbidden["result"]["isError"] is True
    assert "forbidden scope pattern" in res_forbidden["result"]["content"][0]["text"]

    # 4. dequantize_block (Q8_0)
    # 32 bytes of zeros
    hex_zeros = "00" * 32
    req_dq = json.dumps({
        "jsonrpc": "2.0",
        "id": 14,
        "method": "tools/call",
        "params": {
            "name": "dequantize_block",
            "arguments": {"hex_data": hex_zeros, "format": "q8_0", "scale": 0.5},
        },
    })
    res_dq = json.loads(server.process_message(req_dq))
    dq = json.loads(res_dq["result"]["content"][0]["text"])
    assert len(dq["values"]) == 32
    assert dq["scale"] == 0.5

    # 5. evaluate_critique
    req_crit = json.dumps({
        "jsonrpc": "2.0",
        "id": 15,
        "method": "tools/call",
        "params": {
            "name": "evaluate_critique",
            "arguments": {
                "trace_id": "test_trace",
                "problem": "math problem",
                "trace": "Generation hits token limit copying loop",
                "chosen_option": "A",
            },
        },
    })
    res_crit = json.loads(server.process_message(req_crit))
    crit = json.loads(res_crit["result"]["content"][0]["text"])
    assert "critique_pair" in crit
    assert "is_correct" in crit

    # 6. prune_traces
    req_tpt = json.dumps({
        "jsonrpc": "2.0",
        "id": 16,
        "method": "tools/call",
        "params": {
            "name": "prune_traces",
            "arguments": {
                "rollouts": [
                    {"rollout_id": "r1", "problem": "2+2", "reasoning_trace": "...", "candidate_answer": "4", "expected_answer": "4"},
                    {"rollout_id": "r2", "problem": "2+2", "reasoning_trace": "...", "candidate_answer": "5", "expected_answer": "4"},
                ]
            },
        },
    })
    res_tpt = json.loads(server.process_message(req_tpt))
    tpt = json.loads(res_tpt["result"]["content"][0]["text"])
    assert tpt["total_rollouts"] == 2
    assert tpt["pruned_correct_count"] == 1
    assert tpt["success_rate"] == 0.5

    # 7. run_rsi_cycle
    req_rsi = json.dumps({
        "jsonrpc": "2.0",
        "id": 17,
        "method": "tools/call",
        "params": {
            "name": "run_rsi_cycle",
            "arguments": {
                "rollouts": [
                    {"rollout_id": "r1", "problem": "Find root", "reasoning_trace": "Hits copying loop", "candidate_answer": "wrong", "expected_answer": "correct"},
                    {"rollout_id": "r2", "problem": "Find root", "reasoning_trace": "Correct derivation", "candidate_answer": "correct", "expected_answer": "correct"},
                ]
            },
        },
    })
    res_rsi = json.loads(server.process_message(req_rsi))
    rsi = json.loads(res_rsi["result"]["content"][0]["text"])
    assert rsi["total_sampled"] == 2
    assert rsi["retained_correct_count"] == 1
    assert len(rsi["projected_traces"]) == 1

    # 8. export_error_catalog
    req_cat = json.dumps({
        "jsonrpc": "2.0",
        "id": 18,
        "method": "tools/call",
        "params": {
            "name": "export_error_catalog",
            "arguments": {},
        },
    })
    res_cat = json.loads(server.process_message(req_cat))
    cat = json.loads(res_cat["result"]["content"][0]["text"])
    assert cat["total_patterns"] >= 4

    # 9. simulate_sot_debate
    req_sot = json.dumps({
        "jsonrpc": "2.0",
        "id": 19,
        "method": "tools/call",
        "params": {
            "name": "simulate_sot_debate",
            "arguments": {
                "problem": "Factorize 91",
                "candidate_trace": "91 is prime because it ends in 1",
            },
        },
    })
    res_sot = json.loads(server.process_message(req_sot))
    sot = json.loads(res_sot["result"]["content"][0]["text"])
    assert "analyst_perspective" in sot
    assert "skeptic_perspective" in sot
    assert "synthesized_resolution" in sot

    # 10. plan_distillation
    req_dist = json.dumps({
        "jsonrpc": "2.0",
        "id": 20,
        "method": "tools/call",
        "params": {
            "name": "plan_distillation",
            "arguments": {
                "student_checkpoint": path,
                "cond_limit": 100.0,
            },
        },
    })
    res_dist = json.loads(server.process_message(req_dist))
    dist = json.loads(res_dist["result"]["content"][0]["text"])
    assert "target_layers_to_train" in dist
    assert "layer_plans" in dist


def test_distillation_and_sot_engines(tiny_checkpoint):
    path, _ = tiny_checkpoint
    from engine import (
        DistillationController,
        SocietyOfThoughtEngine,
    )
    import torch

    # 1. Society of Thought simulation
    debate = SocietyOfThoughtEngine.simulate_debate(
        problem="Solve quadratic equation",
        candidate_trace="Attention copying loop hits maximum token limit",
    )
    assert "Analyst Perspective" in debate.analyst_perspective
    assert "Skeptic Perspective" in debate.skeptic_perspective
    assert "Verifier Perspective" in debate.verifier_perspective
    assert debate.confidence_score > 0.0

    # 2. Distillation strategy planning
    rec = DistillationController.plan_distillation_strategy(path, cond_limit=100.0)
    assert rec.total_layers_analyzed > 0
    assert len(rec.protected_layers_to_freeze) > 0  # Embeddings/heads frozen
    assert len(rec.target_layers_to_train) > 0

    # 3. KD Loss computation
    s_logits = torch.randn(4, 10)
    t_logits = torch.randn(4, 10)
    labels = torch.randint(0, 10, (4,))
    loss = DistillationController.compute_kd_loss(s_logits, t_logits, labels, temperature=2.0, alpha=0.5)
    assert float(loss.item()) > 0.0

    # 4. Protected Distillation update
    orig_w = torch.zeros(8, 8)
    delta = torch.ones(8, 8) * 0.1
    # Null basis: rank-2 basis
    basis, _ = torch.linalg.qr(torch.randn(8, 2))
    updated_w = DistillationController.apply_protected_distillation_update(orig_w, delta, basis)
    assert updated_w.shape == orig_w.shape


def test_recursive_self_improvement_engine():
    from engine import (
        ReasoningRollout,
        RecursiveSelfImprovementEngine,
    )
    import torch
    import torch.nn as nn

    engine = RecursiveSelfImprovementEngine()

    # 1. Trace-space cycle
    rollouts = [
        ReasoningRollout("r1", "3*3", "3*3=9", "9"),
        ReasoningRollout("r2", "3*3", "3*3=6", "6"),
    ]
    res_trace = engine.run_cycle_on_traces(rollouts, lambda p, a: a == "9")
    assert res_trace["retained_correct_count"] == 1
    assert res_trace["rejected_count"] == 1
    assert len(res_trace["projected_traces"]) == 1

    # 2. Weight-space cycle: Sustainable improvement
    base = nn.Linear(8, 2, bias=False)
    cand = nn.Linear(8, 2, bias=False)
    cand.weight.data.copy_(base.weight.data)
    # Add a slight beneficial bias along sample 0
    with torch.no_grad():
        cand.weight.data[0, 0] += 0.05

    target_samples = [torch.ones(1, 8)]
    anchor_samples = [torch.zeros(1, 8)]

    def score_fn(m, x):
        return float(m(x)[0, 0].item())

    cycle_res = engine.execute_weight_space_cycle(
        base_model=base,
        candidate_model=cand,
        target_tensor_name="layers.0.mlp.down_proj.weight",
        target_eval_samples=target_samples,
        anchor_eval_samples=anchor_samples,
        scoring_fn=score_fn,
    )
    assert cycle_res.accepted is True
    assert cycle_res.status == "accepted"

    # 3. Weight-space cycle: Reverted degenerative improvement
    bad_cand = nn.Linear(8, 2, bias=False)
    bad_cand.weight.data.copy_(base.weight.data)
    with torch.no_grad():
        bad_cand.weight.data[0, 0] -= 0.1  # Degrades target performance

    bad_cycle = engine.execute_weight_space_cycle(
        base_model=base,
        candidate_model=bad_cand,
        target_tensor_name="layers.0.mlp.down_proj.weight",
        target_eval_samples=target_samples,
        anchor_eval_samples=anchor_samples,
        scoring_fn=score_fn,
    )
    assert bad_cycle.accepted is False
    assert bad_cycle.rollback_performed is True
    assert bad_cycle.status == "reverted_degenerative"
