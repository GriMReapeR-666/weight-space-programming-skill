#!/usr/bin/env python3
"""WSP CLI — unified command-line interface for all weight-space programming tools.

Usage:
    python -m weight_space_programming_skill inspect model.pt
    python -m weight_space_programming_skill inspect model.safetensors --stats
    python -m weight_space_programming_skill inspect model.gguf

    python -m weight_space_programming_skill patch model.pt --patch patch.json --output patched.pt
    python -m weight_space_programming_skill patch model.safetensors --patch patch.json --output patched.safetensors

    python -m weight_space_programming_skill merge linear base.pt delta.pt --alpha 0.5 --output merged.pt
    python -m weight_space_programming_skill merge slerp base.safetensors delta.safetensors --output merged.safetensors
    python -m weight_space_programming_skill merge ties base.pt ft1.pt ft2.pt --density 0.5 --output merged.pt
    python -m weight_space_programming_skill merge dare base.pt ft.pt --density 0.5 --output merged.pt

    python -m weight_space_programming_skill lora inspect model.pt
    python -m weight_space_programming_skill lora scaffold model.pt --layer q_proj --rank 8 --output lora.pt
    python -m weight_space_programming_skill lora merge model.pt --lora-file lora.pt --output merged.pt

    python -m weight_space_programming_skill scaffold model.pt --output adapter.py --map arch.json
    python -m weight_space_programming_skill log --ledger exp.jsonl --event patch --model model.pt --notes "..."

    python -m weight_space_programming_skill profile model.safetensors [--cond-threshold 100.0]
    python -m weight_space_programming_skill plan model.safetensors [--objective "obj"]
    python -m weight_space_programming_skill gate --tensor "model.layers.0.mlp.down_proj.weight"
    python -m weight_space_programming_skill rsi --catalog
    python -m weight_space_programming_skill distill student.safetensors [--cond-limit 250.0]
    python -m weight_space_programming_skill sot --problem "calculate derivative"
    python -m weight_space_programming_skill mcp
"""
import sys
import os

# Add the tools directory to path so we can import the tool modules
_TOOLS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "wsp_protocol", "tools")
sys.path.insert(0, os.path.abspath(_TOOLS_DIR))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]
    # Rewrite argv so sub-tools parse cleanly
    sys.argv = sys.argv[1:]

    if cmd == "inspect":
        _route_inspect()
    elif cmd == "patch":
        _route_patch()
    elif cmd == "merge":
        _route_merge()
    elif cmd == "lora":
        _route_lora()
    elif cmd == "scaffold":
        _route_scaffold()
    elif cmd == "log":
        _route_log()
    elif cmd == "profile":
        _route_profile()
    elif cmd == "plan":
        _route_plan()
    elif cmd == "gate":
        _route_gate()
    elif cmd == "rsi":
        _route_rsi()
    elif cmd == "distill":
        _route_distill()
    elif cmd == "sot":
        _route_sot()
    elif cmd == "mcp":
        _route_mcp()
    elif cmd in ("-h", "--help", "help"):
        print(__doc__)
    else:
        print(f"Unknown command: {cmd}\n\nAvailable: inspect patch merge lora scaffold log profile plan gate rsi distill sot mcp")
        sys.exit(1)


def _route_inspect():
    """Route inspect command to correct adapter based on file extension."""
    import argparse
    p = argparse.ArgumentParser(description="Inspect a model checkpoint")
    p.add_argument("checkpoint")
    p.add_argument("--stats", action="store_true", help="Compute per-tensor statistics")
    args = p.parse_args()

    ext = args.checkpoint.lower().rsplit(".", 1)[-1]
    if ext in ("pt", "pth"):
        import tensor_summary
        sys.argv = ["tensor_summary", args.checkpoint]
        tensor_summary.main()
    elif ext == "safetensors":
        import safetensors_adapter
        safetensors_adapter.cmd_inspect(args)
    elif ext == "gguf":
        import gguf_reader
        sys.argv = ["gguf_reader", "inspect", args.checkpoint]
        gguf_reader.main()
    else:
        print(f"Unsupported format: .{ext}", file=sys.stderr)
        sys.exit(1)


def _route_patch():
    import argparse
    p = argparse.ArgumentParser(description="Apply a patch to a checkpoint")
    p.add_argument("checkpoint")
    p.add_argument("--patch", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    ext = args.checkpoint.lower().rsplit(".", 1)[-1]
    if ext in ("pt", "pth"):
        import patch_delta
        sys.argv = ["patch_delta", "--checkpoint", args.checkpoint, "--patch", args.patch]
        patch_delta.main()
    elif ext == "safetensors":
        import json
        from pathlib import Path
        import safetensors_adapter
        spec = json.loads(Path(args.patch).read_text())
        result = safetensors_adapter.apply_patch(args.checkpoint, spec, args.output)
        print(json.dumps(result, indent=2))
    else:
        print(f"Unsupported format: .{ext}", file=sys.stderr)
        sys.exit(1)


def _route_merge():
    import merge_models
    merge_models.main()


def _route_lora():
    import lora_inject
    lora_inject.main()


def _route_scaffold():
    import scaffold_adapter
    scaffold_adapter.main()


def _route_log():
    import experiment_ledger
    experiment_ledger.main()


def _route_mcp():
    from weight_space_programming_skill.engine import MCPServer
    server = MCPServer()
    server.run_stdio()


def _route_profile():
    import argparse
    import json
    p = argparse.ArgumentParser(description="Profile weight-space manifolds and condition numbers")
    p.add_argument("checkpoint", help="Path to checkpoint file")
    p.add_argument("--cond-threshold", type=float, default=100.0, help="Condition number limit")
    args = p.parse_args()

    from weight_space_programming_skill.engine import profile_model_subspaces
    from weight_space_programming_skill.engine.mcp_server import _to_serializable
    mapping = profile_model_subspaces(args.checkpoint, cond_threshold=args.cond_threshold)
    print(json.dumps(_to_serializable(mapping), indent=2))


def _route_plan():
    import argparse
    import json
    p = argparse.ArgumentParser(description="Plan Pareto-constrained surgical intervention")
    p.add_argument("checkpoint", help="Path to checkpoint file")
    p.add_argument("--objective", default="Surgical parameter enhancement", help="Target goal")
    p.add_argument("--epsilon", type=float, default=0.01, help="Preservation epsilon")
    p.add_argument("--cond-limit", type=float, default=250.0, help="Condition limit")
    args = p.parse_args()

    from weight_space_programming_skill.engine import ParetoSurgeryPlanner
    from weight_space_programming_skill.engine.mcp_server import _to_serializable
    plan = ParetoSurgeryPlanner.formulate_plan(
        checkpoint_path=args.checkpoint,
        objective=args.objective,
        preservation_epsilon=args.epsilon,
        cond_limit=args.cond_limit,
    )
    print(json.dumps(_to_serializable(plan), indent=2))


def _route_gate():
    import argparse
    p = argparse.ArgumentParser(description="Verify target tensor against Edit Gate policies")
    p.add_argument("--tensor", required=True, help="Target tensor name")
    args = p.parse_args()

    from weight_space_programming_skill.engine import CausalFalsificationVerifier, EditGateViolation
    try:
        CausalFalsificationVerifier.verify_target_allowed(args.tensor)
        print(f"PASS: Target '{args.tensor}' is allowed for editing.")
    except EditGateViolation as exc:
        print(f"REJECTED: {exc}", file=sys.stderr)
        sys.exit(1)


def _route_rsi():
    import argparse
    import json
    p = argparse.ArgumentParser(description="Recursive Self-Improvement (RSI) commands")
    p.add_argument("--catalog", action="store_true", help="Print persistent error catalog")
    p.add_argument("--rollouts", help="Path to rollouts JSON file to run an RSI cycle")
    args = p.parse_args()

    from weight_space_programming_skill.engine import (
        ErrorCatalogManager,
        ReasoningRollout,
        RecursiveSelfImprovementEngine,
    )

    if args.catalog or not args.rollouts:
        catalog = ErrorCatalogManager()
        patterns = catalog.export_catalog()
        print(json.dumps({"total_patterns": len(patterns), "patterns": patterns}, indent=2))
        return

    with open(args.rollouts, "r", encoding="utf-8") as f:
        data = json.load(f)

    rollouts = [
        ReasoningRollout(
            rollout_id=r.get("rollout_id", f"r_{i}"),
            problem=r.get("problem", ""),
            reasoning_trace=r.get("reasoning_trace", ""),
            candidate_answer=str(r.get("candidate_answer", "")),
        )
        for i, r in enumerate(data)
    ]
    expected_map = {r.get("problem", ""): str(r.get("expected_answer", "")) for r in data if "expected_answer" in r}

    def verifier(prob: str, ans: str) -> bool:
        expected = expected_map.get(prob)
        return ans.strip().lower() == expected.strip().lower() if expected else True

    engine = RecursiveSelfImprovementEngine()
    result = engine.run_cycle_on_traces(rollouts, verifier)
    print(json.dumps(result, indent=2))


def _route_distill():
    import argparse
    import json
    p = argparse.ArgumentParser(description="Weight-space guided distillation planning")
    p.add_argument("student", help="Path to student checkpoint")
    p.add_argument("--cond-limit", type=float, default=250.0, help="Condition number threshold")
    p.add_argument("--rank", type=int, default=8, help="Default LoRA rank")
    args = p.parse_args()

    from weight_space_programming_skill.engine import DistillationController
    from weight_space_programming_skill.engine.mcp_server import _to_serializable
    rec = DistillationController.plan_distillation_strategy(
        student_checkpoint=args.student,
        cond_limit=args.cond_limit,
        lora_rank_default=args.rank,
    )
    print(json.dumps(_to_serializable(rec), indent=2))


def _route_sot():
    import argparse
    import json
    p = argparse.ArgumentParser(description="Society of Thought (SOT) internal debate")
    p.add_argument("--problem", required=True, help="Problem prompt")
    p.add_argument("--trace", default="", help="Candidate trace to critique")
    args = p.parse_args()

    from weight_space_programming_skill.engine import SocietyOfThoughtEngine
    from weight_space_programming_skill.engine.mcp_server import _to_serializable
    debate = SocietyOfThoughtEngine.simulate_debate(problem=args.problem, candidate_trace=args.trace)
    print(json.dumps(_to_serializable(debate), indent=2))


if __name__ == "__main__":
    main()
