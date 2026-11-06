"""Model Context Protocol (MCP) server for Weight-Space Programming Engine (v5.0).

Exposes the WSP agent tools over JSON-RPC 2.0 stdio protocol.
Zero external MCP dependencies required; stdlib JSON-RPC handler.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
from typing import Any, Callable, Dict, List, Optional

import torch

from .causal_falsification_verifier import (
    CausalFalsificationVerifier,
)
from .graph_engine import identify_architecture, reconstruct_graph
from .introspective_simulation import (
    CriticDiscernmentEngine,
    ErrorCatalogManager,
)
from .loader import ModelLoader
from .pareto_surgery_planner import ParetoSurgeryPlanner
from .patch_engine import apply_patch, rollback_patch, validate_patch
from .selective_precision_controller import SelectivePrecisionController
from .subspace_feature_analyzer import profile_model_subspaces
from .tensor_engine import (
    analyze_weight_directions,
    inspect_model,
    inspect_tensor,
    summarize_weights,
)
from .tpt_sampler import ReasoningRollout, ThinkPruneTrainEngine
from .types import PatchManifest, PatchOperation

logger = logging.getLogger("wsp.mcp")


def _to_serializable(obj: Any) -> Any:
    """Convert dataclass, enum, or custom object to JSON-serializable structure."""
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)  # type: ignore[arg-type]
    if hasattr(obj, "value"):
        return obj.value
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    return str(obj) if not isinstance(obj, (int, float, bool, type(None), str)) else obj


def _load_manifest_file(patch_path: str) -> PatchManifest:
    """Load PatchManifest from a JSON file path."""
    with open(patch_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ops: List[PatchOperation] = []
    for op in data.get("operations", []):
        raw_data = op.get("data")
        tensor_data = torch.tensor(raw_data) if raw_data is not None else None
        ops.append(
            PatchOperation(
                tensor_name=op["tensor_name"],
                operation=op.get("operation", "add"),
                shape=op.get("shape", []),
                data=tensor_data,
                alpha=float(op.get("alpha", 1.0)),
                rank=op.get("rank"),
            )
        )
    return PatchManifest(
        patch_id=data.get("patch_id", "patch_auto"),
        target_model_path=data.get("target_model_path", ""),
        base_model_hash=data.get("base_model_hash", ""),
        created_at=data.get("created_at", ""),
        author=data.get("author", "agent"),
        description=data.get("description", ""),
        operations=ops,
        is_reversible=data.get("is_reversible", True),
    )


def _handle_analyze_directions(args: Dict[str, Any]) -> Dict[str, Any]:
    checkpoint_path = args["checkpoint_path"]
    tensor_name = args["tensor_name"]
    top_k = args.get("top_k", 5)
    for k, tensor in ModelLoader.stream_tensors(checkpoint_path):
        if k == tensor_name:
            return analyze_weight_directions(tensor, top_k=top_k)
    raise KeyError(f"Tensor '{tensor_name}' not found in {checkpoint_path}")


def _handle_profile_subspaces(args: Dict[str, Any]) -> Dict[str, Any]:
    checkpoint_path = args["checkpoint_path"]
    cond_threshold = float(args.get("cond_threshold", 100.0))
    subspace_map = profile_model_subspaces(checkpoint_path, cond_threshold=cond_threshold)
    return _to_serializable(subspace_map)


def _handle_plan_surgery(args: Dict[str, Any]) -> Dict[str, Any]:
    checkpoint_path = args["checkpoint_path"]
    objective = args.get("objective", "Surgical parameter enhancement")
    preservation_epsilon = float(args.get("preservation_epsilon", 0.01))
    cond_limit = float(args.get("cond_limit", 250.0))
    plan = ParetoSurgeryPlanner.formulate_plan(
        checkpoint_path=checkpoint_path,
        objective=objective,
        preservation_epsilon=preservation_epsilon,
        cond_limit=cond_limit,
    )
    return _to_serializable(plan)


def _handle_verify_edit_gate(args: Dict[str, Any]) -> Dict[str, Any]:
    tensor_name = args["tensor_name"]
    CausalFalsificationVerifier.verify_target_allowed(tensor_name)

    orig_data = args.get("original")
    cand_data = args.get("candidate")
    total_params = int(args.get("total_model_parameters", 1_000_000))

    result: Dict[str, Any] = {"target_allowed": True, "passed": True}
    if orig_data is not None and cand_data is not None:
        orig_t = torch.tensor(orig_data, dtype=torch.float32)
        cand_t = torch.tensor(cand_data, dtype=torch.float32)
        CausalFalsificationVerifier.verify_no_uniform_scaling(orig_t, cand_t)
        metrics = CausalFalsificationVerifier.verify_edit_distance_and_fraction(
            orig_t, cand_t, total_model_parameters=total_params
        )
        result["metrics"] = metrics
    return result


def _handle_dequantize_block(args: Dict[str, Any]) -> Dict[str, Any]:
    block_format = args.get("format", "q8_0").lower()
    hex_str = args["hex_data"]
    scale = float(args.get("scale", 1.0))
    data_bytes = bytes.fromhex(hex_str)
    if block_format == "q8_0":
        tensor = SelectivePrecisionController.dequantize_q8_0_block(data_bytes, scale)
    elif block_format == "q4_0":
        tensor = SelectivePrecisionController.dequantize_q4_0_block(data_bytes, scale)
    else:
        raise ValueError(f"Unsupported block format '{block_format}'. Supported: 'q8_0', 'q4_0'.")
    return {"format": block_format, "scale": scale, "values": tensor.tolist()}


def _handle_evaluate_critique(args: Dict[str, Any]) -> Dict[str, Any]:
    trace_id = args.get("trace_id", "trace_001")
    problem = args.get("problem", "")
    trace = args.get("trace", "")
    catalog = ErrorCatalogManager()
    pair = CriticDiscernmentEngine.generate_critique_pair(trace_id, problem, trace, catalog)
    chosen_option = args.get("chosen_option")
    if chosen_option:
        is_correct, score = CriticDiscernmentEngine.evaluate_discernment(pair, chosen_option)
        return {
            "critique_pair": _to_serializable(pair),
            "chosen_option": chosen_option,
            "is_correct": is_correct,
            "reward_score": score,
        }
    return {"critique_pair": _to_serializable(pair)}


def _handle_prune_traces(args: Dict[str, Any]) -> Dict[str, Any]:
    raw_rollouts = args.get("rollouts", [])
    rollouts = [
        ReasoningRollout(
            rollout_id=r.get("rollout_id", f"r_{i}"),
            problem=r.get("problem", ""),
            reasoning_trace=r.get("reasoning_trace", ""),
            candidate_answer=str(r.get("candidate_answer", "")),
            metadata=r.get("metadata", {}),
        )
        for i, r in enumerate(raw_rollouts)
    ]
    expected_map = {
        r.get("problem", ""): str(r.get("expected_answer", ""))
        for r in raw_rollouts
        if "expected_answer" in r
    }

    def verifier(prob: str, ans: str) -> bool:
        expected = expected_map.get(prob)
        if expected is not None:
            return ans.strip().lower() == expected.strip().lower()
        return True

    res = ThinkPruneTrainEngine.prune_and_filter(rollouts, verifier)
    return _to_serializable(res)


# Declarative tool registry with schemas
TOOLS_REGISTRY: Dict[str, Dict[str, Any]] = {
    "inspect_model": {
        "description": "Inspect model checkpoint metadata, tensor manifest, formats, and parameter counts.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "checkpoint_path": {"type": "string", "description": "Absolute path to model checkpoint file"},
            },
            "required": ["checkpoint_path"],
        },
        "handler": lambda args: inspect_model(args["checkpoint_path"]),
    },
    "identify_architecture": {
        "description": "Detect model architecture family, attention layers, GQA heads, and dimensions from tensor names.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "checkpoint_path": {"type": "string", "description": "Path to checkpoint file or tensor dictionary"},
            },
            "required": ["checkpoint_path"],
        },
        "handler": lambda args: identify_architecture(args["checkpoint_path"]),
    },
    "reconstruct_graph": {
        "description": "Reconstruct computation DAG, execution layers, and tied parameter relationships from checkpoint.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "checkpoint_path": {"type": "string", "description": "Path to checkpoint file"},
            },
            "required": ["checkpoint_path"],
        },
        "handler": lambda args: reconstruct_graph(args["checkpoint_path"]),
    },
    "inspect_tensor": {
        "description": "Extract detailed numerical statistics (mean, std, min, max, top-k singular values) for a specific tensor.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "checkpoint_path": {"type": "string", "description": "Path to checkpoint file"},
                "tensor_name": {"type": "string", "description": "Name of tensor to inspect"},
            },
            "required": ["checkpoint_path", "tensor_name"],
        },
        "handler": lambda args: inspect_tensor(args["checkpoint_path"], args["tensor_name"]),
    },
    "summarize_weights": {
        "description": "Generate bulk statistics and numerical health metrics across checkpoint tensors.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "checkpoint_path": {"type": "string", "description": "Path to checkpoint file"},
                "tensor_names": {"type": "array", "items": {"type": "string"}, "description": "Optional subset of tensor names"},
            },
            "required": ["checkpoint_path"],
        },
        "handler": lambda args: summarize_weights(args["checkpoint_path"], args.get("tensor_names")),
    },
    "analyze_weight_directions": {
        "description": "Compute SVD principal directions, condition number, and effective stable rank for a 2D weight matrix.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "checkpoint_path": {"type": "string", "description": "Path to checkpoint file"},
                "tensor_name": {"type": "string", "description": "Name of 2D weight matrix"},
                "top_k": {"type": "integer", "description": "Number of singular values to return", "default": 5},
            },
            "required": ["checkpoint_path", "tensor_name"],
        },
        "handler": _handle_analyze_directions,
    },
    "validate_patch": {
        "description": "Verify hash compatibility, rank bounds, and shape consistency of a patch before application.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target_model_path": {"type": "string", "description": "Path to base model checkpoint"},
                "patch_path": {"type": "string", "description": "Path to patch manifest JSON file"},
            },
            "required": ["target_model_path", "patch_path"],
        },
        "handler": lambda args: validate_patch(_load_manifest_file(args["patch_path"]), args["target_model_path"]),
    },
    "apply_patch": {
        "description": "Apply a validated weight delta patch to create an updated checkpoint transactionally.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target_model_path": {"type": "string", "description": "Path to original base checkpoint"},
                "patch_path": {"type": "string", "description": "Path to patch manifest JSON file"},
                "output_model_path": {"type": "string", "description": "Path for new modified checkpoint"},
            },
            "required": ["target_model_path", "patch_path", "output_model_path"],
        },
        "handler": lambda args: apply_patch(_load_manifest_file(args["patch_path"]), args["target_model_path"], args["output_model_path"]),
    },
    "rollback_patch": {
        "description": "Rollback an applied patch and verify byte/tensor equivalence with original base model.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "patched_model_path": {"type": "string", "description": "Path to patched checkpoint"},
                "patch_path": {"type": "string", "description": "Path to patch manifest JSON file"},
                "restored_model_path": {"type": "string", "description": "Path for restored checkpoint"},
            },
            "required": ["patched_model_path", "patch_path", "restored_model_path"],
        },
        "handler": lambda args: rollback_patch(args["patched_model_path"], _load_manifest_file(args["patch_path"]), args["restored_model_path"]),
    },
    "profile_subspaces": {
        "description": "Profile weight-space manifolds, singular spectrum condition numbers, and effective stable rank.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "checkpoint_path": {"type": "string", "description": "Path to checkpoint file"},
                "cond_threshold": {"type": "number", "description": "Condition number threshold for degenerate manifolds", "default": 100.0},
            },
            "required": ["checkpoint_path"],
        },
        "handler": _handle_profile_subspaces,
    },
    "plan_surgery": {
        "description": "Formulate Pareto-constrained minimal-rank surgery plans with execution DAG and preservation bounds.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "checkpoint_path": {"type": "string", "description": "Path to checkpoint file"},
                "objective": {"type": "string", "description": "High-level transformation goal", "default": "Surgical parameter enhancement"},
                "preservation_epsilon": {"type": "number", "description": "Max permitted regression on anchor capabilities", "default": 0.01},
                "cond_limit": {"type": "number", "description": "Condition number limit for spectral stabilization", "default": 250.0},
            },
            "required": ["checkpoint_path"],
        },
        "handler": _handle_plan_surgery,
    },
    "verify_edit_gate": {
        "description": "Verify multi-layer Edit Gate safety: locks embeddings/norms/heads, enforces 1% cap and normalized L2 limits.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tensor_name": {"type": "string", "description": "Name of target tensor to modify"},
                "original": {"type": "array", "description": "Optional original tensor values for numerical validation"},
                "candidate": {"type": "array", "description": "Optional modified candidate tensor values"},
                "total_model_parameters": {"type": "integer", "description": "Total parameter count of base model", "default": 1000000},
            },
            "required": ["tensor_name"],
        },
        "handler": _handle_verify_edit_gate,
    },
    "dequantize_block": {
        "description": "Surgically dequantize 32-element Q8_0 or Q4_0 quantization blocks from hex bytes to float32 values.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "hex_data": {"type": "string", "description": "Hex-encoded raw quantized bytes (32 bytes for Q8_0, 16 bytes for Q4_0)"},
                "format": {"type": "string", "enum": ["q8_0", "q4_0"], "description": "Quantization format", "default": "q8_0"},
                "scale": {"type": "number", "description": "Block FP16 scaling factor", "default": 1.0},
            },
            "required": ["hex_data"],
        },
        "handler": _handle_dequantize_block,
    },
    "evaluate_critique": {
        "description": "Execute Critic-Discernment Game (CDG) to generate contrasting Helpful vs. Misleading critique pairs and score discernment.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "trace_id": {"type": "string", "description": "Unique identifier for trajectory", "default": "trace_001"},
                "problem": {"type": "string", "description": "Problem statement being solved"},
                "trace": {"type": "string", "description": "Reasoning trace or diagnosis"},
                "chosen_option": {"type": "string", "enum": ["A", "B"], "description": "Optional Prover selection ('A' or 'B') to evaluate"},
            },
            "required": ["trace"],
        },
        "handler": _handle_evaluate_critique,
    },
    "prune_traces": {
        "description": "Execute Think-Prune-Train (TPT) ground-truth anchored trajectory pruning to filter hallucinated reasoning steps.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "rollouts": {
                    "type": "array",
                    "description": "List of sampled candidate reasoning rollouts",
                    "items": {
                        "type": "object",
                        "properties": {
                            "rollout_id": {"type": "string"},
                            "problem": {"type": "string"},
                            "reasoning_trace": {"type": "string"},
                            "candidate_answer": {"type": "string"},
                            "expected_answer": {"type": "string"},
                        },
                        "required": ["problem", "candidate_answer"],
                    },
                },
            },
            "required": ["rollouts"],
        },
        "handler": _handle_prune_traces,
    },
}


class MCPServer:
    """Standard Model Context Protocol (MCP) JSON-RPC 2.0 server over stdio."""

    def __init__(self) -> None:
        self.handlers: Dict[str, Callable[[Any], Any]] = {
            "initialize": self._handle_initialize,
            "notifications/initialized": self._handle_initialized,
            "ping": self._handle_ping,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
        }

    def _handle_initialize(self, params: Any) -> Dict[str, Any]:
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {"listChanged": False},
            },
            "serverInfo": {
                "name": "weight-space-programming-engine",
                "version": "5.0.0",
            },
        }

    def _handle_initialized(self, params: Any) -> None:
        logger.info("MCP client initialized session")

    def _handle_ping(self, params: Any) -> Dict[str, Any]:
        return {}

    def _handle_tools_list(self, params: Any) -> Dict[str, Any]:
        tools: List[Dict[str, Any]] = []
        for name, entry in TOOLS_REGISTRY.items():
            tools.append({
                "name": name,
                "description": entry["description"],
                "inputSchema": entry["inputSchema"],
            })
        return {"tools": tools}

    def _handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if tool_name not in TOOLS_REGISTRY:
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Unknown tool: '{tool_name}'"}],
            }

        try:
            handler = TOOLS_REGISTRY[tool_name]["handler"]
            result = handler(arguments)
            serialized = json.dumps(_to_serializable(result), indent=2)
            return {
                "content": [{"type": "text", "text": serialized}],
            }
        except Exception as exc:
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Tool execution failed: {type(exc).__name__}: {str(exc)}"}],
            }

    def process_message(self, line: str) -> Optional[str]:
        """Process a single JSON-RPC message line and return response string if applicable."""
        line = line.strip()
        if not line:
            return None

        try:
            req = json.loads(line)
        except json.JSONDecodeError as err:
            return json.dumps({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {str(err)}"},
            })

        msg_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        # Notifications (no id)
        if msg_id is None:
            if method in self.handlers:
                self.handlers[method](params)
            return None

        # Request
        if method not in self.handlers:
            return json.dumps({
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Method not found: '{method}'"},
            })

        try:
            result = self.handlers[method](params)
            return json.dumps({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": result,
            })
        except Exception as exc:
            return json.dumps({
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32603, "message": f"Internal error: {str(exc)}"},
            })

    def run_stdio(self) -> None:
        """Run stdio loop reading line-delimited JSON-RPC messages."""
        for line in sys.stdin:
            response = self.process_message(line)
            if response:
                sys.stdout.write(response + "\n")
                sys.stdout.flush()


def main() -> None:
    server = MCPServer()
    server.run_stdio()


if __name__ == "__main__":
    main()
