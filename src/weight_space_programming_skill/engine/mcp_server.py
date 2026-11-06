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

from .graph_engine import identify_architecture, reconstruct_graph
from .loader import ModelLoader
from .patch_engine import apply_patch, rollback_patch, validate_patch
from .tensor_engine import (
    analyze_weight_directions,
    inspect_model,
    inspect_tensor,
    summarize_weights,
)
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
