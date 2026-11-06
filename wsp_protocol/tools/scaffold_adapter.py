#!/usr/bin/env python3
"""Auto-scaffold generator — inspect any checkpoint and emit an adapter skeleton.

Given a checkpoint file, this tool:
  1. Reads all tensor names and shapes
  2. Classifies each tensor's likely role using heuristic name matching
  3. Groups tensors by probable module (layer, block, head)
  4. Emits a skeleton adapter Python file + a JSON architecture map

Usage:
    python scaffold_adapter.py model.pt --output my_adapter.py
    python scaffold_adapter.py model.safetensors --output my_adapter.py --map arch_map.json
    python scaffold_adapter.py model.pt --json  # just print the architecture map

The generated adapter skeleton is a starting point — it implements the
L1-L2 contract and stubs L3-L4 for you to fill in.
"""
import argparse
import json
import re
import sys
from pathlib import Path


# Heuristic patterns for classifying tensor roles
# (pattern, role, component_family)
ROLE_PATTERNS = [
    # Embeddings
    (r"embed(?:ding)?s?(?:_weight)?|token_emb|wte|wpe", "embedding", "embedding"),
    # Attention
    (r"(?:self_attn|attention|attn)\.q(?:_proj)?", "attention_query", "transformer_attention"),
    (r"(?:self_attn|attention|attn)\.k(?:_proj)?", "attention_key", "transformer_attention"),
    (r"(?:self_attn|attention|attn)\.v(?:_proj)?", "attention_value", "transformer_attention"),
    (r"(?:self_attn|attention|attn)\.o(?:_proj)?|out_proj", "attention_output", "transformer_attention"),
    (r"(?:self_attn|attention|attn)\.(?:in_proj|qkv)", "attention_qkv_fused", "transformer_attention"),
    # MLP / FFN
    (r"mlp\.(?:gate|fc1|up)_proj|ffn\.(?:gate|w1|up)", "ffn_gate", "transformer_ffn"),
    (r"mlp\.(?:down|fc2)_proj|ffn\.(?:down|w2)", "ffn_down", "transformer_ffn"),
    (r"mlp\.(?:up|fc3)_proj|ffn\.up", "ffn_up", "transformer_ffn"),
    (r"mlp\.|ffn\.|feed_forward\.", "ffn_generic", "transformer_ffn"),
    # MoE
    (r"expert|router|gate(?:_weight)?", "moe_routing", "moe"),
    # Normalization
    (r"(?:layer_norm|rms_norm|ln_[12f]|norm)\.weight", "norm_scale", "normalization"),
    (r"(?:layer_norm|rms_norm|ln_[12f]|norm)\.bias", "norm_bias", "normalization"),
    # LM head / classifier
    (r"lm_head|output_projection|classifier|head\.", "output_head", "output"),
    # Convolution
    (r"conv(?:olution)?(?:\d+)?\.(weight|bias)", "conv_weight", "cnn"),
    # Positional / RoPE / ALiBi
    (r"rotary|rope|alibi|pos(?:ition)?_emb", "positional_encoding", "positional"),
    # State space / SSM
    (r"ssm|state_space|mamba|A_log|B_proj|C_proj|dt_proj|D\b", "ssm_parameter", "ssm"),
    # Bias terms
    (r"\.bias$", "bias", "generic"),
    # Generic weight
    (r"\.weight$", "weight", "generic"),
]


def classify_tensor(name: str, shape: list) -> dict:
    """Return heuristic role classification for a tensor."""
    name_lower = name.lower()
    for pattern, role, family in ROLE_PATTERNS:
        if re.search(pattern, name_lower):
            return {"role": role, "component_family": family, "confidence": "heuristic"}
    # Shape-based fallback
    if len(shape) == 1:
        return {"role": "bias_or_scale", "component_family": "generic", "confidence": "shape_heuristic"}
    if len(shape) == 2:
        return {"role": "linear_weight", "component_family": "generic", "confidence": "shape_heuristic"}
    if len(shape) >= 4:
        return {"role": "conv_or_embedding", "component_family": "cnn_or_embedding", "confidence": "shape_heuristic"}
    return {"role": "unknown", "component_family": "unknown", "confidence": "none"}


def infer_layer_index(name: str) -> int | None:
    """Extract layer/block index from tensor name."""
    m = re.search(r"(?:layers?|blocks?|blk)\.(\d+)", name)
    if m:
        return int(m.group(1))
    m = re.search(r"\.(\d+)\.", name)
    if m:
        return int(m.group(1))
    return None


def build_arch_map(tensors: list) -> dict:
    """Build architecture map from tensor inventory."""
    families = {}
    layers = {}
    total_params = 0

    for t in tensors:
        name, shape = t["name"], t["shape"]
        cls = classify_tensor(name, shape)
        numel = 1
        for s in shape:
            numel *= s
        total_params += numel

        fam = cls["component_family"]
        families.setdefault(fam, {"tensors": [], "total_params": 0})
        families[fam]["tensors"].append(name)
        families[fam]["total_params"] += numel

        layer_idx = infer_layer_index(name)
        if layer_idx is not None:
            layers.setdefault(layer_idx, []).append(name)

    # Infer architecture family
    fam_names = set(families.keys())
    if "transformer_attention" in fam_names and "transformer_ffn" in fam_names:
        arch_guess = "dense_transformer"
    elif "moe" in fam_names:
        arch_guess = "mixture_of_experts"
    elif "ssm" in fam_names:
        arch_guess = "state_space_model"
    elif "cnn" in fam_names:
        arch_guess = "cnn"
    elif "embedding" in fam_names and len(fam_names) <= 3:
        arch_guess = "embedding_model"
    else:
        arch_guess = "unknown"

    return {
        "architecture_guess": arch_guess,
        "confidence": "heuristic_only",
        "total_parameters": total_params,
        "n_tensors": len(tensors),
        "n_layers": len(layers),
        "component_families": {k: {"total_params": v["total_params"], "n_tensors": len(v["tensors"])}
                               for k, v in families.items()},
        "layer_indices": sorted(layers.keys()),
        "tensors": [
            {**t, **classify_tensor(t["name"], t["shape"]), "layer_index": infer_layer_index(t["name"])}
            for t in tensors
        ],
    }


def load_tensor_inventory(path: str) -> list:
    """Load tensor names and shapes from any supported format."""
    import torch
    p = Path(path)

    if p.suffix in (".pt", ".pth"):
        obj = torch.load(str(p), map_location="cpu", weights_only=True)
        if not isinstance(obj, dict):
            raise SystemExit("Expected a state_dict (.pt)")
        return [{"name": k, "shape": list(v.shape), "dtype": str(v.dtype)}
                for k, v in obj.items() if isinstance(v, torch.Tensor)]

    elif p.suffix == ".safetensors":
        from safetensors import safe_open
        result = []
        with safe_open(str(p), framework="pt", device="cpu") as f:
            for key in f.keys():
                sl = f.get_slice(key)
                result.append({"name": key, "shape": list(sl.get_shape()), "dtype": str(sl.get_dtype())})
        return result

    elif p.suffix == ".gguf":
        try:
            import gguf
            reader = gguf.GGUFReader(str(p), "r")
            return [{"name": t.name, "shape": list(reversed(t.shape.tolist())), "dtype": str(t.tensor_type.name)}
                    for t in reader.tensors]
        except ImportError:
            raise SystemExit("gguf not installed. Run: pip install gguf")

    else:
        raise SystemExit(f"Unsupported format: {p.suffix}. Supported: .pt, .pth, .safetensors, .gguf")


ADAPTER_TEMPLATE = '''#!/usr/bin/env python3
"""Auto-generated adapter scaffold for: {checkpoint_name}

Architecture guess: {arch_guess} (confidence: heuristic_only)
Total parameters: {total_params:,}
N layers: {n_layers}

This scaffold implements L1-L2 of the universal adapter contract.
Fill in the TODO sections to reach L3 (diagnostic) and L4 (transform).

Generated by: scaffold_adapter.py
"""
import json
from pathlib import Path
import torch

CHECKPOINT_PATH = "{checkpoint_path}"
ARCH_MAP_PATH = "{map_path}"  # set to path of arch_map.json if generated

MATURITY = {{
    "L1_read_only": "implemented",
    "L2_structural": "partial",
    "L3_diagnostic": "stub",
    "L4_transform": "stub",
    "L5_training": "not_implemented",
    "L6_export": "not_implemented",
}}


def load_metadata(path: str = CHECKPOINT_PATH) -> dict:
    """L1: Return tensor inventory without loading all weights."""
    p = Path(path)
    if p.suffix in (".pt", ".pth"):
        sd = torch.load(str(p), map_location="cpu", weights_only=True)
        return {{
            "path": str(p),
            "n_tensors": len(sd),
            "tensors": [
                {{"name": k, "shape": list(v.shape), "dtype": str(v.dtype)}}
                for k, v in sd.items() if isinstance(v, torch.Tensor)
            ],
        }}
    elif p.suffix == ".safetensors":
        from safetensors import safe_open
        tensors = []
        with safe_open(str(p), framework="pt", device="cpu") as f:
            for key in f.keys():
                sl = f.get_slice(key)
                tensors.append({{"name": key, "shape": list(sl.get_shape()), "dtype": str(sl.get_dtype())}})
        return {{"path": str(p), "n_tensors": len(tensors), "tensors": tensors}}
    raise NotImplementedError(f"Format not supported: {{p.suffix}}")


def enumerate_parameters(path: str = CHECKPOINT_PATH) -> list:
    """L1: Return per-tensor statistics."""
    # TODO: stream tensors and compute stats (use tensor_summary.py as reference)
    return load_metadata(path)["tensors"]


def map_parameter_to_component(name: str) -> dict:
    """L2: Map a tensor name to its architectural role."""
    # Based on scaffold analysis — refine with actual model source/config
    ROLE_MAP = {{
{role_map_entries}
    }}
    return ROLE_MAP.get(name, {{"role": "unknown", "component_family": "unknown"}})


def baseline(prompts: list, **kwargs) -> dict:
    """L3: Run model on prompts and return task metrics. STUB — implement for your task."""
    # TODO: load model, run inference, compute task-specific metrics
    raise NotImplementedError("Implement baseline() for task-specific evaluation")


def plan_transform(spec: dict) -> dict:
    """L4: Validate a transform spec and return a plan. STUB."""
    # TODO: validate tensor key, operation, bounds; return equation and rollback plan
    return {{
        "tensor_key": spec.get("tensor_key"),
        "operation": spec.get("operation"),
        "preconditions": ["tensor exists", "shape matches", "dtype compatible"],
        "equation": "TODO: fill in math",
        "rollback": "restore from base checkpoint",
        "status": "stub",
    }}


def apply_transform(spec: dict, destination: str):
    """L4: Apply a transform to a copy. STUB — use patch_delta.py for .pt or safetensors_adapter.py for .safetensors."""
    raise NotImplementedError("Use patch_delta.py or safetensors_adapter.py")


if __name__ == "__main__":
    import json
    print(json.dumps(load_metadata(), indent=2))
'''


def generate_adapter_code(arch_map: dict, checkpoint_path: str, map_path: str) -> str:
    """Generate the adapter Python file from the arch map."""
    # Build role map entries
    entries = []
    for t in arch_map["tensors"][:100]:  # cap at 100 for readability
        name = t["name"]
        role = t.get("role", "unknown")
        family = t.get("component_family", "unknown")
        entries.append(f'        "{name}": {{"role": "{role}", "component_family": "{family}"}},')

    return ADAPTER_TEMPLATE.format(
        checkpoint_name=Path(checkpoint_path).name,
        arch_guess=arch_map["architecture_guess"],
        total_params=arch_map["total_parameters"],
        n_layers=arch_map["n_layers"],
        checkpoint_path=checkpoint_path,
        map_path=map_path or "",
        role_map_entries="\n".join(entries),
    )


def main():
    p = argparse.ArgumentParser(description="Auto-scaffold adapter from any checkpoint")
    p.add_argument("checkpoint", help=".pt, .safetensors, or .gguf checkpoint file")
    p.add_argument("--output", help="Output adapter .py file")
    p.add_argument("--map", help="Output architecture map .json file")
    p.add_argument("--json", action="store_true", help="Print arch map JSON only (no adapter code)")
    args = p.parse_args()

    tensors = load_tensor_inventory(args.checkpoint)
    arch_map = build_arch_map(tensors)

    if args.map:
        Path(args.map).write_text(json.dumps(arch_map, indent=2))
        print(f"Architecture map written to {args.map}", file=sys.stderr)

    if args.json or not args.output:
        # Print compact summary
        summary = {k: v for k, v in arch_map.items() if k != "tensors"}
        print(json.dumps(summary, indent=2))
        return

    if args.output:
        code = generate_adapter_code(arch_map, args.checkpoint, args.map or "")
        Path(args.output).write_text(code)
        print(f"Adapter scaffold written to {args.output}", file=sys.stderr)
        # Also print summary
        summary = {k: v for k, v in arch_map.items() if k != "tensors"}
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
