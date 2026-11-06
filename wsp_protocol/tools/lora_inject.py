#!/usr/bin/env python3
"""LoRA injection tool — add trainable low-rank delta matrices to any linear layer.

What it does:
    Given a checkpoint and a target linear layer, injects LoRA A/B matrices
    (rank-decomposition delta: ΔW = B @ A) WITHOUT modifying the original weights.
    The delta can then be:
      - Merged into the base weights: W' = W + (alpha/rank) * B @ A
      - Saved as a separate LoRA adapter file
      - Applied as a patch using patch_delta.py

Usage:
    # Inspect which layers are patchable
    python lora_inject.py inspect model.pt

    # Compute and merge a random-init LoRA (baseline test)
    python lora_inject.py merge model.pt --layer "weight" --rank 8 --alpha 1.0 --output merged.pt

    # Merge from a saved LoRA adapter file
    python lora_inject.py merge model.pt --lora-file lora_adapter.pt --output merged.pt

    # Save a zero-delta LoRA scaffold (ready for training)
    python lora_inject.py scaffold model.pt --layer "weight" --rank 8 --output lora_scaffold.pt

LoRA math:
    W_new = W + (alpha / rank) * B @ A
    A: [rank, in_features]  — initialized with kaiming_uniform
    B: [out_features, rank] — initialized with zeros (so delta starts at 0)
"""
import argparse
import json
import sys
import os
import tempfile
from pathlib import Path


def _load_state_dict(path: str):
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict):
        raise SystemExit(f"Expected a state_dict (.pt), got {type(obj)}")
    return obj


def _save_state_dict(obj: dict, path: str):
    import torch
    fd, tmp = tempfile.mkstemp(suffix=".pt", dir=Path(path).parent)
    os.close(fd)
    try:
        torch.save(obj, tmp)
        # Verify reload
        torch.load(tmp, map_location="cpu", weights_only=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def inspect_linear_layers(state_dict: dict) -> list:
    """Return all keys that look like weight matrices of linear layers."""
    import torch
    layers = []
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.dim() == 2:  # linear weights are 2D
            out_features, in_features = tensor.shape
            layers.append({
                "key": key,
                "shape": [out_features, in_features],
                "dtype": str(tensor.dtype),
                "numel": tensor.numel(),
                "lora_params_rank8": 8 * (out_features + in_features),
                "compression_ratio_rank8": round(tensor.numel() / (8 * (out_features + in_features)), 1),
            })
    return sorted(layers, key=lambda x: x["numel"], reverse=True)


def create_lora_matrices(out_features: int, in_features: int, rank: int) -> tuple:
    """Create A and B LoRA matrices. B=0 so initial delta is zero."""
    import torch
    A = torch.empty(rank, in_features)
    torch.nn.init.kaiming_uniform_(A)
    B = torch.zeros(out_features, rank)
    return A, B


def merge_lora(state_dict: dict, key: str, A, B, alpha: float, rank: int) -> dict:
    """Merge LoRA delta into base weights: W' = W + (alpha/rank) * B @ A"""
    import torch
    if key not in state_dict:
        raise SystemExit(f"Key not found: {key}")
    W = state_dict[key]
    if W.dim() != 2:
        raise SystemExit(f"Key {key} is not a 2D weight matrix.")
    delta = (alpha / rank) * (B.to(W.dtype) @ A.to(W.dtype))
    if delta.shape != W.shape:
        raise SystemExit(f"Delta shape {delta.shape} != weight shape {W.shape}")
    W_new = W + delta
    if not torch.isfinite(W_new).all():
        raise SystemExit("Merged weights contain non-finite values.")
    result = dict(state_dict)
    result[key] = W_new
    return result


def cmd_inspect(args):
    sd = _load_state_dict(args.checkpoint)
    layers = inspect_linear_layers(sd)
    print(json.dumps({
        "checkpoint": args.checkpoint,
        "total_tensors": len(sd),
        "linear_layers": len(layers),
        "layers": layers,
    }, indent=2))


def cmd_scaffold(args):
    """Save a zero-delta LoRA adapter file (A, B matrices) ready for training."""
    import torch
    sd = _load_state_dict(args.checkpoint)

    # Find target keys
    target_keys = [k for k in sd if args.layer in k and sd[k].dim() == 2]
    if not target_keys:
        raise SystemExit(f"No 2D weight matrices found matching '{args.layer}'")

    lora_adapter = {}
    for key in target_keys:
        W = sd[key]
        out_f, in_f = W.shape
        A, B = create_lora_matrices(out_f, in_f, args.rank)
        lora_adapter[f"{key}.lora_A"] = A
        lora_adapter[f"{key}.lora_B"] = B
        lora_adapter[f"{key}.alpha"] = torch.tensor(float(args.alpha))
        lora_adapter[f"{key}.rank"] = torch.tensor(float(args.rank))

    _save_state_dict(lora_adapter, args.output)
    print(json.dumps({
        "status": "ok",
        "output": args.output,
        "rank": args.rank,
        "alpha": args.alpha,
        "patched_keys": target_keys,
        "note": "B matrices initialized to zero — initial delta is 0. Train A and B to adapt.",
    }, indent=2))


def cmd_merge(args):
    import torch
    sd = _load_state_dict(args.checkpoint)

    if args.lora_file:
        # Load existing LoRA adapter
        lora = _load_state_dict(args.lora_file)
        # Discover target keys from lora file
        keys_with_A = [k.removesuffix(".lora_A") for k in lora if k.endswith(".lora_A")]
        result = dict(sd)
        merged_keys = []
        for base_key in keys_with_A:
            if base_key not in sd:
                print(f"WARNING: {base_key} not in checkpoint, skipping", file=sys.stderr)
                continue
            A = lora[f"{base_key}.lora_A"]
            B = lora[f"{base_key}.lora_B"]
            alpha = float(lora.get(f"{base_key}.alpha", torch.tensor(1.0)))
            rank = int(lora.get(f"{base_key}.rank", torch.tensor(A.shape[0])))
            result = merge_lora(result, base_key, A, B, alpha, rank)
            merged_keys.append(base_key)
    else:
        # Create random LoRA (useful for testing that the pipeline works)
        target_keys = [k for k in sd if args.layer in k and sd[k].dim() == 2]
        if not target_keys:
            raise SystemExit(f"No 2D weight matrices matching '{args.layer}'")
        result = dict(sd)
        merged_keys = []
        for key in target_keys:
            W = sd[key]
            A, B = create_lora_matrices(W.shape[0], W.shape[1], args.rank)
            result = merge_lora(result, key, A, B, args.alpha, args.rank)
            merged_keys.append(key)

    _save_state_dict(result, args.output)
    print(json.dumps({
        "status": "ok",
        "output": args.output,
        "merged_keys": merged_keys,
    }, indent=2))


def main():
    p = argparse.ArgumentParser(description="LoRA injection — add rank-decomposition deltas to linear layers")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect", help="List all linear layers and LoRA param counts")
    pi.add_argument("checkpoint")

    ps = sub.add_parser("scaffold", help="Create a zero-delta LoRA adapter file (ready for training)")
    ps.add_argument("checkpoint")
    ps.add_argument("--layer", required=True, help="Layer name substring to target")
    ps.add_argument("--rank", type=int, default=8)
    ps.add_argument("--alpha", type=float, default=16.0)
    ps.add_argument("--output", required=True)

    pm = sub.add_parser("merge", help="Merge LoRA delta into base weights")
    pm.add_argument("checkpoint")
    pm.add_argument("--lora-file", help="Existing LoRA adapter .pt file")
    pm.add_argument("--layer", help="Layer name substring (if not using --lora-file)")
    pm.add_argument("--rank", type=int, default=8)
    pm.add_argument("--alpha", type=float, default=1.0)
    pm.add_argument("--output", required=True)

    args = p.parse_args()
    {"inspect": cmd_inspect, "scaffold": cmd_scaffold, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    main()
