#!/usr/bin/env python3
"""Safetensors adapter — inspect, enumerate and patch .safetensors checkpoints.

Implements the L1-L4 universal adapter contract (references/universal-adapter-contract-v2.md).

Usage:
    python safetensors_adapter.py inspect model.safetensors
    python safetensors_adapter.py patch model.safetensors --patch patch.json --output patched.safetensors
    python safetensors_adapter.py merge base.safetensors delta.safetensors --alpha 0.5 --output merged.safetensors

Supported operations in patch JSON:
    {"tensor_key": "model.layers.0.self_attn.q_proj.weight", "operation": "scale", "scale": 0.98, "output": "out.safetensors"}
    {"tensor_key": "...", "operation": "add", "delta_file": "delta.safetensors", "output": "out.safetensors"}
    {"base_sha256": "...", ...}  # optional hash verification
"""
import argparse
import hashlib
import json
import sys
import tempfile
import os
from pathlib import Path


def _require_safetensors():
    try:
        import safetensors.torch as st
        return st
    except ImportError:
        raise SystemExit("safetensors not installed. Run: pip install safetensors")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# L1 — Read-only: metadata and parameter enumeration
# ---------------------------------------------------------------------------

def load_metadata(path: str) -> dict:
    """Return header metadata without loading tensors into memory."""
    st = _require_safetensors()
    from safetensors import safe_open
    result = {"path": str(path), "tensors": [], "metadata": {}}
    with safe_open(path, framework="pt", device="cpu") as f:
        result["metadata"] = f.metadata() or {}
        for key in f.keys():
            info = f.get_slice(key)
            result["tensors"].append({
                "key": key,
                "shape": list(info.get_shape()),
                "dtype": str(info.get_dtype()),
            })
    return result


def enumerate_parameters(path: str) -> list:
    """Return per-tensor stats. Streams tensors one at a time."""
    import torch
    from safetensors import safe_open
    out = []
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            t = f.get_tensor(key).float()
            finite = torch.isfinite(t)
            y = t[finite]
            rec = {
                "key": key,
                "shape": list(t.shape),
                "dtype": str(f.get_slice(key).get_dtype()),
                "numel": t.numel(),
                "finite_fraction": float(finite.float().mean()),
            }
            if y.numel():
                qs = torch.quantile(y.flatten(), torch.tensor([0, .01, .1, .5, .9, .99, 1.])).tolist()
                rec.update({
                    "min": float(y.min()), "max": float(y.max()),
                    "mean": float(y.mean()), "std": float(y.std(unbiased=False)),
                    "l2_norm": float(torch.linalg.vector_norm(y)),
                    "zero_fraction": float((y == 0).float().mean()),
                    "quantiles_0_1_10_50_90_99_100": qs,
                })
            out.append(rec)
    return out


# ---------------------------------------------------------------------------
# L4 — Transform: apply patch to a copy
# ---------------------------------------------------------------------------

def apply_patch(checkpoint: str, spec: dict, output: str) -> dict:
    """Apply a JSON-declared patch to a copy. Never overwrites source."""
    import torch
    st = _require_safetensors()
    from safetensors import safe_open

    ckpt = Path(checkpoint).resolve()
    out_path = Path(output).resolve()
    if ckpt == out_path:
        raise SystemExit("Output must differ from source checkpoint.")

    # Optional hash check
    actual_hash = sha256_file(str(ckpt))
    expected = spec.get("base_sha256")
    if expected and actual_hash != expected:
        raise SystemExit(f"Hash mismatch: expected {expected}, got {actual_hash}")

    key = spec["tensor_key"]
    op = spec["operation"]

    # Load all tensors
    tensors = {}
    metadata = {}
    with safe_open(str(ckpt), framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    if key not in tensors:
        raise SystemExit(f"Tensor key not found: {key}")

    t = tensors[key].clone()

    if op == "scale":
        factor = float(spec["scale"])
        if not (0.0 <= factor <= 2.0):
            raise SystemExit("Scale outside conservative bound [0, 2].")
        tensors[key] = t * factor

    elif op == "add":
        delta_path = Path(spec["delta_file"]).resolve()
        with safe_open(str(delta_path), framework="pt", device="cpu") as df:
            delta_keys = list(df.keys())
            # Delta file may have single tensor or matching key
            dk = key if key in delta_keys else delta_keys[0]
            delta = df.get_tensor(dk)
        if tuple(delta.shape) != tuple(t.shape):
            raise SystemExit(f"Delta shape {delta.shape} != tensor shape {t.shape}")
        tensors[key] = (t + delta.to(dtype=t.dtype))

    else:
        raise SystemExit(f"Unsupported operation: {op}. Supported: scale, add")

    # Validate no NaNs/Infs introduced
    if not torch.isfinite(tensors[key]).all():
        raise SystemExit("Patch produced non-finite values; refusing to save.")

    # Write to temp then atomically rename
    fd, tmp = tempfile.mkstemp(suffix=".safetensors", dir=out_path.parent)
    os.close(fd)
    try:
        st.save_file(tensors, tmp, metadata=metadata)
        # Verify reload
        with safe_open(tmp, framework="pt", device="cpu") as vf:
            reloaded = vf.get_tensor(key)
            if tuple(reloaded.shape) != tuple(tensors[key].shape):
                raise SystemExit("Reload verification failed.")
        os.replace(tmp, str(out_path))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

    out_hash = sha256_file(str(out_path))
    return {
        "status": "ok",
        "base_sha256": actual_hash,
        "output": str(out_path),
        "output_sha256": out_hash,
        "changed_tensor": key,
        "operation": op,
    }


def merge_checkpoints(base: str, delta: str, alpha: float, output: str) -> dict:
    """Linear interpolation: result = base * (1-alpha) + delta * alpha (task arithmetic)."""
    import torch
    st = _require_safetensors()
    from safetensors import safe_open

    base_path, delta_path, out_path = Path(base).resolve(), Path(delta).resolve(), Path(output).resolve()
    if not (0.0 <= alpha <= 1.0):
        raise SystemExit("Alpha must be in [0, 1].")

    base_tensors, delta_tensors, metadata = {}, {}, {}
    with safe_open(str(base_path), framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        for k in f.keys():
            base_tensors[k] = f.get_tensor(k)
    with safe_open(str(delta_path), framework="pt", device="cpu") as f:
        for k in f.keys():
            delta_tensors[k] = f.get_tensor(k)

    merged = {}
    for k, bt in base_tensors.items():
        if k in delta_tensors:
            dt = delta_tensors[k].to(dtype=bt.dtype)
            if bt.shape == dt.shape:
                merged[k] = bt * (1 - alpha) + dt * alpha
            else:
                merged[k] = bt  # shape mismatch — keep base
        else:
            merged[k] = bt

    fd, tmp = tempfile.mkstemp(suffix=".safetensors", dir=out_path.parent)
    os.close(fd)
    try:
        st.save_file(merged, tmp, metadata=metadata)
        os.replace(tmp, str(out_path))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

    return {"status": "ok", "output": str(out_path), "alpha": alpha,
            "base": str(base_path), "delta": str(delta_path),
            "tensors_merged": sum(1 for k in base_tensors if k in delta_tensors)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_inspect(args):
    meta = load_metadata(args.checkpoint)
    if args.stats:
        meta["tensors"] = enumerate_parameters(args.checkpoint)
    print(json.dumps(meta, indent=2))


def cmd_patch(args):
    spec = json.loads(Path(args.patch).read_text())
    result = apply_patch(args.checkpoint, spec, args.output)
    print(json.dumps(result, indent=2))


def cmd_merge(args):
    result = merge_checkpoints(args.base, args.delta, args.alpha, args.output)
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description="Safetensors adapter — inspect, patch, merge")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect", help="List tensors and metadata")
    pi.add_argument("checkpoint")
    pi.add_argument("--stats", action="store_true", help="Compute per-tensor statistics (slower)")

    pp = sub.add_parser("patch", help="Apply a JSON patch to a safetensors checkpoint")
    pp.add_argument("checkpoint")
    pp.add_argument("--patch", required=True, help="Path to patch JSON file")
    pp.add_argument("--output", required=True, help="Output .safetensors path")

    pm = sub.add_parser("merge", help="Linear interpolation between two checkpoints")
    pm.add_argument("base")
    pm.add_argument("delta")
    pm.add_argument("--alpha", type=float, default=0.5, help="Mix weight for delta (0=base, 1=delta)")
    pm.add_argument("--output", required=True)

    args = p.parse_args()
    {"inspect": cmd_inspect, "patch": cmd_patch, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    main()
