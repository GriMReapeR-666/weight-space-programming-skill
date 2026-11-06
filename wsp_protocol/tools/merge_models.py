#!/usr/bin/env python3
"""merge_models.py — model merging algorithms: SLERP, Task Arithmetic, TIES, DARE.

Usage:
    # Linear interpolation (task arithmetic)
    python merge_models.py linear base.pt delta.pt --alpha 0.7 --output merged.pt

    # SLERP (spherical interpolation — better for LLMs)
    python merge_models.py slerp base.pt delta.pt --alpha 0.5 --output merged.pt

    # TIES (prune then merge — reduces interference)
    python merge_models.py ties base.pt delta1.pt delta2.pt --density 0.5 --output merged.pt

    # DARE (random parameter dropping before merge)
    python merge_models.py dare base.pt delta.pt --density 0.5 --alpha 0.5 --output merged.pt

Algorithms:
    linear:  W = W_base * (1-alpha) + W_delta * alpha
    slerp:   Spherical interpolation along the geodesic between two weight vectors
    ties:    Prune small delta values, resolve sign conflicts, merge remaining
    dare:    Randomly drop delta parameters before merging (reduces interference)

All operations work on plain .pt state_dicts and .safetensors files.
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def _load(path: str) -> dict:
    import torch
    p = Path(path)
    if p.suffix in (".pt", ".pth"):
        obj = torch.load(str(p), map_location="cpu", weights_only=True)
        if not isinstance(obj, dict):
            raise SystemExit(f"Expected state_dict, got {type(obj)}")
        return obj
    elif p.suffix == ".safetensors":
        from safetensors import safe_open
        result = {}
        with safe_open(str(p), framework="pt", device="cpu") as f:
            for k in f.keys():
                result[k] = f.get_tensor(k)
        return result
    raise SystemExit(f"Unsupported format: {p.suffix}")


def _save(obj: dict, path: str):
    import torch
    p = Path(path)
    if p.suffix == ".safetensors":
        import safetensors.torch as st
        fd, tmp = tempfile.mkstemp(suffix=".safetensors", dir=p.parent)
        os.close(fd)
        try:
            st.save_file(obj, tmp)
            os.replace(tmp, str(p))
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    else:
        fd, tmp = tempfile.mkstemp(suffix=".pt", dir=p.parent)
        os.close(fd)
        try:
            torch.save(obj, tmp)
            torch.load(tmp, map_location="cpu", weights_only=True)
            os.replace(tmp, str(p))
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


def _slerp_tensor(t0, t1, alpha: float):
    """Spherical linear interpolation between two tensors."""
    import torch
    t0f = t0.float().flatten()
    t1f = t1.float().flatten()
    dot = (t0f * t1f).sum().clamp(-1.0, 1.0)
    # If vectors are nearly parallel, fall back to linear
    if dot.abs() > 0.9995:
        return (t0 * (1 - alpha) + t1 * alpha).to(t0.dtype)
    theta = dot.acos()
    sin_theta = theta.sin()
    w0 = ((1 - alpha) * theta).sin() / sin_theta
    w1 = (alpha * theta).sin() / sin_theta
    result = (w0 * t0f + w1 * t1f).reshape(t0.shape)
    return result.to(t0.dtype)


def merge_linear(base: dict, delta: dict, alpha: float) -> dict:
    import torch
    result = {}
    n_merged = 0
    for k, bv in base.items():
        if k in delta and isinstance(bv, torch.Tensor) and isinstance(delta[k], torch.Tensor):
            dv = delta[k].to(dtype=bv.dtype)
            if bv.shape == dv.shape:
                result[k] = bv * (1 - alpha) + dv * alpha
                n_merged += 1
            else:
                result[k] = bv
        else:
            result[k] = bv
    return result, n_merged


def merge_slerp(base: dict, delta: dict, alpha: float) -> dict:
    import torch
    result = {}
    n_merged = 0
    for k, bv in base.items():
        if k in delta and isinstance(bv, torch.Tensor) and isinstance(delta[k], torch.Tensor):
            dv = delta[k].to(dtype=bv.dtype)
            if bv.shape == dv.shape and bv.dim() >= 1:
                result[k] = _slerp_tensor(bv, dv, alpha)
                n_merged += 1
            else:
                result[k] = bv
        else:
            result[k] = bv
    return result, n_merged


def merge_ties(base: dict, models: list, density: float, alpha: float) -> dict:
    """
    TIES merge (Trim, Elect Sign, Merge):
    1. Compute delta for each model vs base
    2. Trim small delta values (keep top-density fraction)
    3. Elect sign by majority vote
    4. Merge only tensors with elected sign
    """
    import torch
    result = {}
    n_merged = 0

    all_keys = set(base.keys())
    for m in models:
        all_keys &= set(m.keys())

    for k in base.keys():
        bv = base[k]
        if k not in all_keys or not isinstance(bv, torch.Tensor):
            result[k] = bv
            continue

        deltas = []
        for m in models:
            dv = m[k].to(dtype=bv.dtype)
            if dv.shape == bv.shape:
                deltas.append(dv - bv)

        if not deltas:
            result[k] = bv
            continue

        # 1. Trim: keep top density% by abs value per model
        trimmed = []
        for d in deltas:
            flat = d.flatten()
            k_keep = max(1, int(flat.numel() * density))
            threshold = flat.abs().topk(k_keep).values.min()
            mask = d.abs() >= threshold
            trimmed.append(d * mask)

        # 2. Elect sign: majority vote
        sign_stack = torch.stack([t.sign() for t in trimmed])
        elected_sign = sign_stack.sum(dim=0).sign()

        # 3. Merge: average only where sign matches elected
        merged_delta = torch.zeros_like(bv)
        counts = torch.zeros_like(bv)
        for t in trimmed:
            match = (t.sign() == elected_sign) | (elected_sign == 0)
            merged_delta += t * match
            counts += match.float()

        counts = counts.clamp(min=1)
        merged_delta = merged_delta / counts
        result[k] = bv + alpha * merged_delta
        n_merged += 1

    return result, n_merged


def merge_dare(base: dict, delta: dict, density: float, alpha: float, seed: int = 42) -> dict:
    """
    DARE merge (Drop And REscale):
    1. Compute delta = delta_model - base
    2. Randomly drop (1-density) fraction of delta parameters
    3. Rescale remaining by 1/density to preserve expected magnitude
    4. Merge: result = base + alpha * rescaled_delta
    """
    import torch
    rng = torch.Generator()
    rng.manual_seed(seed)
    result = {}
    n_merged = 0

    for k, bv in base.items():
        if k not in delta or not isinstance(bv, torch.Tensor):
            result[k] = bv
            continue
        dv = delta[k].to(dtype=bv.dtype)
        if dv.shape != bv.shape:
            result[k] = bv
            continue

        d = dv - bv
        # Random binary mask
        mask = torch.bernoulli(torch.full_like(d, density, dtype=torch.float), generator=rng).to(dtype=d.dtype)
        # Rescale
        rescaled = d * mask / max(density, 1e-8)
        result[k] = bv + alpha * rescaled
        n_merged += 1

    return result, n_merged


def _validate_no_nans(tensors: dict) -> list:
    import torch
    bad = [k for k, v in tensors.items() if isinstance(v, torch.Tensor) and not torch.isfinite(v).all()]
    return bad


def main():
    p = argparse.ArgumentParser(description="Model merging: linear, slerp, ties, dare")
    sub = p.add_subparsers(dest="method", required=True)

    for name in ("linear", "slerp"):
        sp = sub.add_parser(name)
        sp.add_argument("base")
        sp.add_argument("delta")
        sp.add_argument("--alpha", type=float, default=0.5, help="Mix weight (0=base, 1=delta)")
        sp.add_argument("--output", required=True)

    sp = sub.add_parser("ties")
    sp.add_argument("base")
    sp.add_argument("models", nargs="+", help="One or more fine-tuned model paths")
    sp.add_argument("--density", type=float, default=0.5, help="Fraction of delta params to keep")
    sp.add_argument("--alpha", type=float, default=1.0)
    sp.add_argument("--output", required=True)

    sp = sub.add_parser("dare")
    sp.add_argument("base")
    sp.add_argument("delta")
    sp.add_argument("--density", type=float, default=0.5, help="Fraction of delta params to keep")
    sp.add_argument("--alpha", type=float, default=1.0)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--output", required=True)

    args = p.parse_args()

    if args.method in ("linear", "slerp"):
        base = _load(args.base)
        delta = _load(args.delta)
        fn = merge_linear if args.method == "linear" else merge_slerp
        result, n = fn(base, delta, args.alpha)
    elif args.method == "ties":
        base = _load(args.base)
        models = [_load(m) for m in args.models]
        result, n = merge_ties(base, models, args.density, args.alpha)
    elif args.method == "dare":
        base = _load(args.base)
        delta = _load(args.delta)
        result, n = merge_dare(base, delta, args.density, args.alpha, args.seed)

    bad = _validate_no_nans(result)
    if bad:
        raise SystemExit(f"Non-finite values in {len(bad)} tensors after merge: {bad[:5]}")

    _save(result, args.output)
    print(json.dumps({
        "status": "ok",
        "method": args.method,
        "output": args.output,
        "tensors_merged": n,
        "total_tensors": len(result),
    }, indent=2))


if __name__ == "__main__":
    main()
