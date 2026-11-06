#!/usr/bin/env python3
"""GGUF reader — inspect and dequantize GGUF model files for analysis.

GGUF is llama.cpp's container format. This tool:
  1. Parses the GGUF header (metadata key-value pairs)
  2. Lists all tensors with their quantization type, shape, name
  3. Dequantizes selected tensors to float32 for inspection/patching
  4. Exports dequantized tensors as a plain .pt state_dict (for use with patch_delta.py)

Usage:
    python gguf_reader.py inspect model.gguf
    python gguf_reader.py dequant model.gguf --tensors "token_embd.weight,blk.0.attn_q.weight" --output dequant.pt
    python gguf_reader.py stats model.gguf --top-k 10

WARNING: Dequantization is lossy. The exported float32 tensors cannot be
re-quantized to byte-perfect GGUF without quantization error. Always measure
round-trip error before using for production patching.

Requires: gguf (pip install gguf)
"""
import argparse
import json
import sys
from pathlib import Path


def _require_gguf():
    try:
        import gguf
        return gguf
    except ImportError:
        raise SystemExit("gguf not installed. Run: pip install gguf")


def inspect_gguf(path: str) -> dict:
    """Parse GGUF header and tensor inventory without loading weights."""
    gguf = _require_gguf()
    reader = gguf.GGUFReader(path, "r")

    metadata = {}
    for field in reader.fields.values():
        try:
            if field.types and field.types[0] == gguf.GGUFValueType.STRING:
                metadata[field.name] = str(bytes(field.parts[-1]), encoding="utf-8")
            elif len(field.data) == 1:
                metadata[field.name] = field.data[0]
            else:
                metadata[field.name] = list(field.data)
        except Exception:
            metadata[field.name] = "<unparseable>"

    tensors = []
    for tensor in reader.tensors:
        tensors.append({
            "name": tensor.name,
            "shape": list(reversed(tensor.shape.tolist())),  # GGUF shape is row-major reversed
            "n_elements": int(tensor.n_elements),
            "quant_type": str(tensor.tensor_type.name),
            "data_offset": int(tensor.data_offset),
        })

    return {
        "path": str(path),
        "n_tensors": len(tensors),
        "metadata": metadata,
        "tensors": tensors,
    }


def dequantize_tensors(path: str, tensor_names: list = None, output_path: str = None) -> dict:
    """
    Dequantize selected (or all) tensors to float32 and optionally save as .pt state_dict.

    Returns dict of {name: torch.Tensor}
    """
    gguf = _require_gguf()
    import torch
    import numpy as np

    reader = gguf.GGUFReader(path, "r")
    result = {}
    errors = {}

    for tensor in reader.tensors:
        name = tensor.name
        if tensor_names and name not in tensor_names:
            continue
        try:
            # Use gguf's built-in dequantization
            data = tensor.data  # returns numpy array, dequantized to float32
            shape = list(reversed(tensor.shape.tolist()))
            arr = np.array(data, dtype=np.float32).reshape(shape)
            result[name] = torch.from_numpy(arr)
        except Exception as e:
            errors[name] = str(e)

    if errors:
        print(f"WARNING: Failed to dequantize {len(errors)} tensors: {list(errors.keys())}", file=sys.stderr)

    if output_path and result:
        import os, tempfile
        p = Path(output_path)
        fd, tmp = tempfile.mkstemp(suffix=".pt", dir=p.parent)
        os.close(fd)
        try:
            torch.save(result, tmp)
            torch.load(tmp, map_location="cpu", weights_only=True)  # verify
            os.replace(tmp, str(p))
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        print(f"Saved {len(result)} dequantized tensors to {output_path}", file=sys.stderr)

    return result


def compute_quant_error(original_path: str, dequant_tensors: dict, sample_keys: list = None) -> list:
    """
    Measure dequantization error: compare dequantized float32 against
    re-quantized values. Reports per-tensor RMSE and max absolute error.
    """
    import torch
    import numpy as np

    keys = sample_keys or list(dequant_tensors.keys())[:8]  # default: first 8
    results = []

    for key in keys:
        if key not in dequant_tensors:
            continue
        t = dequant_tensors[key].float()
        # Simulate Q8_0 quantization error (block quantization, 32 elements per block)
        block_size = 32
        flat = t.flatten()
        n_blocks = flat.numel() // block_size
        if n_blocks == 0:
            continue
        flat_blocks = flat[:n_blocks * block_size].reshape(n_blocks, block_size)
        scales = flat_blocks.abs().max(dim=1, keepdim=True).values / 127.0
        quantized = torch.round(flat_blocks / scales.clamp(min=1e-8)).clamp(-127, 127)
        reconstructed = quantized * scales
        error = flat_blocks - reconstructed
        results.append({
            "key": key,
            "shape": list(t.shape),
            "quant_simulation": "Q8_0_approx",
            "rmse": float(error.pow(2).mean().sqrt()),
            "max_abs_error": float(error.abs().max()),
            "relative_rmse": float(error.pow(2).mean().sqrt() / (flat_blocks.abs().mean() + 1e-8)),
        })

    return results


def cmd_inspect(args):
    info = inspect_gguf(args.model)
    print(json.dumps(info, indent=2))


def cmd_dequant(args):
    names = [t.strip() for t in args.tensors.split(",")] if args.tensors else None
    result = dequantize_tensors(args.model, names, args.output)
    summary = {
        "dequantized": list(result.keys()),
        "output": args.output,
        "note": "Use patch_delta.py or safetensors_adapter.py for further editing.",
    }
    print(json.dumps(summary, indent=2))


def cmd_stats(args):
    import torch
    names_filter = [t.strip() for t in args.tensors.split(",")] if getattr(args, "tensors", None) else None
    tensors = dequantize_tensors(args.model, names_filter)
    stats = []
    for key, t in list(tensors.items())[:args.top_k]:
        y = t.float().flatten()
        finite = torch.isfinite(y)
        y_f = y[finite]
        if y_f.numel() == 0:
            stats.append({"key": key, "shape": list(t.shape), "error": "all non-finite"})
            continue
        qs = torch.quantile(y_f, torch.tensor([0., .25, .5, .75, 1.])).tolist()
        stats.append({
            "key": key,
            "shape": list(t.shape),
            "numel": t.numel(),
            "mean": float(y_f.mean()),
            "std": float(y_f.std(unbiased=False)),
            "l2_norm": float(torch.linalg.vector_norm(y_f)),
            "quantiles_0_25_50_75_100": qs,
            "zero_fraction": float((y_f == 0).float().mean()),
            "finite_fraction": float(finite.float().mean()),
        })
    print(json.dumps(stats, indent=2))


def main():
    p = argparse.ArgumentParser(description="GGUF reader — inspect and dequantize GGUF models")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect", help="Parse GGUF header and tensor inventory")
    pi.add_argument("model")

    pd = sub.add_parser("dequant", help="Dequantize tensors to float32 .pt state_dict")
    pd.add_argument("model")
    pd.add_argument("--tensors", help="Comma-separated tensor names (default: all)")
    pd.add_argument("--output", required=True, help="Output .pt file")

    ps = sub.add_parser("stats", help="Compute statistics on dequantized tensors")
    ps.add_argument("model")
    ps.add_argument("--tensors", help="Comma-separated tensor names to analyze")
    ps.add_argument("--top-k", type=int, default=8, help="Max tensors to analyze")

    args = p.parse_args()
    {"inspect": cmd_inspect, "dequant": cmd_dequant, "stats": cmd_stats}[args.cmd](args)


if __name__ == "__main__":
    main()
