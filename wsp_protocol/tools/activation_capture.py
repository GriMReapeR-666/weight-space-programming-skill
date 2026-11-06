#!/usr/bin/env python3
"""Activation capture tool — instrument a PyTorch model and stream per-layer stats.

Usage:
    python activation_capture.py --model model.pt --prompts prompts.txt --layers "0,1,2" --output activations.jsonl
    python activation_capture.py --model model.safetensors --config config.json --prompts prompts.txt --output activations.jsonl

What it records per forward pass per layer:
  - layer name, call index, input/output shapes
  - output norm, mean, std, finite fraction
  - dead unit fraction (fraction of output neurons that are zero across all tokens)
  - optional: per-token entropy of softmax(output) for classifier heads

Does NOT retain raw tensors — only compact statistics. Safe for large models.
"""
import argparse
import json
import sys
from pathlib import Path


def _make_hook(layer_name: str, records: list, call_counters: dict, top_k_outliers: int = 4):
    """Return a PyTorch forward hook that appends compact stats to records."""
    import torch

    def hook(module, inp, out):
        idx = call_counters.get(layer_name, 0)
        call_counters[layer_name] = idx + 1

        # Handle tuple outputs (e.g. attention returns (attn_out, attn_weights))
        tensor = out[0] if isinstance(out, (tuple, list)) else out
        if not isinstance(tensor, torch.Tensor):
            return

        x = tensor.detach().float()
        finite = torch.isfinite(x)
        finite_frac = float(finite.float().mean())
        y = x[finite]

        rec = {
            "layer": layer_name,
            "call_index": idx,
            "input_shapes": [list(i.shape) for i in inp if isinstance(i, torch.Tensor)],
            "output_shape": list(tensor.shape),
            "output_dtype": str(tensor.dtype),
            "finite_fraction": finite_frac,
        }

        if y.numel() > 0:
            # Flatten to [batch*seq, hidden] or just flat
            flat = y.reshape(-1, y.shape[-1]) if y.dim() >= 2 else y.unsqueeze(0)
            norms = flat.norm(dim=-1)
            rec.update({
                "output_norm_mean": float(norms.mean()),
                "output_norm_std": float(norms.std(unbiased=False)) if norms.numel() > 1 else 0.0,
                "output_mean": float(y.mean()),
                "output_std": float(y.std(unbiased=False)) if y.numel() > 1 else 0.0,
                "output_max_abs": float(y.abs().max()),
                # Dead unit: neurons that are zero across the whole batch/sequence
                "dead_unit_fraction": float((flat.abs().max(dim=0).values == 0).float().mean()),
            })
            # Outlier detection — top-k by abs value
            flat_abs = y.abs().flatten()
            k = min(top_k_outliers, flat_abs.numel())
            vals, _ = flat_abs.topk(k)
            rec["top_abs_values"] = [float(v) for v in vals]

        records.append(rec)

    return hook


def capture(
    model_path: str,
    prompts: list,
    layer_filter: list = None,
    output_path: str = None,
    config_path: str = None,
) -> list:
    """
    Attach hooks to all named modules (or filtered subset), run forward passes,
    collect per-layer stats, return list of records.
    """
    import torch

    # --- Load model ---
    p = Path(model_path)
    if p.suffix in (".pt", ".pth"):
        obj = torch.load(str(p), map_location="cpu", weights_only=True)
        if isinstance(obj, dict):
            raise SystemExit(
                "File is a state_dict, not a full model. "
                "Pass a full serialized nn.Module or use with a config."
            )
        model = obj
    elif p.suffix == ".safetensors":
        # Try loading with transformers if config present
        try:
            from transformers import AutoModel
            cfg_dir = str(Path(config_path).parent) if config_path else str(p.parent)
            model = AutoModel.from_pretrained(cfg_dir, torch_dtype=torch.float32)
        except Exception as e:
            raise SystemExit(f"Cannot load safetensors model: {e}\nProvide a HuggingFace model directory.")
    else:
        raise SystemExit(f"Unsupported format: {p.suffix}. Supported: .pt, .pth, .safetensors")

    model.eval()

    # --- Attach hooks ---
    records = []
    call_counters = {}
    handles = []
    for name, module in model.named_modules():
        if layer_filter and not any(f in name for f in layer_filter):
            continue
        if name == "":  # skip root module
            continue
        h = module.register_forward_hook(_make_hook(name, records, call_counters))
        handles.append(h)

    # --- Run forward passes ---
    try:
        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                # Minimal tokenization — works for nn.Embedding-based models
                # For real LLMs, caller should pass token IDs
                if isinstance(prompt, str):
                    # Convert to ASCII codes as a rough stand-in; real use passes token IDs
                    ids = torch.tensor([[ord(c) % 256 for c in prompt[:128]]], dtype=torch.long)
                else:
                    ids = torch.tensor([prompt], dtype=torch.long)
                try:
                    model(ids)
                except Exception as e:
                    # Mark error but continue
                    records.append({"prompt_index": i, "error": str(e)})
    finally:
        for h in handles:
            h.remove()

    if output_path:
        with open(output_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {len(records)} records to {output_path}", file=sys.stderr)

    return records


def main():
    p = argparse.ArgumentParser(description="Capture per-layer activation statistics")
    p.add_argument("--model", required=True, help="Path to .pt/.pth model or safetensors dir")
    p.add_argument("--config", help="config.json path (for safetensors/HF models)")
    p.add_argument("--prompts", required=True, help="Text file with one prompt per line")
    p.add_argument("--layers", help="Comma-separated layer name substrings to filter (default: all)")
    p.add_argument("--output", help="Output JSONL file (default: stdout)")
    p.add_argument("--top-k", type=int, default=4, help="Top-k outlier values to record")
    args = p.parse_args()

    prompts = Path(args.prompts).read_text().strip().splitlines()
    layer_filter = [x.strip() for x in args.layers.split(",")] if args.layers else None

    records = capture(
        model_path=args.model,
        prompts=prompts,
        layer_filter=layer_filter,
        output_path=args.output,
        config_path=args.config,
    )

    if not args.output:
        print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
