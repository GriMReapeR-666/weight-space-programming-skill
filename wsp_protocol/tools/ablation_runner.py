#!/usr/bin/env python3
"""Activation ablation runner — causal probe: does zeroing this layer change the output?

Runs a forward pass normally, then with selected activations zeroed/replaced,
and measures the behavioral delta. This is a *diagnostic probe*, not a weight edit.

Usage:
    python ablation_runner.py --model model.pt --prompts prompts.txt --layer "layers.0.mlp" --output ablation.jsonl
    python ablation_runner.py --model model.pt --prompts prompts.txt --layer "attn" --mode mean --output ablation.jsonl

Ablation modes:
    zero   — replace activation with zeros
    mean   — replace with mean value (preserves scale)
    noise  — replace with Gaussian noise matching std
    freeze — freeze activation at value from first example (counterfactual patch)

Output per example per layer:
    - clean output logits norm, top-token
    - ablated output logits norm, top-token
    - cosine similarity between clean and ablated final representations
    - L2 distance between clean and ablated final representations
    - effect size: how much did this layer matter?
"""
import argparse
import json
import sys
from pathlib import Path
from contextlib import contextmanager


def _cosine_sim(a, b):
    import torch
    a_n = a / (a.norm() + 1e-8)
    b_n = b / (b.norm() + 1e-8)
    return float((a_n * b_n).sum())


@contextmanager
def _ablation_hook(model, layer_name: str, mode: str, frozen_activation=None):
    """Context manager that attaches an ablation hook and removes it on exit."""
    import torch
    handle = None

    def hook(module, inp, out):
        tensor = out[0] if isinstance(out, (tuple, list)) else out
        if not isinstance(tensor, torch.Tensor):
            return out

        if mode == "zero":
            ablated = torch.zeros_like(tensor)
        elif mode == "mean":
            ablated = torch.full_like(tensor, tensor.mean().item())
        elif mode == "noise":
            ablated = torch.randn_like(tensor) * tensor.std()
        elif mode == "freeze":
            if frozen_activation is not None:
                ablated = frozen_activation.expand_as(tensor).to(tensor.dtype)
            else:
                ablated = torch.zeros_like(tensor)
        else:
            return out

        if isinstance(out, tuple):
            return (ablated,) + out[1:]
        return ablated

    # Find target module
    target = None
    for name, module in model.named_modules():
        if layer_name in name:
            target = module
            break
    if target is None:
        raise ValueError(f"Layer '{layer_name}' not found in model")

    handle = target.register_forward_hook(hook)
    try:
        yield target
    finally:
        handle.remove()


def run_ablation(
    model_path: str,
    prompts: list,
    layer_name: str,
    mode: str = "zero",
    output_path: str = None,
) -> list:
    import torch

    p = Path(model_path)
    if p.suffix not in (".pt", ".pth"):
        raise SystemExit("ablation_runner currently supports .pt/.pth full models only.")

    model = torch.load(str(p), map_location="cpu", weights_only=False)
    if isinstance(model, dict):
        raise SystemExit("File is a state_dict. Need a full nn.Module.")
    model.eval()

    results = []
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            ids = torch.tensor([[ord(c) % 256 for c in prompt[:128]]], dtype=torch.long) \
                  if isinstance(prompt, str) else torch.tensor([prompt], dtype=torch.long)

            # --- Clean pass ---
            try:
                clean_out = model(ids)
                clean_tensor = clean_out[0] if isinstance(clean_out, tuple) else clean_out
                clean_flat = clean_tensor.detach().float().flatten()
            except Exception as e:
                results.append({"prompt_index": i, "error": f"clean pass failed: {e}"})
                continue

            # --- Ablated pass ---
            try:
                frozen = clean_tensor.detach().clone() if mode == "freeze" and i == 0 else None
                with _ablation_hook(model, layer_name, mode, frozen):
                    ablated_out = model(ids)
                    ablated_tensor = ablated_out[0] if isinstance(ablated_out, tuple) else ablated_out
                    ablated_flat = ablated_tensor.detach().float().flatten()
            except Exception as e:
                results.append({"prompt_index": i, "error": f"ablated pass failed: {e}"})
                continue

            # Measure effect
            cos = _cosine_sim(clean_flat, ablated_flat)
            l2 = float((clean_flat - ablated_flat).norm())
            effect_size = 1.0 - cos  # 0 = no effect, 1 = complete change

            rec = {
                "prompt_index": i,
                "layer": layer_name,
                "mode": mode,
                "clean_output_norm": float(clean_flat.norm()),
                "ablated_output_norm": float(ablated_flat.norm()),
                "cosine_similarity": cos,
                "l2_distance": l2,
                "effect_size": effect_size,
                "interpretation": (
                    "negligible" if effect_size < 0.05 else
                    "small" if effect_size < 0.2 else
                    "moderate" if effect_size < 0.5 else
                    "large"
                ),
            }

            # Top-token comparison if output looks like logits [batch, seq, vocab]
            if clean_tensor.dim() == 3:
                ct = clean_tensor[0, -1].float()
                at = ablated_tensor[0, -1].float()
                rec["clean_top_token_id"] = int(ct.argmax())
                rec["ablated_top_token_id"] = int(at.argmax())
                rec["top_token_changed"] = rec["clean_top_token_id"] != rec["ablated_top_token_id"]

            results.append(rec)

    if output_path:
        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {len(results)} ablation records to {output_path}", file=sys.stderr)

    return results


def summarize(records: list) -> dict:
    """Aggregate ablation results — which layers had the largest effect."""
    import statistics
    effects = [r["effect_size"] for r in records if "effect_size" in r]
    if not effects:
        return {"error": "no valid records"}
    return {
        "layer": records[0].get("layer"),
        "mode": records[0].get("mode"),
        "n_prompts": len(effects),
        "mean_effect_size": statistics.mean(effects),
        "max_effect_size": max(effects),
        "min_effect_size": min(effects),
        "verdict": (
            "CRITICAL — ablating this layer significantly changes output"
            if statistics.mean(effects) > 0.3 else
            "MODERATE — this layer has measurable effect"
            if statistics.mean(effects) > 0.1 else
            "LOW — ablating this layer has little effect"
        ),
    }


def main():
    p = argparse.ArgumentParser(description="Activation ablation — causal probe")
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", required=True, help="Text file, one prompt per line")
    p.add_argument("--layer", required=True, help="Layer name substring to ablate")
    p.add_argument("--mode", default="zero", choices=["zero", "mean", "noise", "freeze"])
    p.add_argument("--output", help="JSONL output file")
    p.add_argument("--summary", action="store_true", help="Print summary instead of full records")
    args = p.parse_args()

    prompts = Path(args.prompts).read_text().strip().splitlines()
    records = run_ablation(args.model, prompts, args.layer, args.mode, args.output)

    if args.summary:
        print(json.dumps(summarize(records), indent=2))
    elif not args.output:
        print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
