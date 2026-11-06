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
    elif cmd in ("-h", "--help", "help"):
        print(__doc__)
    else:
        print(f"Unknown command: {cmd}\n\nAvailable: inspect patch merge lora scaffold log")
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
        import safetensors_adapter, json
        from pathlib import Path
        spec = json.loads(Path(args.patch).read_text())
        result = safetensors_adapter.apply_patch(args.checkpoint, spec, args.output)
        import json as _json; print(_json.dumps(result, indent=2))
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


if __name__ == "__main__":
    main()
