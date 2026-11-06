#!/usr/bin/env python3
"""Download a HuggingFace model into models/<name>/ — weights stay gitignored."""
import argparse, os, sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="Download an HF model into models/")
    p.add_argument("repo_id", help="e.g. jaswanthsanjay88/rev-decision-model")
    p.add_argument("--name", help="Local folder name (default: repo name part)")
    p.add_argument("--no-bin", action="store_true", help="Skip legacy .bin files")
    a = p.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub not installed — run: pip install huggingface_hub")

    name = a.name or a.repo_id.split("/")[-1]
    out = Path("models") / name
    out.mkdir(parents=True, exist_ok=True)

    ignore = ["optimizer*", "training_args*", "rng_state*"]
    if a.no_bin:
        ignore.append("*.bin")

    print(f"Downloading {a.repo_id} -> {out} ...")
    path = snapshot_download(repo_id=a.repo_id, local_dir=str(out), ignore_patterns=ignore)
    # Report
    files = list(Path(path).rglob("*"))
    total = sum(f.stat().st_size for f in files if f.is_file())
    print(f"Done. {len([f for f in files if f.is_file()])} files, {total/1e6:.1f} MB")
    print(f"Path: {path}")

if __name__ == "__main__":
    main()
