#!/usr/bin/env python3
"""Apply a small JSON-declared patch to a PyTorch state_dict checkpoint copy.

This is a deliberately narrow reference utility, not a universal checkpoint editor.
Requires torch. It never overwrites the input file. Use only trusted checkpoints.
Supported patch JSON:
{
  "base_sha256": "...",
  "tensor_key": "layer.weight",
  "operation": "scale" | "add",
  "scale": 0.99,
  "delta_file": "delta.pt",
  "output": "patched.pt"
}
For add, delta_file must contain a tensor with the exact selected tensor shape.
"""
import argparse, hashlib, json, os, tempfile
from pathlib import Path

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--patch",required=True)
    args=ap.parse_args()
    import torch
    ckpt=Path(args.checkpoint).resolve()
    spec=json.loads(Path(args.patch).read_text())
    actual=sha256(ckpt)
    expected=spec.get("base_sha256")
    if expected and actual != expected:
        raise SystemExit(f"Base hash mismatch: expected {expected}, got {actual}")
    if ckpt.suffix.lower() not in {".pt",".pth"}:
        raise SystemExit("Reference utility supports only .pt/.pth; use a format-specific adapter.")
    obj=torch.load(ckpt,map_location="cpu",weights_only=True)
    # Support plain state_dict only; nested training checkpoints need an adapter.
    if not isinstance(obj,dict) or not all(isinstance(k,str) for k in obj):
        raise SystemExit("Expected a plain state_dict mapping.")
    key=spec["tensor_key"]
    if key not in obj or not torch.is_tensor(obj[key]):
        raise SystemExit(f"Tensor key not found: {key}")
    t=obj[key].clone()
    op=spec["operation"]
    if op=="scale":
        factor=float(spec["scale"])
        if not (0.0 <= factor <= 2.0):
            raise SystemExit("Scale outside conservative reference bound [0,2].")
        obj[key]=t*factor
    elif op=="add":
        delta_path=Path(spec["delta_file"]).resolve()
        delta=torch.load(delta_path,map_location="cpu",weights_only=True)
        if not torch.is_tensor(delta) or tuple(delta.shape)!=tuple(t.shape):
            raise SystemExit("Delta must be a tensor with exact matching shape.")
        obj[key]=t+delta.to(dtype=t.dtype)
    else:
        raise SystemExit("Unsupported operation.")
    if not torch.isfinite(obj[key]).all():
        raise SystemExit("Patch produced non-finite values; refusing to save.")
    out=Path(spec["output"]).resolve()
    if out==ckpt:
        raise SystemExit("Output must differ from source checkpoint.")
    out.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=out.name+".",suffix=".tmp",dir=out.parent)
    os.close(fd)
    try:
        torch.save(obj,tmp)
        # Verify it can be reloaded before promotion.
        chk=torch.load(tmp,map_location="cpu",weights_only=True)
        if tuple(chk[key].shape)!=tuple(obj[key].shape):
            raise SystemExit("Reload verification failed.")
        os.replace(tmp,out)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    print(json.dumps({"status":"ok","base_sha256":actual,"output":str(out),"output_sha256":sha256(out),"changed_tensor":key},indent=2))

if __name__=="__main__": main()
