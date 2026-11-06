#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()
p=argparse.ArgumentParser()
p.add_argument("directory")
p.add_argument("--output",default="checkpoint-manifest.json")
a=p.parse_args()
root=Path(a.directory)
items=[{"path":str(x.relative_to(root)),"size":x.stat().st_size,"sha256":sha(x)}
       for x in sorted(root.rglob("*")) if x.is_file()]
Path(a.output).write_text(json.dumps(items,indent=2),encoding="utf-8")
print(f"Wrote {len(items)} entries to {a.output}")
