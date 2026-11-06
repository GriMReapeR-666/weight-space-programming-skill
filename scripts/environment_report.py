#!/usr/bin/env python3
import json
import os
import platform
import shutil
import sys

r = {
    "python": sys.version,
    "platform": platform.platform(),
    "cpu_count": os.cpu_count(),
    "disk_free_bytes": shutil.disk_usage(".").free,
    "packages": {},
    "gpus": [],
}

for name in ["torch", "transformers", "safetensors", "peft", "jax", "tensorflow", "numpy", "scipy"]:
    try:
        m = __import__(name)
        r["packages"][name] = getattr(m, "__version__", "installed")
    except Exception:
        r["packages"][name] = None

try:
    import torch
    r["cuda_available"] = torch.cuda.is_available()
    if r["cuda_available"]:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            r["gpus"].append({"name": p.name, "memory_bytes": p.total_memory})
except Exception:
    r["cuda_available"] = False

print(json.dumps(r, indent=2))
