"""ModelLoader: Memory-bounded multi-format checkpoint reader and inspector."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .types import ModelFormat, ModelMetadata


class ModelLoader:
    """Safely loads and inspects model checkpoints with strict memory constraints."""

    @staticmethod
    def detect_format(path: str) -> ModelFormat:
        """Detect checkpoint format from magic bytes and file extension."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ext = p.suffix.lower()
        try:
            with open(path, "rb") as f:
                header = f.read(16)
        except Exception:
            header = b""

        # 1. GGUF magic bytes: 'GGUF' (0x47 0x47 0x55 0x46)
        if header.startswith(b"GGUF"):
            return ModelFormat.GGUF

        # 2. Safetensors: First 8 bytes are uint64 indicating JSON header size
        if ext == ".safetensors":
            return ModelFormat.SAFETENSORS
        if len(header) >= 8:
            header_size = struct.unpack("<Q", header[:8])[0]
            # Valid safetensors JSON header is typically 100 bytes to 100MB
            if 10 < header_size < 100 * 1024 * 1024:
                return ModelFormat.SAFETENSORS

        # 3. PyTorch (.pt, .pth, .bin)
        if ext in (".pt", ".pth", ".bin"):
            return ModelFormat.PYTORCH

        # 4. ONNX
        if ext == ".onnx" or header.startswith(b"\x08"):
            return ModelFormat.ONNX

        return ModelFormat.UNKNOWN

    @staticmethod
    def compute_sha256(path: str, chunk_size: int = 4 * 1024 * 1024) -> str:
        """Compute cryptographic SHA-256 fingerprint with streaming buffer."""
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                sha.update(chunk)
        return sha.hexdigest()

    @staticmethod
    def load_metadata(path: str) -> ModelMetadata:
        """Extract model metadata without loading full tensor arrays into RAM."""
        p = Path(path)
        fmt = ModelLoader.detect_format(path)
        size = p.stat().st_size

        tensors: List[Dict[str, Any]] = []
        extra: Dict[str, Any] = {}
        dtype_counts: Dict[str, int] = {}
        total_params = 0

        if fmt == ModelFormat.SAFETENSORS:
            try:
                from safetensors import safe_open
                with safe_open(path, framework="pt", device="cpu") as sf:
                    for k in sf.keys():
                        sl = sf.get_slice(k)
                        shape = list(sl.get_shape())
                        dtype_str = str(sl.get_dtype())
                        numel = 1
                        for d in shape:
                            numel *= d
                        total_params += numel
                        dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1
                        tensors.append({"name": k, "shape": shape, "dtype": dtype_str})
                    extra["metadata"] = sf.metadata() or {}
            except Exception as e:
                extra["error"] = str(e)

        elif fmt == ModelFormat.GGUF:
            try:
                import gguf
                reader = gguf.GGUFReader(path)
                for t in reader.tensors:
                    shape = list(t.shape)
                    dtype_str = str(t.tensor_type.name)
                    numel = int(t.n_elements)
                    total_params += numel
                    dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1
                    tensors.append({"name": t.name, "shape": shape, "dtype": dtype_str})
                extra["architecture"] = reader.get_field("general.architecture")
            except Exception as e:
                extra["error"] = str(e)

        elif fmt == ModelFormat.PYTORCH:
            try:
                import torch
                # weights_only=True guarantees safe deserialization
                sd = torch.load(path, map_location="cpu", weights_only=True)
                if isinstance(sd, dict):
                    for k, v in sd.items():
                        if hasattr(v, "shape"):
                            shape = list(v.shape)
                            dtype_str = str(v.dtype)
                            numel = int(v.numel())
                            total_params += numel
                            dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1
                            tensors.append({"name": k, "shape": shape, "dtype": dtype_str})
            except Exception as e:
                extra["error"] = str(e)

        # Infer basic layer count from tensor names
        layer_indices = set()
        for t_info in tensors:
            import re
            m = re.search(r"(?:layers?|blocks?|blk)\.(\d+)", str(t_info["name"]))
            if m:
                layer_indices.add(int(m.group(1)))
        layer_count = len(layer_indices)

        return ModelMetadata(
            path=str(p.resolve()),
            format=fmt,
            file_size_bytes=size,
            n_tensors=len(tensors),
            total_parameters=total_params,
            architecture_guess="dense_transformer" if layer_count > 0 else "unknown",
            layer_count=layer_count,
            data_types=dtype_counts,
            extra_metadata=extra,
        )

    @staticmethod
    def stream_tensors(
        path: str, max_tensors: Optional[int] = None
    ) -> Iterator[Tuple[str, Any]]:
        """Yield tensors on CPU one at a time for chunk-wise, low-RAM processing."""
        fmt = ModelLoader.detect_format(path)
        count = 0

        if fmt == ModelFormat.SAFETENSORS:
            from safetensors import safe_open
            with safe_open(path, framework="pt", device="cpu") as sf:
                for k in sf.keys():
                    if max_tensors and count >= max_tensors:
                        break
                    yield k, sf.get_tensor(k)
                    count += 1

        elif fmt == ModelFormat.PYTORCH:
            import torch
            sd = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(sd, dict):
                for k, v in sd.items():
                    if max_tensors and count >= max_tensors:
                        break
                    yield k, v
                    count += 1
