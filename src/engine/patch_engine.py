"""PatchEngine: Transactional, reversible, non-destructive weight delta application and rollback."""

from __future__ import annotations

import datetime
import os
import tempfile
import uuid
from pathlib import Path
from typing import List

import torch

from .loader import ModelLoader
from .types import ModelFormat, PatchManifest, PatchOperation


def generate_patch(
    target_model_path: str,
    operations: List[PatchOperation],
    description: str,
    author: str = "WSP-Engine-v5",
) -> PatchManifest:
    """Generate a cryptographic, reversible patch manifest."""
    base_hash = ModelLoader.compute_sha256(target_model_path)
    patch_id = f"wsp-patch-{uuid.uuid4().hex[:12]}"

    return PatchManifest(
        patch_id=patch_id,
        target_model_path=str(Path(target_model_path).resolve()),
        base_model_hash=base_hash,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        author=author,
        description=description,
        operations=operations,
        is_reversible=True,
    )


def validate_patch(manifest: PatchManifest, target_model_path: str) -> bool:
    """Validate that a patch manifest matches the target model hash and parameter shapes."""
    current_hash = ModelLoader.compute_sha256(target_model_path)
    if current_hash != manifest.base_model_hash:
        raise ValueError(
            f"Base model hash mismatch! Expected {manifest.base_model_hash}, got {current_hash}"
        )

    # Validate target tensor existence and shape
    existing_shapes = {}
    for name, tensor in ModelLoader.stream_tensors(target_model_path):
        existing_shapes[name] = list(tensor.shape)

    for op in manifest.operations:
        if op.tensor_name not in existing_shapes:
            raise KeyError(f"Target tensor '{op.tensor_name}' does not exist in model")
        if existing_shapes[op.tensor_name] != op.shape:
            raise ValueError(
                f"Shape mismatch on '{op.tensor_name}': expected {op.shape}, "
                f"found {existing_shapes[op.tensor_name]}"
            )

    return True


def apply_patch(
    manifest: PatchManifest,
    target_model_path: str,
    output_path: str,
) -> str:
    """Apply patch operations to target model non-destructively, writing to output_path."""
    validate_patch(manifest, target_model_path)
    fmt = ModelLoader.detect_format(target_model_path)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Build operations map
    op_map = {op.tensor_name: op for op in manifest.operations}

    if fmt == ModelFormat.SAFETENSORS:
        from safetensors import safe_open
        import safetensors.torch as st

        tensors = {}
        with safe_open(target_model_path, framework="pt", device="cpu") as sf:
            for k in sf.keys():
                t = sf.get_tensor(k)
                if k in op_map and op_map[k].data is not None:
                    op = op_map[k]
                    raw_data = op.data
                    assert raw_data is not None
                    delta = raw_data if torch.is_tensor(raw_data) else torch.tensor(raw_data)
                    t = t + (op.alpha * delta.to(t.device, dtype=t.dtype))
                tensors[k] = t

        fd, tmp = tempfile.mkstemp(suffix=".safetensors", dir=out_p.parent)
        os.close(fd)
        try:
            st.save_file(tensors, tmp)
            os.replace(tmp, str(out_p))
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    elif fmt == ModelFormat.PYTORCH:
        sd = torch.load(target_model_path, map_location="cpu", weights_only=True)
        if not isinstance(sd, dict):
            raise SystemExit("Expected a PyTorch state_dict")

        for k in list(sd.keys()):
            if k in op_map and op_map[k].data is not None:
                op = op_map[k]
                raw_data = op.data
                assert raw_data is not None
                delta = raw_data if torch.is_tensor(raw_data) else torch.tensor(raw_data)
                sd[k] = sd[k] + (op.alpha * delta.to(sd[k].device, dtype=sd[k].dtype))

        fd, tmp = tempfile.mkstemp(suffix=".pt", dir=out_p.parent)
        os.close(fd)
        try:
            torch.save(sd, tmp)
            os.replace(tmp, str(out_p))
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    else:
        raise NotImplementedError(f"Patching format '{fmt}' not supported directly.")

    return str(out_p.resolve())


def verify_model_equivalence(path_a: str, path_b: str, atol: float = 1e-5) -> bool:
    """Verify that two checkpoints have identical tensor names, shapes, and values."""
    tensors_a = dict(ModelLoader.stream_tensors(path_a))
    tensors_b = dict(ModelLoader.stream_tensors(path_b))

    if set(tensors_a.keys()) != set(tensors_b.keys()):
        return False

    for k in tensors_a:
        ta = tensors_a[k]
        tb = tensors_b[k]
        if ta.shape != tb.shape:
            return False
        # Adapt numerical tolerance for half-precision floating point representations
        eff_atol = 5e-4 if ta.dtype in (torch.float16, torch.bfloat16) else atol
        if not torch.allclose(ta.float(), tb.float(), atol=eff_atol):
            return False
    return True


def rollback_patch(
    patched_model_path: str,
    manifest: PatchManifest,
    output_path: str,
) -> str:
    """Reverse a patch by subtracting the applied delta and verifying restoration."""
    if not manifest.is_reversible:
        raise ValueError(f"Patch '{manifest.patch_id}' is marked as non-reversible")

    # Invert operations
    inverse_ops = []
    for op in manifest.operations:
        inv_op = PatchOperation(
            tensor_name=op.tensor_name,
            operation="subtract",
            shape=op.shape,
            data=op.data,
            alpha=-1.0 * op.alpha,
        )
        inverse_ops.append(inv_op)

    inv_manifest = PatchManifest(
        patch_id=f"rollback-{manifest.patch_id}",
        target_model_path=patched_model_path,
        base_model_hash=ModelLoader.compute_sha256(patched_model_path),
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        author=manifest.author,
        description=f"Rollback of {manifest.patch_id}",
        operations=inverse_ops,
        is_reversible=False,
    )

    out_file = apply_patch(inv_manifest, patched_model_path, output_path)

    # Verify restored model tensor values match original base model
    if not verify_model_equivalence(manifest.target_model_path, out_file):
        raise RuntimeError(
            f"Rollback verification failed: restored weights in '{out_file}' "
            f"do not match original base weights in '{manifest.target_model_path}'"
        )

    return out_file
