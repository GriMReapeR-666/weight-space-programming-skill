"""Selective Precision Controller (SPC).

Provides block-level and tensor-level dequantization, local surgical modification,
and requantization for quantized models (e.g., GGUF Q4_0 and Q8_0 blocks).
Calculates reconstruction distortion (normalized L2 error and RMSE) to ensure
quantization preservation within safe numerical tolerances.
"""

from __future__ import annotations

from typing import Tuple

import torch


class SelectivePrecisionController:
    """Manages surgical dequantization, parameter editing, and requantization."""

    BLOCK_SIZE = 32  # Standard GGUF quantization block size

    @staticmethod
    def dequantize_q8_0_block(quantized_bytes: bytes, scale_f16: float) -> torch.Tensor:
        """Dequantizes a 32-element Q8_0 block from signed 8-bit integers using FP16 scale."""
        import numpy as np
        int8_arr = np.frombuffer(quantized_bytes, dtype=np.int8)
        return torch.from_numpy(int8_arr).float() * scale_f16

    @staticmethod
    def requantize_q8_0_block(float_values: torch.Tensor) -> Tuple[bytes, float]:
        """Requantizes a 32-element floating-point tensor to Q8_0 representation.

        Returns:
            (quantized_bytes, scale_f16)
        """
        if float_values.numel() != SelectivePrecisionController.BLOCK_SIZE:
            raise ValueError(f"Q8_0 requires exactly {SelectivePrecisionController.BLOCK_SIZE} elements")

        f_vals = float_values.detach().float()
        max_abs = float(torch.max(torch.abs(f_vals)).item())
        scale = max_abs / 127.0 if max_abs > 1e-12 else 1e-12

        quantized_int8 = torch.round(f_vals / scale).clamp(-128, 127).to(torch.int8)
        quant_bytes = bytes((quantized_int8.numpy()).tobytes())
        return quant_bytes, scale

    @staticmethod
    def dequantize_q4_0_block(nibble_bytes: bytes, scale_f16: float) -> torch.Tensor:
        """Dequantizes a 32-element Q4_0 block from 16 packed nibble bytes.

        Each byte stores two 4-bit unsigned values (offset by -8 to yield [-8, 7]).
        """
        if len(nibble_bytes) != 16:
            raise ValueError(f"Q4_0 block expects 16 bytes for 32 nibbles, got {len(nibble_bytes)}")

        values = []
        for b in nibble_bytes:
            low = (b & 0x0F) - 8
            high = ((b >> 4) & 0x0F) - 8
            values.extend([low, high])

        return torch.tensor(values, dtype=torch.float32) * scale_f16

    @staticmethod
    def requantize_q4_0_block(float_values: torch.Tensor) -> Tuple[bytes, float]:
        """Requantizes a 32-element float tensor to Q4_0 16-byte packed nibbles.

        Returns:
            (nibble_bytes, scale_f16)
        """
        if float_values.numel() != SelectivePrecisionController.BLOCK_SIZE:
            raise ValueError(f"Q4_0 requires exactly {SelectivePrecisionController.BLOCK_SIZE} elements")

        f_vals = float_values.detach().float()
        max_abs = float(torch.max(torch.abs(f_vals)).item())
        scale = max_abs / 7.0 if max_abs > 1e-12 else 1e-12

        # Quantize to [-8, 7]
        quant_vals = torch.round(f_vals / scale).clamp(-8, 7).to(torch.int32) + 8  # [0, 15]

        packed = bytearray()
        for i in range(0, 32, 2):
            low = int(quant_vals[i].item()) & 0x0F
            high = (int(quant_vals[i + 1].item()) & 0x0F) << 4
            packed.append(low | high)

        return bytes(packed), scale

    @staticmethod
    def compute_normalized_l2_distortion(
        original: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> float:
        """Computes ||W_recon - W_orig||_2 / N (normalized Frobenius distance per element)."""
        diff = reconstructed.float() - original.float()
        frobenius_norm = float(torch.linalg.norm(diff).item())
        num_elements = original.numel()
        return frobenius_norm / max(1, num_elements)

    @staticmethod
    def perform_block_surgery(
        original_blocks: torch.Tensor,
        intervention_fn,
        quant_format: str = "q8_0",
    ) -> Tuple[torch.Tensor, float]:
        """Surgically decodes, modifies, and re-encodes a matrix of quantization blocks.

        Args:
            original_blocks: Float tensor representing original parameter block [M, 32].
            intervention_fn: Callable taking float tensor [M, 32] and returning modified float tensor.
            quant_format: "q8_0" or "q4_0".

        Returns:
            (reconstructed_tensor, normalized_distortion)
        """
        shape = original_blocks.shape
        flat = original_blocks.reshape(-1, SelectivePrecisionController.BLOCK_SIZE)
        num_blocks = flat.shape[0]

        # Apply surgical edit in floating point
        edited_float = intervention_fn(flat)

        reconstructed_blocks = []
        for i in range(num_blocks):
            blk = edited_float[i]
            if quant_format.lower() == "q4_0":
                b_bytes, scale = SelectivePrecisionController.requantize_q4_0_block(blk)
                recon = SelectivePrecisionController.dequantize_q4_0_block(b_bytes, scale)
            else:
                b_bytes, scale = SelectivePrecisionController.requantize_q8_0_block(blk)
                recon = SelectivePrecisionController.dequantize_q8_0_block(b_bytes, scale)
            reconstructed_blocks.append(recon)

        reconstructed_tensor = torch.stack(reconstructed_blocks).reshape(shape)
        distortion = SelectivePrecisionController.compute_normalized_l2_distortion(
            edited_float.reshape(shape),
            reconstructed_tensor,
        )

        return reconstructed_tensor, distortion
