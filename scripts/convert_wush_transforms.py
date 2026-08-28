#!/usr/bin/env python3
"""Convert WUSH-KV calibration transforms into an SGLang runtime checkpoint.

The WUSH-KV exporter writes a dict with two tensors::

    {
        "k": Tensor[num_layers, num_kv_heads, num_blocks, d_k, d_k],
        "v": Tensor[num_layers, num_kv_heads, num_blocks, d_v, d_v],
    }

The tensors are canonical column-vector transforms ``T_k`` and ``T_v``.  The
SGLang attention implementation uses row vectors, so this script materializes
the four right-multiplication matrices needed by the non-folded runtime path::

    K' = K @ k_right       where k_right = T_k.T
    Q' = Q @ q_right       where q_right = inv(T_k)
    V' = V @ v_right       where v_right = T_v.T
    O  = O' @ o_right      where o_right = inv(T_v).T

All matrices remain global (all layers and all KV heads).  The runtime loader
is responsible for selecting its pipeline-parallel layers and attention-TP KV
heads.

Example::

    python scripts/convert_wush_transforms.py \
        --input /path/to/transforms/Qwen3-8B/WUSH.pt \
        --output /path/to/transforms/Qwen3-8B/WUSH-sglang.pt
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import torch

FORMAT_NAME = "sglang_wush_kv_runtime_v1"

_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _validate_source_tensor(name: str, tensor: Any) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Source key {name!r} must contain a tensor")
    if tensor.ndim != 5:
        raise ValueError(
            f"Source {name!r} must have shape "
            "[layers, kv_heads, blocks, block_dim, block_dim], "
            f"got {tuple(tensor.shape)}"
        )
    if tensor.shape[-1] != tensor.shape[-2]:
        raise ValueError(
            f"Source {name!r} transform blocks must be square, "
            f"got {tuple(tensor.shape[-2:])}"
        )
    if tensor.shape[-1] <= 0 or tensor.shape[-3] <= 0:
        raise ValueError(f"Source {name!r} has an empty block dimension")
    if not tensor.is_floating_point():
        raise TypeError(
            f"Source {name!r} must be floating point, got dtype={tensor.dtype}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError(f"Source {name!r} contains NaN or infinity")
    return tensor.detach().to(device="cpu").contiguous()


def _invert_in_batches(
    matrices: torch.Tensor,
    *,
    batch_size: int,
    name: str,
) -> tuple[torch.Tensor, float]:
    """Invert the trailing square matrices in FP64 with bounded workspace."""
    if batch_size <= 0:
        raise ValueError(f"inverse_batch_size must be positive, got {batch_size}")

    matrix_dim = matrices.shape[-1]
    flat_source = matrices.reshape(-1, matrix_dim, matrix_dim)
    flat_inverse = torch.empty(
        flat_source.shape,
        dtype=torch.float32,
        device="cpu",
    )
    identity = torch.eye(matrix_dim, dtype=torch.float64, device="cpu")
    max_residual = 0.0

    for start in range(0, flat_source.shape[0], batch_size):
        end = min(start + batch_size, flat_source.shape[0])
        source64 = flat_source[start:end].to(torch.float64)
        inverse64, info = torch.linalg.inv_ex(source64, check_errors=False)

        bad = torch.nonzero(info != 0, as_tuple=False).flatten()
        if bad.numel() > 0:
            flat_index = start + int(bad[0])
            outer_index = tuple(
                torch.unravel_index(torch.tensor(flat_index), matrices.shape[:-2])
            )
            raise ValueError(
                f"Source {name!r} contains a singular transform at "
                f"index={outer_index} (inv_ex info={int(info[bad[0]])})"
            )

        residual = source64 @ inverse64 - identity
        max_residual = max(max_residual, float(residual.abs().amax()))
        flat_inverse[start:end].copy_(inverse64.to(torch.float32))

    return flat_inverse.reshape_as(matrices).contiguous(), max_residual


def _max_identity_residual(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    batch_size: int,
) -> float:
    """Return max(abs(left @ right - I)) using the stored tensor precision."""
    matrix_dim = left.shape[-1]
    flat_left = left.reshape(-1, matrix_dim, matrix_dim)
    flat_right = right.reshape(-1, matrix_dim, matrix_dim)
    identity = torch.eye(matrix_dim, dtype=torch.float32, device="cpu")
    max_residual = 0.0

    for start in range(0, flat_left.shape[0], batch_size):
        end = min(start + batch_size, flat_left.shape[0])
        residual = (
            flat_left[start:end].to(torch.float32)
            @ flat_right[start:end].to(torch.float32)
            - identity
        )
        max_residual = max(max_residual, float(residual.abs().amax()))

    return max_residual


def convert_checkpoint(
    input_path: Path,
    output_path: Path,
    *,
    output_dtype: torch.dtype,
    inverse_batch_size: int,
    overwrite: bool,
) -> dict[str, Any]:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )

    source = torch.load(input_path, map_location="cpu", weights_only=True)
    if not isinstance(source, dict):
        raise TypeError(
            f"Expected {input_path} to contain a dict, got {type(source).__name__}"
        )
    missing = {"k", "v"} - source.keys()
    if missing:
        raise ValueError(f"Source checkpoint is missing keys: {sorted(missing)}")

    transform_k = _validate_source_tensor("k", source["k"])
    transform_v = _validate_source_tensor("v", source["v"])

    if transform_k.dtype != transform_v.dtype:
        raise ValueError(
            f"K/V transform dtypes differ: K={transform_k.dtype}, V={transform_v.dtype}"
        )
    if transform_k.shape[:2] != transform_v.shape[:2]:
        raise ValueError(
            "K and V transforms must have the same [layers, kv_heads] shape, "
            f"got K={tuple(transform_k.shape[:2])}, "
            f"V={tuple(transform_v.shape[:2])}"
        )

    transform_k_inv, k_inverse_residual = _invert_in_batches(
        transform_k,
        batch_size=inverse_batch_size,
        name="k",
    )
    transform_v_inv, v_inverse_residual = _invert_in_batches(
        transform_v,
        batch_size=inverse_batch_size,
        name="v",
    )

    # Materialize contiguous right-multiplication matrices.  Keeping the block
    # axis makes the format work for both the current one-block-per-head export
    # and future sub-head block transforms.
    k_right = transform_k.transpose(-1, -2).contiguous().to(output_dtype)
    q_right = transform_k_inv.contiguous().to(output_dtype)
    v_right = transform_v.transpose(-1, -2).contiguous().to(output_dtype)
    o_right = transform_v_inv.transpose(-1, -2).contiguous().to(output_dtype)

    # These products are the exact invariants used by attention in row-vector
    # convention: q_right @ k_right.T == I and v_right @ o_right == I.
    stored_k_residual = _max_identity_residual(
        q_right,
        k_right.transpose(-1, -2),
        batch_size=inverse_batch_size,
    )
    stored_v_residual = _max_identity_residual(
        v_right,
        o_right,
        batch_size=inverse_batch_size,
    )

    output: dict[str, Any] = {
        "format": FORMAT_NAME,
        "version": 1,
        "num_layers": int(transform_k.shape[0]),
        "num_kv_heads": int(transform_k.shape[1]),
        "k_num_blocks": int(transform_k.shape[2]),
        "k_block_size": int(transform_k.shape[-1]),
        "v_num_blocks": int(transform_v.shape[2]),
        "v_block_size": int(transform_v.shape[-1]),
        "source_dtype": str(transform_k.dtype).removeprefix("torch."),
        "runtime_dtype": str(output_dtype).removeprefix("torch."),
        "k_inverse_fp64_max_residual": k_inverse_residual,
        "v_inverse_fp64_max_residual": v_inverse_residual,
        "k_stored_max_residual": stored_k_residual,
        "v_stored_max_residual": stored_v_residual,
        "q_right": q_right,
        "k_right": k_right,
        "v_right": v_right,
        "o_right": o_right,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        torch.save(output, temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert WUSH-KV transforms for the SGLang runtime."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="WUSH-KV .pt file containing the 'k' and 'v' tensors.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination for the versioned SGLang runtime checkpoint.",
    )
    parser.add_argument(
        "--output-dtype",
        choices=sorted(_DTYPES),
        default="float32",
        help="Stored runtime matrix dtype (default: float32).",
    )
    parser.add_argument(
        "--inverse-batch-size",
        type=int,
        default=64,
        help="Maximum number of transform blocks inverted together (default: 64).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    output = convert_checkpoint(
        args.input,
        args.output,
        output_dtype=_DTYPES[args.output_dtype],
        inverse_batch_size=args.inverse_batch_size,
        overwrite=args.overwrite,
    )

    logging.info("Saved %s checkpoint to %s", FORMAT_NAME, args.output)
    logging.info(
        "Shape: layers=%d kv_heads=%d K=(%d blocks x %d) V=(%d blocks x %d)",
        output["num_layers"],
        output["num_kv_heads"],
        output["k_num_blocks"],
        output["k_block_size"],
        output["v_num_blocks"],
        output["v_block_size"],
    )
    logging.info(
        "Residuals after storage cast: K=%.3e V=%.3e",
        output["k_stored_max_residual"],
        output["v_stored_max_residual"],
    )


if __name__ == "__main__":
    main()
