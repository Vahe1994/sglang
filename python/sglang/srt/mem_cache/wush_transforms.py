"""Runtime loading and application of WUSH-KV transforms.

The converter in ``scripts/convert_wush_transforms.py`` stores matrices in
SGLang's row-vector convention.  Checkpoints remain global; this module slices
them to the current pipeline stage and attention tensor-parallel rank.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

WUSH_RUNTIME_FORMAT = "sglang_wush_kv_runtime_v1"


@dataclass(frozen=True)
class WushRuntimeTransforms:
    """Local WUSH matrices, including an explicit transform-block axis."""

    q_right: torch.Tensor
    k_right: torch.Tensor
    v_right: torch.Tensor
    o_right: torch.Tensor


def _global_kv_head_indices(
    *, total_num_kv_heads: int, attn_tp_size: int, attn_tp_rank: int
) -> slice:
    if total_num_kv_heads <= 0:
        raise ValueError(
            f"total_num_kv_heads must be positive, got {total_num_kv_heads}"
        )
    if attn_tp_size <= 0:
        raise ValueError(f"attn_tp_size must be positive, got {attn_tp_size}")
    if not 0 <= attn_tp_rank < attn_tp_size:
        raise ValueError(
            f"attn_tp_rank must be in [0, {attn_tp_size}), got {attn_tp_rank}"
        )

    if total_num_kv_heads >= attn_tp_size:
        if total_num_kv_heads % attn_tp_size != 0:
            raise ValueError(
                f"num_kv_heads={total_num_kv_heads} is not divisible by "
                f"attn_tp_size={attn_tp_size}"
            )
        local_heads = total_num_kv_heads // attn_tp_size
        start = attn_tp_rank * local_heads
        return slice(start, start + local_heads)

    if attn_tp_size % total_num_kv_heads != 0:
        raise ValueError(
            f"attn_tp_size={attn_tp_size} is not divisible by replicated "
            f"num_kv_heads={total_num_kv_heads}"
        )
    replicas = attn_tp_size // total_num_kv_heads
    global_head = attn_tp_rank // replicas
    return slice(global_head, global_head + 1)


def _validate_matrix_tensor(
    checkpoint: dict,
    name: str,
    *,
    num_layers: int,
    num_kv_heads: int,
) -> torch.Tensor:
    value = checkpoint.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"WUSH checkpoint is missing tensor {name!r}")
    if value.ndim != 5:
        raise ValueError(
            f"WUSH tensor {name!r} must be [layers, kv_heads, blocks, d, d], "
            f"got {tuple(value.shape)}"
        )
    if value.shape[0] != num_layers or value.shape[1] != num_kv_heads:
        raise ValueError(
            f"WUSH tensor {name!r} has leading shape {tuple(value.shape[:2])}; "
            f"expected ({num_layers}, {num_kv_heads})"
        )
    if value.shape[-1] != value.shape[-2]:
        raise ValueError(
            f"WUSH tensor {name!r} has non-square blocks {tuple(value.shape[-2:])}"
        )
    if not value.is_floating_point():
        raise ValueError(f"WUSH tensor {name!r} must be floating point")
    return value


def load_wush_runtime_transforms(
    path: str,
    *,
    start_layer: int,
    layer_num: int,
    total_num_kv_heads: int,
    local_num_kv_heads: int,
    k_head_dim: int,
    v_head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    attn_tp_size: Optional[int] = None,
    attn_tp_rank: Optional[int] = None,
) -> WushRuntimeTransforms:
    """Load a converted checkpoint and select this worker's layers/heads."""
    if not path:
        raise ValueError(
            "SGLANG_WUSH_TRANSFORM_PATH must point to a converted WUSH checkpoint"
        )
    if attn_tp_size is None or attn_tp_rank is None:
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
        attn_tp_size = parallel.attn_tp_size
        attn_tp_rank = parallel.attn_tp_rank

    checkpoint = torch.load(
        Path(path), map_location="cpu", mmap=True, weights_only=True
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("WUSH checkpoint must contain a dictionary")
    if checkpoint.get("format") != WUSH_RUNTIME_FORMAT:
        raise ValueError(
            f"Unsupported WUSH checkpoint format {checkpoint.get('format')!r}; "
            f"expected {WUSH_RUNTIME_FORMAT!r}. Convert it with "
            "scripts/convert_wush_transforms.py."
        )
    if int(checkpoint.get("version", -1)) != 1:
        raise ValueError(
            f"Unsupported WUSH checkpoint version {checkpoint.get('version')!r}; "
            "expected 1"
        )

    num_layers = int(checkpoint.get("num_layers", -1))
    num_kv_heads = int(checkpoint.get("num_kv_heads", -1))
    if num_kv_heads != total_num_kv_heads:
        raise ValueError(
            f"WUSH checkpoint has {num_kv_heads} global KV heads, but the model "
            f"has {total_num_kv_heads}"
        )
    end_layer = start_layer + layer_num
    if start_layer < 0 or end_layer > num_layers:
        raise ValueError(
            f"Worker layer range [{start_layer}, {end_layer}) is outside WUSH "
            f"checkpoint range [0, {num_layers})"
        )

    matrices = {
        name: _validate_matrix_tensor(
            checkpoint,
            name,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
        )
        for name in ("q_right", "k_right", "v_right", "o_right")
    }

    if matrices["q_right"].shape[2:] != matrices["k_right"].shape[2:]:
        raise ValueError("WUSH Q and K transform shapes do not match")
    if matrices["v_right"].shape[2:] != matrices["o_right"].shape[2:]:
        raise ValueError("WUSH V and O transform shapes do not match")
    if matrices["k_right"].shape[2] * matrices["k_right"].shape[-1] != k_head_dim:
        raise ValueError(
            "WUSH K transform blocks do not cover the model head dimension: "
            f"{matrices['k_right'].shape[2]} * {matrices['k_right'].shape[-1]} "
            f"!= {k_head_dim}"
        )
    if matrices["v_right"].shape[2] * matrices["v_right"].shape[-1] != v_head_dim:
        raise ValueError(
            "WUSH V transform blocks do not cover the model value dimension: "
            f"{matrices['v_right'].shape[2]} * {matrices['v_right'].shape[-1]} "
            f"!= {v_head_dim}"
        )

    head_slice = _global_kv_head_indices(
        total_num_kv_heads=total_num_kv_heads,
        attn_tp_size=attn_tp_size,
        attn_tp_rank=attn_tp_rank,
    )

    def _select(name: str) -> torch.Tensor:
        selected = matrices[name][start_layer:end_layer, head_slice]
        if selected.shape[1] != local_num_kv_heads:
            raise ValueError(
                f"WUSH TP selection produced {selected.shape[1]} local KV heads, "
                f"but the pool has {local_num_kv_heads}"
            )
        if not torch.isfinite(selected).all():
            raise ValueError(f"WUSH tensor {name!r} contains NaN or infinity")
        return selected.to(device=device, dtype=dtype).contiguous()

    return WushRuntimeTransforms(
        q_right=_select("q_right"),
        k_right=_select("k_right"),
        v_right=_select("v_right"),
        o_right=_select("o_right"),
    )


def apply_wush_kv_transform(
    tensor: torch.Tensor,
    right: torch.Tensor,
    *,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Apply blockwise right matrices to ``[tokens, kv_heads, head_dim]``."""
    if tensor.ndim != 3 or right.ndim != 4:
        raise ValueError(
            f"Expected tensor [tokens, heads, dim] and right [heads, blocks, d, d], "
            f"got {tuple(tensor.shape)} and {tuple(right.shape)}"
        )
    tokens, heads, head_dim = tensor.shape
    if heads != right.shape[0] or head_dim != right.shape[1] * right.shape[-1]:
        raise ValueError(
            f"WUSH KV transform shape mismatch: tensor={tuple(tensor.shape)}, "
            f"right={tuple(right.shape)}"
        )
    block_size = right.shape[-1]
    transformed = torch.einsum(
        "thbi,hbij->thbj",
        tensor.to(right.dtype).reshape(tokens, heads, right.shape[1], block_size),
        right,
    ).reshape(tokens, heads, head_dim)
    target_dtype = tensor.dtype if output_dtype is None else output_dtype
    return transformed.to(target_dtype).contiguous()


def apply_wush_gqa_transform(
    tensor: torch.Tensor,
    right: torch.Tensor,
    *,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Apply each KV head's transform to its grouped query/output heads."""
    if tensor.ndim != 3 or right.ndim != 4:
        raise ValueError(
            f"Expected tensor [tokens, q_heads, dim] and right [kv_heads, blocks, d, d], "
            f"got {tuple(tensor.shape)} and {tuple(right.shape)}"
        )
    tokens, query_heads, head_dim = tensor.shape
    kv_heads, blocks, block_size, block_size_2 = right.shape
    if block_size != block_size_2 or head_dim != blocks * block_size:
        raise ValueError(
            f"WUSH GQA transform dimension mismatch: tensor={tuple(tensor.shape)}, "
            f"right={tuple(right.shape)}"
        )
    if query_heads % kv_heads != 0:
        raise ValueError(
            f"query_heads={query_heads} is not divisible by kv_heads={kv_heads}"
        )
    queries_per_kv = query_heads // kv_heads
    grouped = tensor.to(right.dtype).reshape(
        tokens, kv_heads, queries_per_kv, blocks, block_size
    )
    transformed = torch.einsum("thgbi,hbij->thgbj", grouped, right).reshape(
        tokens, query_heads, head_dim
    )
    target_dtype = tensor.dtype if output_dtype is None else output_dtype
    return transformed.to(target_dtype).contiguous()
