import subprocess
import sys
from pathlib import Path

import pytest
import torch

from sglang.srt.mem_cache.wush_transforms import (
    WUSH_RUNTIME_FORMAT,
    apply_wush_gqa_transform,
    apply_wush_kv_transform,
    load_wush_runtime_transforms,
)


def _invertible_matrices(heads: int, dim: int) -> torch.Tensor:
    torch.manual_seed(42)
    value = torch.randn(heads, 1, dim, dim, dtype=torch.float64)
    value += 2.0 * torch.eye(dim, dtype=torch.float64)
    return value


def test_wush_gqa_transform_preserves_attention_products() -> None:
    tokens_q, tokens_k = 5, 7
    kv_heads, queries_per_kv, dim = 2, 3, 4
    query_heads = kv_heads * queries_per_kv

    transform_k = _invertible_matrices(kv_heads, dim)
    transform_v = _invertible_matrices(kv_heads, dim) * 1.25
    k_right = transform_k.transpose(-1, -2).contiguous()
    q_right = torch.linalg.inv(transform_k)
    v_right = transform_v.transpose(-1, -2).contiguous()
    o_right = torch.linalg.inv(transform_v).transpose(-1, -2).contiguous()

    query = torch.randn(tokens_q, query_heads, dim, dtype=torch.float64)
    key = torch.randn(tokens_k, kv_heads, dim, dtype=torch.float64)
    key_per_query_head = key.repeat_interleave(queries_per_kv, dim=1)
    reference_logits = torch.einsum("thd,shd->ths", query, key_per_query_head)

    query_t = apply_wush_gqa_transform(query, q_right)
    key_t = apply_wush_kv_transform(key, k_right)
    transformed_logits = torch.einsum(
        "thd,shd->ths",
        query_t,
        key_t.repeat_interleave(queries_per_kv, dim=1),
    )
    torch.testing.assert_close(transformed_logits, reference_logits)

    output = torch.randn(tokens_q, query_heads, dim, dtype=torch.float64)
    output_t = apply_wush_gqa_transform(output, v_right)
    output_restored = apply_wush_gqa_transform(output_t, o_right)
    torch.testing.assert_close(output_restored, output)


def test_convert_wush_exporter_checkpoint(tmp_path: Path) -> None:
    source_path = tmp_path / "WUSH.pt"
    output_path = tmp_path / "WUSH-sglang.pt"
    transform_k = _invertible_matrices(2, 4).unsqueeze(0).to(torch.bfloat16)
    transform_v = (_invertible_matrices(2, 4) * 1.25).unsqueeze(0).to(torch.bfloat16)
    torch.save({"k": transform_k, "v": transform_v}, source_path)

    repo_root = Path(__file__).resolve().parents[4]
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "convert_wush_transforms.py"),
            "--input",
            str(source_path),
            "--output",
            str(output_path),
            "--inverse-batch-size",
            "2",
        ],
        check=True,
    )
    converted = torch.load(output_path, map_location="cpu", weights_only=True)

    assert converted["format"] == WUSH_RUNTIME_FORMAT
    torch.testing.assert_close(
        converted["k_right"], transform_k.float().transpose(-1, -2)
    )
    torch.testing.assert_close(
        converted["v_right"], transform_v.float().transpose(-1, -2)
    )
    identity_k = converted["q_right"] @ converted["k_right"].transpose(-1, -2)
    identity_v = converted["v_right"] @ converted["o_right"]
    torch.testing.assert_close(identity_k, torch.eye(4).expand_as(identity_k))
    torch.testing.assert_close(identity_v, torch.eye(4).expand_as(identity_v))


def _write_runtime_checkpoint(path: Path, *, layers: int, heads: int, dim: int) -> None:
    base = torch.eye(dim).reshape(1, 1, 1, dim, dim).repeat(layers, heads, 1, 1, 1)
    for layer in range(layers):
        for head in range(heads):
            base[layer, head].mul_(10 * layer + head + 1)
    checkpoint = {
        "format": WUSH_RUNTIME_FORMAT,
        "version": 1,
        "num_layers": layers,
        "num_kv_heads": heads,
        "q_right": base.clone(),
        "k_right": base.clone(),
        "v_right": base.clone(),
        "o_right": base.clone(),
    }
    torch.save(checkpoint, path)


@pytest.mark.parametrize(
    ("total_heads", "tp_size", "tp_rank", "expected_heads"),
    [
        (4, 2, 1, [13.0, 14.0]),
        (2, 4, 2, [12.0]),
    ],
)
def test_wush_loader_selects_attention_tp_heads(
    tmp_path: Path,
    total_heads: int,
    tp_size: int,
    tp_rank: int,
    expected_heads: list[float],
) -> None:
    path = tmp_path / "wush.pt"
    _write_runtime_checkpoint(path, layers=3, heads=total_heads, dim=4)

    transforms = load_wush_runtime_transforms(
        str(path),
        start_layer=1,
        layer_num=2,
        total_num_kv_heads=total_heads,
        local_num_kv_heads=len(expected_heads),
        k_head_dim=4,
        v_head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        attn_tp_size=tp_size,
        attn_tp_rank=tp_rank,
    )

    assert transforms.q_right.shape == (2, len(expected_heads), 1, 4, 4)
    actual = transforms.q_right[0, :, 0, 0, 0]
    torch.testing.assert_close(actual, torch.tensor(expected_heads))
