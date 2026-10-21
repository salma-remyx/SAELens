"""Lossless bitmask compression for sparse activation caches.

Adapted from BitSparse (https://arxiv.org/abs/2610.22256), which stores
sparse ReLU-family activations as a packed nonzero bitmask alongside the
nonzero values so that the dense tensor can be recovered exactly. The
paper targets GPU-resident autograd buffers with fused Triton kernels;
SAELens activation caches live on disk as Huggingface datasets, so this
module implements the same storage format with plain tensor ops at the
cache write/read boundary instead.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

SPARSE_ACTIVATION_FORMAT = "sparse_bitmask"
DENSE_ACTIVATION_FORMAT = "dense"
ACTIVATION_STORAGE_FORMATS = (DENSE_ACTIVATION_FORMAT, SPARSE_ACTIVATION_FORMAT)

BITMASK_COLUMN_SUFFIX = "_bitmask"
VALUES_COLUMN_SUFFIX = "_values"

_BIT_WEIGHTS = (1, 2, 4, 8, 16, 32, 64, 128)


def bitmask_column(hook_name: str) -> str:
    """Name of the packed-bitmask dataset column for a hook."""
    return f"{hook_name}{BITMASK_COLUMN_SUFFIX}"


def values_column(hook_name: str) -> str:
    """Name of the nonzero-values dataset column for a hook."""
    return f"{hook_name}{VALUES_COLUMN_SUFFIX}"


def is_sparse_activation_cache(column_names: list[str], hook_name: str) -> bool:
    """Whether a cached activation dataset stores this hook in sparse format."""
    return (
        bitmask_column(hook_name) in column_names
        and values_column(hook_name) in column_names
    )


def bitmask_num_bytes(num_activations: int) -> int:
    """Number of uint8 bytes needed to bitmap `num_activations` elements."""
    return (num_activations + 7) // 8


def pack_nonzero_mask(mask: torch.Tensor) -> torch.Tensor:
    """Pack a boolean mask into uint8 bytes, one bit per element.

    Element `i` of the flattened mask lands in bit `i % 8` of byte `i // 8`.
    """
    bits = mask.reshape(-1).to(torch.uint8)
    padding = torch.zeros(
        bitmask_num_bytes(bits.numel()) * 8 - bits.numel(), dtype=torch.uint8
    )
    bits = torch.cat([bits, padding]).view(-1, 8).to(torch.int64)
    weights = torch.tensor(_BIT_WEIGHTS, dtype=torch.int64)
    return (bits * weights).sum(dim=1).to(torch.uint8)


def unpack_nonzero_mask(bitmask: torch.Tensor, num_activations: int) -> torch.Tensor:
    """Inverse of `pack_nonzero_mask`, returning the first `num_activations` bits."""
    if bitmask.numel() != bitmask_num_bytes(num_activations):
        raise ValueError(
            f"Bitmask has {bitmask.numel()} bytes, expected "
            f"{bitmask_num_bytes(num_activations)} for {num_activations} activations."
        )
    shifts = torch.arange(8, dtype=torch.int64)
    bits = (bitmask.to(torch.int64).unsqueeze(1) >> shifts) & 1
    return bits.reshape(-1)[:num_activations].to(torch.bool)


def encode_sparse_rows(
    activations: torch.Tensor,
) -> tuple[list[list[int]], list[list[float]]]:
    """Encode activations of shape (num_rows, context_size, d_in) row by row.

    Returns per-row packed bitmasks and the nonzero values of each row in
    row-major order, ready to be written as dataset columns.
    """
    bitmask_rows: list[list[int]] = []
    values_rows: list[list[float]] = []
    for row in activations.reshape(activations.shape[0], -1):
        mask = row != 0
        bitmask_rows.append(pack_nonzero_mask(mask).tolist())
        values_rows.append(row[mask].tolist())
    return bitmask_rows, values_rows


def decode_sparse_rows(
    bitmask_rows: Iterable[object],
    values_rows: Iterable[object],
    context_size: int,
    d_in: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Decode sparse rows back into activations of shape (num_rows, context_size, d_in).

    Each row is scattered into a zero tensor at the positions marked in its
    bitmask, so the round trip is lossless by construction.
    """
    num_activations = context_size * d_in
    rows = []
    for bitmask_row, values_row in zip(bitmask_rows, values_rows, strict=True):
        bitmask = torch.as_tensor(list(bitmask_row), dtype=torch.uint8)
        values = torch.as_tensor(list(values_row), dtype=dtype)
        mask = unpack_nonzero_mask(bitmask, num_activations)
        if int(mask.sum()) != values.numel():
            raise ValueError(
                f"Bitmask marks {int(mask.sum())} nonzero activations but "
                f"{values.numel()} values were stored."
            )
        decoded = torch.zeros(num_activations, dtype=dtype)
        decoded[mask] = values
        rows.append(decoded)
    return torch.stack(rows).reshape(-1, context_size, d_in)
