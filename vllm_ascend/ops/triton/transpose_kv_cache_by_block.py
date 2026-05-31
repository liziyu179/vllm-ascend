# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

# Keep the per-program vector small enough for Ascend vector-core UB usage.
TILE_ELEMS = 1024
MAX_PROGRAMS = 40


@triton.jit
def _transpose_to_workspace_kernel(
    cache,
    block_ids,
    workspace,
    block_stride: tl.constexpr,
    block_size: tl.constexpr,
    head_num: tl.constexpr,
    heads_per_split: tl.constexpr,
    head_dim: tl.constexpr,
    dim_tile_elems: tl.constexpr,
    dim_tile_count: tl.constexpr,
    token_head_count: tl.constexpr,
    total_tasks: tl.constexpr,
):
    """
    Stage 1: read selected cache blocks as
        [split_num, block_size, head_num / split_num, head_dim]
    and write workspace as
        [block_size, head_num, head_dim].

    One task copies one contiguous dim tile for one (token, head) pair in one
    selected cache block. This keeps both source and destination accesses
    contiguous inside each vector load/store.
    """
    task_id = tl.program_id(0)
    task_stride = tl.num_programs(0)

    while task_id < total_tasks:
        dim_tile_idx = task_id % dim_tile_count
        token_head_block_idx = task_id // dim_tile_count
        token_head_idx = token_head_block_idx % token_head_count
        selected_block_idx = token_head_block_idx // token_head_count

        dim_offsets = dim_tile_idx * dim_tile_elems + tl.arange(0, dim_tile_elems)
        dim_mask = dim_offsets < head_dim
        safe_dim_offsets = tl.minimum(dim_offsets, head_dim - 1)

        token_idx = token_head_idx // head_num
        head_idx = token_head_idx - token_idx * head_num
        split_idx = head_idx // heads_per_split
        head_idx_in_split = head_idx - split_idx * heads_per_split

        cache_block_id = tl.load(block_ids + selected_block_idx)

        src_offsets = (
            cache_block_id * block_stride
            + split_idx * block_size * heads_per_split * head_dim
            + token_idx * heads_per_split * head_dim
            + head_idx_in_split * head_dim
            + safe_dim_offsets
        )
        dst_offsets = (
            selected_block_idx * block_stride + token_idx * head_num * head_dim + head_idx * head_dim + safe_dim_offsets
        )

        values = tl.load(cache + src_offsets, mask=dim_mask, other=0.0)
        tl.store(workspace + dst_offsets, values, mask=dim_mask)

        task_id += task_stride


@triton.jit
def _copy_workspace_back_kernel(
    cache,
    block_ids,
    workspace,
    block_stride: tl.constexpr,
    elems_per_block: tl.constexpr,
    tile_elems: tl.constexpr,
    tile_count: tl.constexpr,
    total_tasks: tl.constexpr,
):
    """
    Stage 2: copy already-transposed workspace blocks back to the selected
    cache blocks. Kernel launch ordering provides the global sync between
    stage 1 reads and stage 2 overwrites.
    """
    task_id = tl.program_id(0)
    task_stride = tl.num_programs(0)

    while task_id < total_tasks:
        tile_idx = task_id % tile_count
        selected_block_idx = task_id // tile_count

        offsets = tile_idx * tile_elems + tl.arange(0, tile_elems)
        mask = offsets < elems_per_block
        safe_offsets = tl.minimum(offsets, elems_per_block - 1)

        cache_block_id = tl.load(block_ids + selected_block_idx)
        src_offsets = selected_block_idx * block_stride + safe_offsets
        dst_offsets = cache_block_id * block_stride + safe_offsets

        values = tl.load(workspace + src_offsets, mask=mask, other=0.0)
        tl.store(cache + dst_offsets, values, mask=mask)

        task_id += task_stride


def _check_cache(
    cache: torch.Tensor,
    block_ids: torch.Tensor,
    block_size: int,
    split_num: int,
) -> tuple[int, int, int, int]:
    assert cache.is_contiguous(), "transpose_kv_cache_by_block_triton requires contiguous KV cache tensors"
    assert cache.dim() == 4, f"expected cache shape [num_blocks, block_size, head_num, head_dim], got {cache.shape}"

    head_num = cache.shape[2]
    head_dim = cache.shape[3]
    assert head_num % split_num == 0, f"head_num={head_num} must be divisible by split_num={split_num}"
    assert block_size == cache.shape[1], f"block_size={block_size} does not match cache.shape[1]={cache.shape[1]}"
    assert block_ids.device == cache.device

    heads_per_split = head_num // split_num
    elems_per_block = block_size * head_num * head_dim
    return head_num, heads_per_split, head_dim, elems_per_block


def _run_for_cache(
    cache: torch.Tensor,
    block_ids: torch.Tensor,
    block_size: int,
    split_num: int,
) -> None:
    head_num, heads_per_split, head_dim, elems_per_block = _check_cache(cache, block_ids, block_size, split_num)

    block_stride = cache.stride(0)
    dim_tile_elems = min(head_dim, TILE_ELEMS)
    dim_tile_count = triton.cdiv(head_dim, dim_tile_elems)
    token_head_count = block_size * head_num
    selected_block_count = block_ids.numel()

    workspace = torch.empty(
        (selected_block_count, block_stride),
        dtype=cache.dtype,
        device=cache.device,
    )

    transpose_tasks = selected_block_count * token_head_count * dim_tile_count
    transpose_grid = (min(transpose_tasks, MAX_PROGRAMS),)
    _transpose_to_workspace_kernel[transpose_grid](
        cache,
        block_ids,
        workspace,
        block_stride,
        block_size,
        head_num,
        heads_per_split,
        head_dim,
        dim_tile_elems,
        dim_tile_count,
        token_head_count,
        transpose_tasks,
    )

    copy_tile_count = triton.cdiv(elems_per_block, TILE_ELEMS)
    copy_tasks = selected_block_count * copy_tile_count
    copy_grid = (min(copy_tasks, MAX_PROGRAMS),)
    _copy_workspace_back_kernel[copy_grid](
        cache,
        block_ids,
        workspace,
        block_stride,
        elems_per_block,
        TILE_ELEMS,
        copy_tile_count,
        copy_tasks,
    )


@torch.no_grad()
def transpose_kv_cache_by_block_triton(
    k_caches: list[torch.Tensor],
    v_caches: list[torch.Tensor],
    block_ids: torch.Tensor,
    block_size: int,
    split_num: int,
    block_group_elems: int = 8,
) -> None:
    # Keep the public signature compatible with the previous implementation.
    del block_group_elems

    if not HAS_TRITON:
        raise RuntimeError("Triton is not available")
    if split_num == 1 or block_ids.numel() == 0:
        return

    if block_ids.dtype != torch.int32:
        block_ids = block_ids.to(dtype=torch.int32)
    if not block_ids.is_contiguous():
        block_ids = block_ids.contiguous()

    for k_cache, v_cache in zip(k_caches, v_caches):
        _run_for_cache(k_cache, block_ids, block_size, split_num)
        _run_for_cache(v_cache, block_ids, block_size, split_num)
