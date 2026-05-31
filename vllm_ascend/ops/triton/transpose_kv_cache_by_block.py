# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

# Keep the per-program vector bounded for Ascend vector-core UB usage. The
# no-workspace implementation needs one program to own a full cache block to
# avoid in-place overwrite hazards across programs.
MAX_FULL_BLOCK_ELEMS = 131072
MAX_PROGRAMS = 40


@triton.jit
def _transpose_block_inplace_kernel(
    cache_ptrs,
    block_ids,
    block_stride: tl.constexpr,
    elems_per_block: tl.constexpr,
    block_size: tl.constexpr,
    head_num: tl.constexpr,
    heads_per_split: tl.constexpr,
    head_dim: tl.constexpr,
    vector_elems: tl.constexpr,
    selected_block_count: tl.constexpr,
):
    """
    Read one selected cache block as
        [split_num, block_size, head_num / split_num, head_dim]
    and write it back in-place as
        [block_size, head_num, head_dim].

    One program owns a full cache block. This keeps all reads for the block in
    the same program before the in-place stores, avoiding the cross-program
    overwrite hazard that would require a workspace or a global barrier.
    """
    selected_block_idx = tl.program_id(0)
    selected_block_stride = tl.num_programs(0)
    cache_idx = tl.program_id(1)

    # Copy raw 16-bit elements so the same pointer table works for fp16/bf16.
    cache = tl.load(cache_ptrs + cache_idx).to(tl.pointer_type(tl.uint16))

    while selected_block_idx < selected_block_count:
        offsets = tl.arange(0, vector_elems)
        mask = offsets < elems_per_block
        safe_offsets = tl.minimum(offsets, elems_per_block - 1)

        dim_idx = safe_offsets % head_dim
        head_token_idx = safe_offsets // head_dim
        head_idx = head_token_idx % head_num
        token_idx = head_token_idx // head_num
        split_idx = head_idx // heads_per_split
        head_idx_in_split = head_idx - split_idx * heads_per_split

        src_offsets = (
            split_idx * block_size * heads_per_split * head_dim
            + token_idx * heads_per_split * head_dim
            + head_idx_in_split * head_dim
            + dim_idx
        )

        cache_block_id = tl.load(block_ids + selected_block_idx).to(tl.int64)
        block_base = cache_block_id * block_stride

        values = tl.load(cache + block_base + src_offsets, mask=mask, other=0)
        tl.store(cache + block_base + safe_offsets, values, mask=mask)

        selected_block_idx += selected_block_stride


def _check_cache(
    cache: torch.Tensor,
    block_ids: torch.Tensor,
    block_size: int,
    split_num: int,
) -> tuple[int, int, int, int]:
    assert cache.is_contiguous(), "transpose_kv_cache_by_block_triton requires contiguous KV cache tensors"
    assert cache.dim() == 4, f"expected cache shape [num_blocks, block_size, head_num, head_dim], got {cache.shape}"
    assert cache.element_size() == 2, f"expected 2-byte cache dtype, got {cache.dtype}"

    head_num = cache.shape[2]
    head_dim = cache.shape[3]
    assert head_num % split_num == 0, f"head_num={head_num} must be divisible by split_num={split_num}"
    assert block_size == cache.shape[1], f"block_size={block_size} does not match cache.shape[1]={cache.shape[1]}"
    assert block_ids.device == cache.device
    assert block_ids.dtype in (torch.int32, torch.int64), f"block_ids must be int32 or int64, got {block_ids.dtype}"

    heads_per_split = head_num // split_num
    elems_per_block = block_size * head_num * head_dim
    return head_num, heads_per_split, head_dim, elems_per_block


def _check_caches(
    caches: list[torch.Tensor],
    block_ids: torch.Tensor,
    block_size: int,
    split_num: int,
) -> tuple[int, int, int, int]:
    assert caches, "transpose_kv_cache_by_block_triton requires at least one KV cache tensor"
    head_num, heads_per_split, head_dim, elems_per_block = _check_cache(caches[0], block_ids, block_size, split_num)

    expected_shape = caches[0].shape
    expected_dtype = caches[0].dtype
    expected_device = caches[0].device
    for cache in caches[1:]:
        cur_head_num, cur_heads_per_split, cur_head_dim, cur_elems_per_block = _check_cache(
            cache, block_ids, block_size, split_num
        )
        assert cache.shape == expected_shape, (
            f"all KV cache tensors must have the same shape, got {expected_shape} and {cache.shape}"
        )
        assert cache.dtype == expected_dtype, (
            f"all KV cache tensors must have the same dtype, got {expected_dtype} and {cache.dtype}"
        )
        assert cache.device == expected_device, (
            f"all KV cache tensors must be on the same device, got {expected_device} and {cache.device}"
        )
        assert (cur_head_num, cur_heads_per_split, cur_head_dim, cur_elems_per_block) == (
            head_num,
            heads_per_split,
            head_dim,
            elems_per_block,
        )

    if elems_per_block > MAX_FULL_BLOCK_ELEMS:
        raise ValueError(
            "transpose_kv_cache_by_block_triton no-workspace path requires "
            f"elems_per_block <= {MAX_FULL_BLOCK_ELEMS}, got {elems_per_block}. "
            "Splitting a block across programs would need a workspace or a global barrier to avoid in-place overwrite."
        )

    return head_num, heads_per_split, head_dim, elems_per_block


def _run_for_caches(
    caches: list[torch.Tensor],
    block_ids: torch.Tensor,
    block_size: int,
    split_num: int,
) -> None:
    head_num, heads_per_split, head_dim, elems_per_block = _check_caches(caches, block_ids, block_size, split_num)

    block_stride = caches[0].stride(0)
    selected_block_count = block_ids.numel()
    vector_elems = triton.next_power_of_2(elems_per_block)
    cache_ptrs = torch.tensor([cache.data_ptr() for cache in caches], dtype=torch.int64, device=caches[0].device)

    grid = (min(selected_block_count, MAX_PROGRAMS), len(caches))
    _transpose_block_inplace_kernel[grid](
        cache_ptrs,
        block_ids,
        block_stride,
        elems_per_block,
        block_size,
        head_num,
        heads_per_split,
        head_dim,
        vector_elems,
        selected_block_count,
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

    if not block_ids.is_contiguous():
        block_ids = block_ids.contiguous()

    assert len(k_caches) == len(v_caches), f"k/v cache layer counts differ: {len(k_caches)} vs {len(v_caches)}"
    caches: list[torch.Tensor] = []
    for k_cache, v_cache in zip(k_caches, v_caches):
        caches.append(k_cache)
        caches.append(v_cache)
    _run_for_caches(caches, block_ids, block_size, split_num)
