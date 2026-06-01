# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

# Keep the per-program tensor bounded for Ascend vector-core UB usage. The
# no-workspace implementation needs one program to own a full cache block to
# avoid in-place overwrite hazards across programs.
MAX_PROGRAM_BYTES = 192 * 1024
MAX_PROGRAMS = 40


@triton.jit
def _transpose_block_inplace_kernel(
    cache_ptrs,
    block_ids,
    block_stride: tl.constexpr,
    block_size: tl.constexpr,
    split_num: tl.constexpr,
    group_elems: tl.constexpr,
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
        cache_block_id = tl.load(block_ids + selected_block_idx).to(tl.int64)
        block_base = cache_block_id * block_stride

        src_desc = tl.make_tensor_descriptor(
            cache + block_base,
            shape=[block_size, split_num, group_elems],
            strides=[group_elems, block_size * group_elems, 1],
            block_shape=[block_size, split_num, group_elems],
        )
        dst_desc = tl.make_tensor_descriptor(
            cache + block_base,
            shape=[block_size, split_num, group_elems],
            strides=[split_num * group_elems, group_elems, 1],
            block_shape=[block_size, split_num, group_elems],
        )

        values = src_desc.load([0, 0, 0])
        dst_desc.store([0, 0, 0], values)

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

    if elems_per_block * caches[0].element_size() > MAX_PROGRAM_BYTES:
        raise ValueError(
            "transpose_kv_cache_by_block_triton no-workspace path requires "
            f"one full cache block <= {MAX_PROGRAM_BYTES} bytes, got "
            f"{elems_per_block * caches[0].element_size()} bytes. Splitting a block across programs would need "
            "a workspace or a global barrier to avoid in-place overwrite."
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
    group_elems = heads_per_split * head_dim
    cache_ptrs = torch.tensor([cache.data_ptr() for cache in caches], dtype=torch.int64, device=caches[0].device)

    grid = (min(selected_block_count, MAX_PROGRAMS), len(caches))
    _transpose_block_inplace_kernel[grid](
        cache_ptrs,
        block_ids,
        block_stride,
        block_size,
        split_num,
        group_elems,
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
