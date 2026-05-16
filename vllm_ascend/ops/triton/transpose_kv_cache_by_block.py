# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton


@triton.jit
def _transpose_kv_cache_by_block_kernel(
    cache_ptrs,
    block_ids,
    block_num_stride: tl.constexpr,
    block_size: tl.constexpr,
    head_num: tl.constexpr,
    head_dim: tl.constexpr,
    split_num: tl.constexpr,
    group_elems: tl.constexpr,
    element_size: tl.constexpr,
    block_group_elems: tl.constexpr,
    group_count: tl.constexpr,
    cache_count: tl.constexpr,
    total_tasks: tl.constexpr,
    logical_block_elems: tl.constexpr,
    block_elems: tl.constexpr,
):
    offsets = tl.arange(0, block_elems)
    elem_offsets = offsets // element_size
    byte_offsets = offsets - elem_offsets * element_size

    task_id = tl.program_id(0)
    task_stride = tl.num_programs(0)
    while task_id < total_tasks:
        cache_pid = task_id % cache_count
        group_block_id = task_id // cache_count
        group_pid = group_block_id % group_count
        block_pid = group_block_id // group_count

        cache = tl.load(cache_ptrs + cache_pid)
        cache = cache.to(tl.pointer_type(tl.uint8))
        block_id = tl.load(block_ids + block_pid)
        group_base = group_pid * block_group_elems

        group_offsets = group_base + elem_offsets % block_group_elems
        token_offsets = (elem_offsets // block_group_elems) % block_size
        split_offsets = elem_offsets // (block_group_elems * block_size)

        mask = (offsets < logical_block_elems) & (split_offsets < split_num) & (group_offsets < group_elems)
        block_base = block_id * block_num_stride

        src_offsets = (
            block_base + split_offsets * block_size * group_elems + token_offsets * group_elems + group_offsets
        )
        dst_offsets = block_base + token_offsets * head_num * head_dim + split_offsets * group_elems + group_offsets

        values = tl.load(cache + src_offsets * element_size + byte_offsets, mask=mask)
        tl.store(cache + dst_offsets * element_size + byte_offsets, values, mask=mask)

        task_id += task_stride


def _check_cache(
    cache: torch.Tensor,
    block_ids: torch.Tensor,
    block_size: int,
    split_num: int,
) -> tuple[int, int, int]:
    assert cache.is_contiguous(), "transpose_kv_cache_by_block_triton requires contiguous KV cache tensors"
    assert cache.dim() == 4, f"expected cache shape [num_blocks, block_size, head_num, head_dim], got {cache.shape}"

    head_num = cache.shape[2]
    head_dim = cache.shape[3]
    assert head_num % split_num == 0, f"head_num={head_num} must be divisible by split_num={split_num}"
    assert block_size == cache.shape[1], f"block_size={block_size} does not match cache.shape[1]={cache.shape[1]}"
    assert block_ids.device == cache.device

    group_elems = head_num * head_dim // split_num
    return head_num, head_dim, group_elems


def _run_for_caches(
    caches: list[torch.Tensor],
    block_ids: torch.Tensor,
    block_size: int,
    split_num: int,
    block_group_elems: int,
) -> None:
    head_num, head_dim, group_elems = _check_cache(caches[0], block_ids, block_size, split_num)
    block_num_stride = caches[0].stride(0)
    element_size = caches[0].element_size()

    for cache in caches[1:]:
        cache_head_num, cache_head_dim, cache_group_elems = _check_cache(cache, block_ids, block_size, split_num)
        assert cache_head_num == head_num
        assert cache_head_dim == head_dim
        assert cache_group_elems == group_elems
        assert cache.stride(0) == block_num_stride
        assert cache.element_size() == element_size

    cache_ptrs = torch.tensor([cache.data_ptr() for cache in caches], dtype=torch.int64, device=block_ids.device)
    block_group_elems = min(block_group_elems, group_elems)
    group_count = triton.cdiv(group_elems, block_group_elems)
    cache_count = len(caches)
    total_tasks = block_ids.numel() * group_count * cache_count
    logical_block_elems = split_num * block_size * block_group_elems * element_size
    block_elems = triton.next_power_of_2(logical_block_elems)
    grid = (min(total_tasks, 40),)

    _transpose_kv_cache_by_block_kernel[grid](
        cache_ptrs,
        block_ids,
        block_num_stride,
        block_size,
        head_num,
        head_dim,
        split_num,
        group_elems,
        element_size,
        block_group_elems,
        group_count,
        cache_count,
        total_tasks,
        logical_block_elems,
        block_elems,
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
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available")
    if split_num == 1 or block_ids.numel() == 0:
        return

    if block_ids.dtype != torch.int64:
        block_ids = block_ids.to(dtype=torch.int64)
    if not block_ids.is_contiguous():
        block_ids = block_ids.contiguous()

    caches = []
    for k_cache, v_cache in zip(k_caches, v_caches):
        caches.append(k_cache)
        caches.append(v_cache)
    _run_for_caches(caches, block_ids, block_size, split_num, block_group_elems)
