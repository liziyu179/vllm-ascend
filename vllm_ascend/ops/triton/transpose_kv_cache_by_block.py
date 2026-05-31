# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

# NPU vector core 的 UB 只有 192 KiB。这里固定使用 1024 个元素作为
# 单个 program 的搬运粒度，给 Triton lowering 和运行时临时开销留余量。
TILE_ELEMS = 1024
MAX_THREADS = 40


@triton.jit
def _transpose_to_workspace_kernel(
    cache,
    block_ids,
    workspace,
    block_num_stride: tl.constexpr,
    block_size: tl.constexpr,
    head_num: tl.constexpr,
    split_num: tl.constexpr,
    heads_per_split: tl.constexpr,
    head_dim: tl.constexpr,
    dim_block: tl.constexpr,
    dim_tile_count: tl.constexpr,
    token_head_count: tl.constexpr,
    total_tasks: tl.constexpr,
):
    """
    第一阶段：把一个 cache block 按最终布局搬到 workspace。

    源 block 逻辑布局为：
        [split_num, block_size, head_num / split_num, head_dim]

    workspace 中的目标布局为：
        [block_size, head_num, head_dim]

    这在算法思路上对齐 AscendC 版本：从源布局读取，按照目标布局写入临时缓冲，
    再把临时缓冲连续写回原 cache。

    Triton 不能像 AscendC 那样把每个 core 的 UB 当作 kernel 内的全局暂存区使用，
    单个 Triton kernel 内部也没有 program 之间的全局同步。如果直接做原地
    src -> dst transpose，可能覆盖其他 program 还没读取的源数据。因此这里用
    workspace 明确拆成两个阶段：

        阶段 1：所有 program 从 cache 读取，并把最终布局写到 workspace
        阶段 2：所有 program 再把 workspace 连续写回 cache

    offsets 按目标布局线性枚举。对每个目标 offset，先反推出
    (token, head, dim)，再拆出 (split, head_in_split)，到源布局中的
    (split, token, head_in_split, dim) 位置读取。
    """
    task_id = tl.program_id(0)
    task_stride = tl.num_programs(0)

    while task_id < total_tasks:
        dim_tile_id = task_id % dim_tile_count
        token_head_block_idx = task_id // dim_tile_count
        token_head_idx = token_head_block_idx % token_head_count
        block_idx = token_head_block_idx // token_head_count

        dim_offsets = dim_tile_id * dim_block + tl.arange(0, dim_block)
        mask = dim_offsets < head_dim
        safe_dim_offsets = tl.minimum(dim_offsets, head_dim - 1)

        block_id = tl.load(block_ids + block_idx)
        token_idx = token_head_idx // head_num
        head_idx = token_head_idx - token_idx * head_num
        split_idx = head_idx // heads_per_split
        head_idx_in_split = head_idx - split_idx * heads_per_split

        # 每个 task 只搬一个 (token, head) 的 head_dim 连续片段。
        # 这样源和目标都是连续访存，避免在 NPU 上生成大跨度 gather。
        src_offsets = (
            block_id * block_num_stride
            + split_idx * block_size * heads_per_split * head_dim
            + token_idx * heads_per_split * head_dim
            + head_idx_in_split * head_dim
            + safe_dim_offsets
        )
        workspace_offsets = (
            block_idx * block_num_stride + token_idx * head_num * head_dim + head_idx * head_dim + safe_dim_offsets
        )

        values = tl.load(cache + src_offsets, mask=mask, other=0.0)
        tl.store(workspace + workspace_offsets, values, mask=mask)

        task_id += task_stride


@triton.jit
def _copy_workspace_back_kernel(
    cache,
    block_ids,
    workspace,
    block_num_stride: tl.constexpr,
    total_elems: tl.constexpr,
    block_elems: tl.constexpr,
    tile_count: tl.constexpr,
    total_tasks: tl.constexpr,
):
    """
    第二阶段：把 workspace 中已经排好的最终布局写回原 cache block。

    这里故意拆成独立 kernel。两次 kernel launch 的先后顺序提供了 Triton 单 kernel
    内部没有的全局同步点；执行到本 kernel 时，所有源数据都已经在第一阶段读取完，
    因而可以安全覆盖原 cache。
    """
    task_id = tl.program_id(0)
    task_stride = tl.num_programs(0)

    while task_id < total_tasks:
        tile_id = task_id % tile_count
        block_idx = task_id // tile_count

        offsets = tile_id * block_elems + tl.arange(0, block_elems)
        mask = offsets < total_elems
        safe_offsets = tl.minimum(offsets, total_elems - 1)

        block_id = tl.load(block_ids + block_idx)
        cache_offsets = block_id * block_num_stride + safe_offsets
        workspace_offsets = block_idx * block_num_stride + safe_offsets

        values = tl.load(workspace + workspace_offsets, mask=mask, other=0.0)
        tl.store(cache + cache_offsets, values, mask=mask)

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
    total_elems = block_size * head_num * head_dim
    return head_num, heads_per_split, head_dim, total_elems


def _get_dim_block(head_dim: int) -> int:
    return min(head_dim, TILE_ELEMS)


def _run_for_cache(
    cache: torch.Tensor,
    block_ids: torch.Tensor,
    block_size: int,
    split_num: int,
) -> None:
    head_num, heads_per_split, head_dim, total_elems = _check_cache(cache, block_ids, block_size, split_num)
    block_num_stride = cache.stride(0)
    block_elems = TILE_ELEMS
    dim_block = _get_dim_block(head_dim)

    workspace = torch.empty(
        (block_ids.numel(), block_num_stride),
        dtype=cache.dtype,
        device=cache.device,
    )

    dim_tile_count = triton.cdiv(head_dim, dim_block)
    token_head_count = block_size * head_num
    transpose_tasks = block_ids.numel() * token_head_count * dim_tile_count
    transpose_grid = (min(transpose_tasks, MAX_THREADS),)
    _transpose_to_workspace_kernel[transpose_grid](
        cache,
        block_ids,
        workspace,
        block_num_stride,
        block_size,
        head_num,
        split_num,
        heads_per_split,
        head_dim,
        dim_block,
        dim_tile_count,
        token_head_count,
        transpose_tasks,
    )

    copy_tile_count = triton.cdiv(total_elems, block_elems)
    copy_tasks = block_ids.numel() * copy_tile_count
    copy_grid = (min(copy_tasks, MAX_THREADS),)
    _copy_workspace_back_kernel[copy_grid](
        cache,
        block_ids,
        workspace,
        block_num_stride,
        total_elems,
        block_elems,
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
    # 保持对外函数签名兼容旧实现。新实现根据 UB 容量自动决定 tile 大小，
    # 不再使用这个手动调节参数。
    del block_group_elems

    if not HAS_TRITON:
        raise RuntimeError("Triton is not available")
    if split_num == 1 or block_ids.numel() == 0:
        return

    if block_ids.dtype != torch.int32:
        block_ids = block_ids.to(dtype=torch.int32)
    if not block_ids.is_contiguous():
        block_ids = block_ids.contiguous()

    caches = []
    for k_cache, v_cache in zip(k_caches, v_caches):
        caches.append(k_cache)
        caches.append(v_cache)

    # K/V cache 是相互独立的 buffer。逐个处理可以让每个 cache 的临时
    # workspace 在下一次处理前释放，降低峰值显存占用。
    for cache in caches:
        _run_for_cache(cache, block_ids, block_size, split_num)
