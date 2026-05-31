# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

# 当前硬件约束：最多 40 个 program 并发参与搬运，并共享 192 KiB UB。
# 因此每个 program 的向量搬运粒度需要按该预算计算，尽可能提高 UB 利用率。
UB_SIZE_BYTES = 192 * 1024
MAX_THREADS = 40


@triton.jit
def _transpose_to_workspace_kernel(
    cache,
    block_ids,
    workspace,
    block_num_stride: tl.constexpr,
    block_size: tl.constexpr,
    split_num: tl.constexpr,
    heads_per_split: tl.constexpr,
    head_dim: tl.constexpr,
    total_elems: tl.constexpr,
    block_elems: tl.constexpr,
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
    tile_id = tl.program_id(0)
    block_idx = tl.program_id(1)

    offsets = tile_id * block_elems + tl.arange(0, block_elems)
    mask = offsets < total_elems

    block_id = tl.load(block_ids + block_idx)
    # 目标线性布局：
    #   dst[token, head, dim]
    #     = token * head_num * head_dim
    #       + head * head_dim
    #       + dim
    head_num = split_num * heads_per_split
    dst_head_cell = offsets // head_dim
    dim_idx = offsets - dst_head_cell * head_dim
    token_idx = dst_head_cell // head_num
    head_idx = dst_head_cell - token_idx * head_num
    split_idx = head_idx // heads_per_split
    head_idx_in_split = head_idx - split_idx * heads_per_split

    # 源线性布局：
    #   src[split, token, head_in_split, dim]
    #     = split * block_size * heads_per_split * head_dim
    #       + token * heads_per_split * head_dim
    #       + head_in_split * head_dim
    #       + dim
    src_offsets = (
        block_id * block_num_stride
        + split_idx * block_size * heads_per_split * head_dim
        + token_idx * heads_per_split * head_dim
        + head_idx_in_split * head_dim
        + dim_idx
    )
    workspace_offsets = block_idx * block_num_stride + offsets

    values = tl.load(cache + src_offsets, mask=mask)
    tl.store(workspace + workspace_offsets, values, mask=mask)


@triton.jit
def _copy_workspace_back_kernel(
    cache,
    block_ids,
    workspace,
    block_num_stride: tl.constexpr,
    total_elems: tl.constexpr,
    block_elems: tl.constexpr,
):
    """
    第二阶段：把 workspace 中已经排好的最终布局写回原 cache block。

    这里故意拆成独立 kernel。两次 kernel launch 的先后顺序提供了 Triton 单 kernel
    内部没有的全局同步点；执行到本 kernel 时，所有源数据都已经在第一阶段读取完，
    因而可以安全覆盖原 cache。
    """
    tile_id = tl.program_id(0)
    block_idx = tl.program_id(1)

    offsets = tile_id * block_elems + tl.arange(0, block_elems)
    mask = offsets < total_elems

    block_id = tl.load(block_ids + block_idx)
    cache_offsets = block_id * block_num_stride + offsets
    workspace_offsets = block_idx * block_num_stride + offsets

    values = tl.load(workspace + workspace_offsets, mask=mask)
    tl.store(cache + cache_offsets, values, mask=mask)


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

    heads_per_split = head_num // split_num
    total_elems = block_size * head_num * head_dim
    return heads_per_split, head_dim, total_elems


def _previous_power_of_2(value: int) -> int:
    return 1 << (value.bit_length() - 1)


def _get_block_elems(element_size: int) -> int:
    # 硬件预算是 192 KiB UB，最多 40 个 program 并发共享，因此每个 program
    # 约可使用 UB_SIZE_BYTES / MAX_THREADS 字节。
    #
    # tl.arange 需要编译期常量，并且通常更适合 2 的幂大小。这里取不超过
    # 单 program 字节预算的最大 2 次幂。对 fp16/bf16 来说，block_elems 为
    # 2048，即每个 program 4096 字节；40 个 program 总共约 160 KiB，
    # 给编译器和运行时开销留出一定余量。
    per_program_bytes = max(element_size, UB_SIZE_BYTES // MAX_THREADS)
    return max(1, _previous_power_of_2(per_program_bytes // element_size))


def _run_for_cache(
    cache: torch.Tensor,
    block_ids: torch.Tensor,
    block_size: int,
    split_num: int,
) -> None:
    heads_per_split, head_dim, total_elems = _check_cache(cache, block_ids, block_size, split_num)
    block_num_stride = cache.stride(0)
    block_elems = _get_block_elems(cache.element_size())

    workspace = torch.empty(
        (block_ids.numel(), block_num_stride),
        dtype=cache.dtype,
        device=cache.device,
    )

    grid = (triton.cdiv(total_elems, block_elems), block_ids.numel())
    _transpose_to_workspace_kernel[grid](
        cache,
        block_ids,
        workspace,
        block_num_stride,
        block_size,
        split_num,
        heads_per_split,
        head_dim,
        total_elems,
        block_elems,
    )
    _copy_workspace_back_kernel[grid](
        cache,
        block_ids,
        workspace,
        block_num_stride,
        total_elems,
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
    # 保持对外函数签名兼容旧实现。新实现根据 UB 容量自动决定 tile 大小，
    # 不再使用这个手动调节参数。
    del block_group_elems

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

    # K/V cache 是相互独立的 buffer。逐个处理可以让每个 cache 的临时
    # workspace 在下一次处理前释放，降低峰值显存占用。
    for cache in caches:
        _run_for_cache(cache, block_ids, block_size, split_num)
