# SPDX-License-Identifier: Apache-2.0

import argparse
import gc
import time

import torch

from vllm_ascend.ops.triton.transpose_kv_cache_by_block import transpose_kv_cache_by_block_triton
from vllm_ascend.utils import enable_custom_op

DEVICE = "npu:0"
DTYPE = torch.bfloat16

CACHE_BLOCKS = 1024
BLOCK_SIZE = 128
HEAD_NUM = 4
HEAD_DIM = 128
LAYERS = 94

SPLIT_NUMS = (4, 2)
CAL_BLOCKS_LIST = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)

WARMUP = 5
ITERS = 20


def synchronize(device: str) -> None:
    torch.npu.synchronize()


def make_caches(device: str) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    k_caches = []
    v_caches = []
    shape = (CACHE_BLOCKS, BLOCK_SIZE, HEAD_NUM, HEAD_DIM)
    for _ in range(LAYERS):
        k_cache = torch.empty(shape, dtype=DTYPE, device=device)
        v_cache = torch.empty(shape, dtype=DTYPE, device=device)
        k_caches.append(k_cache)
        v_caches.append(v_cache)
    return k_caches, v_caches


def cache_size_gib() -> float:
    element_size = torch.empty((), dtype=DTYPE).element_size()
    bytes_per_cache = CACHE_BLOCKS * BLOCK_SIZE * HEAD_NUM * HEAD_DIM * element_size
    total_bytes = bytes_per_cache * LAYERS * 2
    return total_bytes / 1024**3


def block_ids_for(cal_blocks: int, device: str) -> torch.Tensor:
    return torch.arange(cal_blocks, dtype=torch.int64, device=device)


def run_ascendc(
    k_caches: list[torch.Tensor],
    v_caches: list[torch.Tensor],
    block_ids: torch.Tensor,
    split_num: int,
) -> None:
    torch.ops._C_ascend.transpose_kv_cache_by_block(
        k_caches,
        v_caches,
        block_ids,
        BLOCK_SIZE,
        HEAD_NUM,
        HEAD_DIM,
        split_num,
        LAYERS,
    )


def run_triton(
    k_caches: list[torch.Tensor],
    v_caches: list[torch.Tensor],
    block_ids: torch.Tensor,
    split_num: int,
) -> None:
    transpose_kv_cache_by_block_triton(k_caches, v_caches, block_ids, BLOCK_SIZE, split_num)


@torch.no_grad()
def run_torch(
    k_caches: list[torch.Tensor],
    v_caches: list[torch.Tensor],
    block_ids: torch.Tensor,
    split_num: int,
) -> None:
    if split_num == 1 or block_ids.numel() == 0:
        return

    num_blocks = block_ids.numel()

    def run(cache: torch.Tensor) -> None:
        selected = cache.index_select(0, block_ids)
        transposed = (
            selected.reshape(num_blocks, split_num, BLOCK_SIZE, -1)
            .transpose(1, 2)
            .contiguous()
            .reshape_as(selected)
        )
        cache.index_copy_(0, block_ids, transposed)

    for k_cache, v_cache in zip(k_caches, v_caches):
        run(k_cache)
        run(v_cache)


def bench(fn, device: str, warmup: int = WARMUP, iters: int = ITERS) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    synchronize(device)
    return (time.perf_counter() - start) * 1000 / iters


def print_header(device: str) -> None:
    print("transpose_kv_cache_by_block benchmark")
    print(
        f"shape=[{CACHE_BLOCKS}, {BLOCK_SIZE}, {HEAD_NUM}, {HEAD_DIM}], "
        f"layers={LAYERS}, dtype={DTYPE}, device={device}, estimated cache memory={cache_size_gib():.2f} GiB"
    )
    print(f"cal_blocks={CAL_BLOCKS_LIST}, split_nums={SPLIT_NUMS}, warmup={WARMUP}, iters={ITERS}")
    print()
    print("| split_num | blocks | ascendc_ms | triton_ms | torch_ms | triton/ascendc | torch/ascendc |")
    print("|----------:|-------:|-----------:|----------:|---------:|---------------:|--------------:|")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=DEVICE, help="NPU tensor device, for example npu:0.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device
    if not device.startswith("npu"):
        raise ValueError(f"Only NPU devices are supported, got {device!r}.")

    if not enable_custom_op():
        raise RuntimeError("AscendC custom op is unavailable, cannot compare against torch.ops._C_ascend.")

    caches = make_caches(device)
    print_header(device)

    try:
        for split_num in SPLIT_NUMS:
            for cal_blocks in CAL_BLOCKS_LIST:
                block_ids = block_ids_for(cal_blocks, device)

                ascendc_ms = bench(
                    lambda block_ids=block_ids, split_num=split_num: run_ascendc(
                        caches[0], caches[1], block_ids, split_num
                    ),
                    device,
                )
                triton_ms = bench(
                    lambda block_ids=block_ids, split_num=split_num: run_triton(
                        caches[0], caches[1], block_ids, split_num
                    ),
                    device,
                )

                torch_ms = bench(
                    lambda block_ids=block_ids, split_num=split_num: run_torch(
                        caches[0], caches[1], block_ids, split_num
                    ),
                    device,
                )
                triton_ratio = triton_ms / ascendc_ms
                torch_ratio = torch_ms / ascendc_ms

                print(
                    f"| {split_num:9d} | {cal_blocks:6d} | {ascendc_ms:10.3f} | "
                    f"{triton_ms:9.3f} | {torch_ms:8.3f} | "
                    f"{triton_ratio:14.3f} | {torch_ratio:13.3f} |"
                )
    finally:
        caches[0].clear()
        caches[1].clear()
        gc.collect()
        torch.npu.empty_cache()


if __name__ == "__main__":
    main()
