# MooncakeConnector PD 分离问题定位 Wiki

本文用于定位 `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`
相关的 Prefill-Decode 分离问题。重点覆盖三类问题：

1. 请求不返回
2. 精度异常
3. 启动、连接与资源初始化异常

适用连接器：

- `MooncakeConnectorV1`
- `vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.MooncakeConnector`

## 1. 背景和核心链路

MooncakeConnector 的 PD 分离链路可以拆成 7 个阶段：

1. P 节点执行 prefill。
2. P 节点 scheduler 在 `request_finished()` 中生成 `kv_transfer_params`。
3. P 节点延迟释放已经计算好的 KV block，等待 D 节点拉取。
4. D 节点收到带 `do_remote_prefill=true` 的请求。
5. D 节点 scheduler 计算需要从 P 节点拉取的 token 和 block。
6. D worker 通过 ZMQ side channel 获取 P worker 的 KV metadata。
7. D worker 通过 Mooncake `TransferEngine.batch_transfer_sync_read()` 从 P 节点读取 KV。
8. D worker 发送 `DONE_RECVING_MSG`，P worker 收到 ACK 流程后释放延迟保留的 KV block。

对应代码入口：

| 阶段 | 代码入口 | 作用 |
|---|---|---|
| P 侧生成传输参数 | `MooncakeConnectorScheduler.request_finished()` | 生成 `kv_transfer_params`，记录待延迟释放的 block |
| D 侧识别 remote prefill | `MooncakeConnectorScheduler.get_num_new_matched_tokens()` | 计算需要异步加载的 external token |
| D 侧记录拉取任务 | `MooncakeConnectorScheduler.update_state_after_alloc()` | 保存 local block 和 remote 参数 |
| D 侧构造 worker metadata | `MooncakeConnectorScheduler.build_connector_meta()` | 将 scheduler 侧请求转成 worker 可消费的 metadata |
| D worker 启动拉取 | `MooncakeConnectorWorker.start_load_kv()` | 拆分端口、block、group pull 信息 |
| ZMQ 获取 P 侧 metadata | `KVCacheRecvingThread._get_remote_metadata()` | 获取 P 侧 KV 地址、stride、TransferEngine 端口 |
| Mooncake 传输 | `KVCacheRecvingThread._transfer_kv_cache_all_groups()` | 生成 src/dst/length 并读 KV |
| P 侧释放 block | `KVCacheSendingThread.run_busy_loop()` | 处理 `DONE_RECVING_MSG` |

## 2. 通用信息收集

定位前先收集以下信息，避免在错误方向上排查：

### 2.1 部署拓扑

记录：

- P 节点数量
- D 节点数量
- 每个节点 NPU 数量
- P 侧 TP、PP、DP、PCP、DCP
- D 侧 TP、PP、DP、PCP、DCP
- P/D 是否跨机
- P/D 是否使用相同模型路径
- P/D 是否使用相同 tokenizer 和 chat template
- 是否启用 prefix cache
- 是否启用 ACLGraph
- 是否启用 NZ KV cache
- 是否启用 fused transpose op
- 是否启用 MTP、Mamba、SWA、Hybrid KV cache

### 2.2 启动参数

重点保存 P/D 两侧完整启动命令，至少包含：

```bash
--kv-transfer-config
--tensor-parallel-size
--pipeline-parallel-size
--data-parallel-size
--enable-prefix-caching
--max-model-len
--block-size
--served-model-name
```

如果使用 proxy，也保存 proxy 对 P 和 D 发出的请求体。

### 2.3 关键环境变量

重点记录：

```bash
env | grep -E "VLLM|ASCEND|MOONCAKE|HCCL|HCCN|LD_LIBRARY_PATH"
```

Mooncake 动态库路径需要包含对应 site-packages 路径，例如：

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
```

### 2.4 日志级别

建议在复现时打开 debug 日志。不同版本启动方式略有差异，通常可用：

```bash
export VLLM_LOGGING_LEVEL=DEBUG
```

然后分别保存：

```bash
prefill.log
decode.log
proxy.log
```

### 2.5 一键提取关键日志

可以先用下面命令粗筛：

```bash
grep -E "Initializing Mooncake|KVCacheSendingThread started|num_blocks|register kv caches metadata|get_num_new_matched_tokens|update_state_after_alloc|request_finished|Delaying free|start_load_kv|Adding request|p_node_cp_group_meta|local_remote_block_port_mappings|Mooncake kv transfer meta|Mooncake transfer request|KV cache transfer for request|DONE_RECVING_MSG|Sending done recving signal|Failed to receive ACK|Receive failed|Send failed|Mooncake transfer failed|Remote kv_group2layeridx is inconsistent|Force freed expired request|Address already in use|Timeout waiting for KV Cache thread" *.log
```

## 3. 请求不返回定位流程

请求不返回通常是以下几段之一卡住：

1. proxy 没有正确串起 P/D 请求。
2. P 节点没有生成可供 D 节点拉取的 `kv_transfer_params`。
3. D scheduler 没有识别 `do_remote_prefill`。
4. D worker 没有启动 KV 拉取。
5. D 侧无法通过 ZMQ 拿到 P 侧 metadata。
6. Mooncake RDMA 传输失败或超时。
7. D 侧没有发送 DONE，或 P 侧没有收到 DONE/ACK，导致资源不释放。
8. 传输失败后 request/block 清理异常，scheduler 一直等待。

### 3.1 第一步：确认请求是否进入 P/D 两段

检查 proxy 日志，确认同一个用户请求是否先打到 P，再打到 D。

P 请求应包含：

```json
{
  "kv_transfer_params": {
    "do_remote_decode": true
  }
}
```

D 请求应包含 P 返回的：

```json
{
  "kv_transfer_params": {
    "do_remote_prefill": true,
    "remote_block_ids": "...",
    "remote_engine_id": "...",
    "remote_request_id": "...",
    "remote_host": "...",
    "remote_port": "..."
  }
}
```

如果 D 请求没有 `do_remote_prefill=true`，优先排查 proxy，而不是 MooncakeConnector。

### 3.2 第二步：确认 P 侧生成了 `kv_transfer_params`

在 P 节点日志中搜索：

```bash
grep -E "MooncakeConnector request_finished|Delaying free|do_remote_decode|kv_transfer_params" prefill.log
```

期望看到类似日志：

```text
MooncakeConnector request_finished, request_status=...
Delaying free of ... blocks for request ...
```

判断方法：

- 有 `Delaying free`：P 侧已经保留 KV block，等待 D 侧拉取。
- 没有 `Delaying free`：P 侧没有进入 remote decode 完成流程。
- 有 `request_finished` 但没有返回 transfer 参数：检查 `request.status` 是否为 `FINISHED_LENGTH_CAPPED`。

P 侧生成 `kv_transfer_params` 的必要条件：

- `request.kv_transfer_params` 不为空。
- `request.kv_transfer_params["do_remote_decode"] == true`。
- request 状态为 `RequestStatus.FINISHED_LENGTH_CAPPED`。
- `computed_block_ids` 非空。

对应代码：

```python
MooncakeConnectorScheduler.request_finished()
```

如果这里失败，继续检查：

- proxy 给 P 请求的 `max_tokens` 是否符合 PD 分离流程。
- P 请求是否被异常中止。
- P 请求是否没有实际生成任何 prompt KV block。
- block size 和 prompt 长度是否导致 `computed_block_ids` 为空。

### 3.3 第三步：检查 P 侧返回字段是否完整

P 侧返回给 D 的 `kv_transfer_params` 至少应包含：

```text
do_remote_prefill
do_remote_decode
remote_block_ids
remote_engine_id
remote_request_id
remote_host
remote_port
remote_pcp_size
remote_dcp_size
remote_ptp_size
last_token_id
remote_multi_nodes_meta_mapping
num_prompt_blocks
```

其中最关键的是：

- `remote_block_ids`
- `remote_engine_id`
- `remote_request_id`
- `remote_host`
- `remote_port`

D 侧如果发现字段缺失，会打印：

```text
Got invalid KVTransferParams
```

检查命令：

```bash
grep -E "Got invalid KVTransferParams|remote_block_ids|remote_engine_id|remote_host|remote_port|remote_request_id" decode.log
```

### 3.4 第四步：确认 D scheduler 识别 remote prefill

在 D 节点日志中搜索：

```bash
grep -E "get_num_new_matched_tokens|update_state_after_alloc|do_remote_prefill|num_external_tokens" decode.log
```

期望看到：

```text
MooncakeConnector get_num_new_matched_tokens: num_computed_tokens=..., kv_transfer_params=...
MooncakeConnector update_state_after_alloc: num_external_tokens=..., kv_transfer_params=...
```

判断方法：

- `num_external_tokens > 0`：D scheduler 会异步等待 remote KV。
- `num_external_tokens == 0`：说明 D 侧认为无需拉 KV，可能是 full prefix cache hit，也可能是 token 计算不符合预期。
- `kv_transfer_params` 为空：proxy 没有把 P 返回参数传给 D。

如果 D scheduler 没识别到 remote prefill，重点检查：

- D 请求体是否包含 `kv_transfer_params`。
- `do_remote_prefill` 是否为 `true`。
- P/D token ids 是否一致。
- prompt 是否被 proxy 改写。

对应代码：

```python
MooncakeConnectorScheduler.get_num_new_matched_tokens()
MooncakeConnectorScheduler.update_state_after_alloc()
```

### 3.5 第五步：确认 D worker 开始拉 KV

搜索：

```bash
grep -E "start_load_kv|Adding request|Trans info" decode.log
```

期望看到：

```text
start_load_kv for request ... from remote engine ...
Adding request ... to the queue.Trans info:...
```

如果没有 `start_load_kv`：

- scheduler metadata 没有被带到 worker。
- 当前请求没有进入包含 KV connector metadata 的 batch。
- request 可能在调度阶段被抢占或取消。

如果有 `start_load_kv` 但没有 `Adding request`：

- `_get_kv_split_metadata()` 可能断言失败。
- P/D TP、PCP、DCP 映射可能不合法。
- 日志中搜索 `AssertionError`。

检查命令：

```bash
grep -E "AssertionError|prefill_tp_size|decode_tp_size|num_external_blocks|num_prompt_blocks|tp_num_need_pulls" decode.log
```

### 3.6 第六步：检查 ZMQ side channel 是否可达

P 侧每个 worker 会启动一个 ZMQ ROUTER socket。端口计算方式大致为：

```text
handshake_port = kv_port + pp_rank * tp_size + tp_rank + pcp_rank * prefill_tp_size
```

P 侧搜索：

```bash
grep -E "KVCacheSendingThread started listening|Address already in use|encountered exception" prefill.log
```

期望看到：

```text
KVCacheSendingThread started listening on path: tcp://<host>:<port>
```

D 侧搜索：

```bash
grep -E "Receive failed|Send failed|Returned socket|Failed to receive ACK|Unexpected error occurred in socket" decode.log
```

常见问题：

- `remote_host` 是容器内 IP，D 节点无法访问。
- `kv_port` 被占用。
- 防火墙或容器网络没有放通。
- 多实例使用了相同 `kv_port`。
- P/D 使用了相同 `engine_id`。

端口冲突时常见日志：

```text
zmq.error.ZMQError: Address already in use
```

处理建议：

- Docker 优先使用 `--net=host`。
- 每个实例使用独立 `kv_port`。
- 8 卡节点建议 `kv_port >= 28000`。
- 16 卡节点建议 `kv_port >= 36000`。
- 避免 `kv_port` 落在 AscendDirectTransport 随机端口范围。

AscendDirectTransport 可能占用：

```text
[20000, 20000 + npu_per_node * 1000)
```

### 3.7 第七步：检查 D 是否拿到 P 侧 metadata

D 侧 `_get_remote_metadata()` 会向 P 侧发送 `GET_META_MSG`，拿到：

- remote `engine_id`
- remote TransferEngine RPC port
- remote KV cache base address
- remote block length
- remote block stride
- remote block size scale
- remote KV group 信息

搜索：

```bash
grep -E "Remote kv_group2layeridx|Returned socket|Receive failed|Send failed" decode.log
```

如果出现：

```text
Conflict engine id ... with local engine id
```

说明 P/D `engine_id` 冲突。每个实例必须唯一。

如果出现：

```text
Remote kv_group2layeridx is inconsistent with local
```

请求可能继续运行，但后续有较高概率出现精度异常或 shape/stride 相关问题。需要检查 P/D 模型、KV cache 配置、Hybrid 配置是否一致。

### 3.8 第八步：检查 Mooncake 传输是否成功

D 侧搜索：

```bash
grep -E "Mooncake transfer request|KV cache transfer for request|Mooncake transfer failed|Failed to transfer KV cache" decode.log
```

期望看到：

```text
Mooncake transfer request=... session id=... src=... dst=... length=...
KV cache transfer for request ... took ... ms
```

如果出现：

```text
Mooncake transfer failed for request. remote_request_id=..., ret=...
```

优先检查：

- Mooncake 是否正确安装。
- `LD_LIBRARY_PATH` 是否包含 mooncake。
- Mooncake master 服务是否正常。
- HCCN/RDMA 网络是否正常。
- P/D NPU IP 是否互通。
- TLS 配置是否一致。
- TransferEngine RPC port 是否可达。
- KV buffer 是否成功 register。

基础网络检查：

```bash
for i in {0..7}; do hccn_tool -i $i -link -g; done
for i in {0..7}; do hccn_tool -i $i -net_health -g; done
for i in {0..7}; do hccn_tool -i $i -tls -g; done | grep switch
cat /etc/hccn.conf
```

A3 16 卡环境把 `{0..7}` 改为 `{0..15}`。

### 3.9 第九步：检查 DONE/ACK 和 P 侧释放

D 侧传输完成后会发送 `DONE_RECVING_MSG`。

D 侧搜索：

```bash
grep -E "Sending done recving signal|Received response for request|Failed to receive ACK|Unexpected error occurred in socket" decode.log
```

P 侧搜索：

```bash
grep -E "Got DONE_RECVING_MSG|finish req not in reqs to process|Force freed expired request" prefill.log
```

判断方法：

- D 有 `Sending done recving signal` 且收到 `ACK`：D 到 P 的清理链路正常。
- D 报 `Failed to receive ACK`：P 侧可能没有处理 DONE，或 ZMQ socket 异常。
- P 报 `finish req not in reqs to process`：请求生命周期跟踪可能不匹配，检查请求是否重复完成、提前释放、request id 是否变化。
- P 报 `Force freed expired request`：P 侧等待 D 拉取超时，说明 DONE 没有及时完成或 D 侧传输失败。

### 3.10 请求不返回决策树

```text
请求不返回
|
+-- P 侧是否有 Delaying free?
|   |
|   +-- 否：检查 proxy -> P 的 do_remote_decode、P request_finished 状态
|   |
|   +-- 是：继续
|
+-- D 请求是否包含 do_remote_prefill?
|   |
|   +-- 否：检查 proxy 是否透传 P 返回的 kv_transfer_params
|   |
|   +-- 是：继续
|
+-- D 侧是否有 start_load_kv / Adding request?
|   |
|   +-- 否：检查 D scheduler metadata、block 分配、断言失败
|   |
|   +-- 是：继续
|
+-- D 是否能 GET_META?
|   |
|   +-- 否：检查 remote_host、kv_port、engine_id、ZMQ 端口、防火墙
|   |
|   +-- 是：继续
|
+-- Mooncake transfer 是否成功?
|   |
|   +-- 否：检查 RDMA/HCCN/Mooncake/TransferEngine/register buffer
|   |
|   +-- 是：继续
|
+-- DONE/ACK 是否完成?
    |
    +-- 否：检查 D -> P ZMQ、request id、P 侧 task tracker
    |
    +-- 是：继续排查 scheduler 是否仍在等待 failed block 或其他请求状态
```

## 4. 精度异常定位流程

精度异常的核心原则：先确认 KV 是否“传对”，再确认传输后是否“排布对”，最后确认 P/D token 流是否“一致”。

### 4.1 建立对照实验

必须使用确定性配置：

```json
{
  "temperature": 0,
  "top_p": 1,
  "max_tokens": 64
}
```

建议按顺序做 5 组对照：

| 组别 | 配置 | 目的 |
|---|---|---|
| A | 非 PD 单体服务 | 获得 baseline |
| B | PD，P/D TP 相同 | 验证基础 PD 流程 |
| C | PD，P/D TP 不同 | 验证 TP pull 和 reformat |
| D | PD，关闭 fused transpose | 隔离 fused op |
| E | PD，关闭 NZ 或切换 eager | 隔离 NZ / 图模式相关问题 |

如果 B 正常、C 异常，重点排查：

- `tp_num_need_pulls`
- `remote_tp_offset`
- reformat
- GQA/MLA/sparse KV head 映射

如果 A 正常、B 异常，重点排查：

- P/D token 是否一致
- block ids 是否一致
- remote/local KV group 是否一致
- `remote_block_ids` 和 `local_block_ids` 是否映射正确

### 4.2 确认 P/D 输入 token 完全一致

精度异常最常见根因之一是 P 和 D 的 prompt 实际 token 不一致。

检查：

- P/D 是否使用同一模型目录。
- P/D 是否使用同一 tokenizer。
- proxy 是否对 P 和 D 使用不同 chat template。
- P/D 是否有不同的截断参数。
- P/D 是否有不同的多模态预处理。
- P/D 是否有不同的 `max_model_len`。

建议在 proxy 层打印：

```text
request_id
prompt text hash
prompt_token_ids length
prompt_token_ids first 16
prompt_token_ids last 16
```

如果 P/D token 不一致，不继续排查 Mooncake 传输。

### 4.3 检查 `kv_transfer_params` 的关键字段

保存 P 返回给 D 的完整 `kv_transfer_params`。

重点检查：

```text
remote_block_ids
remote_engine_id
remote_request_id
remote_host
remote_port
remote_pcp_size
remote_dcp_size
remote_ptp_size
last_token_id
num_prompt_blocks
remote_multi_nodes_meta_mapping
```

判断：

- `remote_block_ids` 为空：D 不会拉 KV，可能退化为本地计算或异常等待。
- `num_prompt_blocks` 不等于预期：检查 P 侧 prompt token 长度和 block size。
- `remote_ptp_size` 和 P 侧 TP 不一致：D 会按错误 TP 计算 remote rank。
- `last_token_id` 和 D 侧 prompt 最后 token 不匹配：P/D token 流不一致。

### 4.4 检查 P/D KV group 是否一致

D 获取 P 侧 metadata 后，会比较：

```python
agent_meta.kv_group2layeridx != self.kv_group2layeridx
```

搜索：

```bash
grep -E "Remote kv_group2layeridx is inconsistent|kv_group2layeridx" decode.log
```

如果出现不一致，重点检查：

- P/D 是否启用了相同的 Hybrid KV cache manager。
- P/D 是否有相同的 Mamba/SWA/FullAttentionSpec 组合。
- P/D 是否使用相同模型 config。
- P/D 是否使用相同 vLLM/vLLM-Ascend 代码版本。
- P/D 是否有相同 speculative/MTP 配置。
- P/D 是否有相同 quantization 配置。

### 4.5 检查 block ids 映射

D 侧搜索：

```bash
grep -E "start_load_kv|Adding request|Mooncake kv transfer meta|Mooncake transfer request" decode.log
```

重点看：

- `local_block_ids`
- `remote_block_ids`
- `group_pulls`
- `remote_handshake_port`
- `tp_num_need_pulls`
- `remote_tp_offset`

普通 TP 相同场景：

- `tp_num_need_pulls` 通常应为 1。
- `remote_block_ids` 和 `local_block_ids` 数量应一致或按 prefix cache 合理裁剪。

P/D TP 不同场景：

- `tp_num_need_pulls` 应等于一个 D rank 需要从多少个 P rank 拉 KV。
- 每个 D rank 的多个 pull 需要覆盖完整 head 范围。
- 只有最后一个 pull 后才应触发 reformat。

PCP/DCP 场景：

- 检查 `local_remote_block_port_mappings`。
- 检查 `remote_port_send_num`。
- 检查最后一个非满 block 是否被放到 D 侧最后 block。

相关日志：

```bash
grep -E "p_node_cp_group_meta|d_node_cp_group_meta|local_remote_block_port_mappings|num_external_blocks|num_prompt_blocks" decode.log
```

### 4.6 检查 prefix cache 影响

prefix cache 会导致 P/D 实际需要传输的 block 数不同。

代码中会根据：

```text
num_external_tokens
num_prompt_blocks
num_computed_tokens
```

裁剪 remote block。

排查方法：

1. 使用完全不命中 prefix cache 的新 prompt 复现。
2. 关闭 prefix cache 后复现。
3. 对比 `num_external_tokens` 和 `num_prompt_blocks`。
4. 检查 `remote_start_idx = num_computed_tokens // remote_kernel_token_size` 是否符合预期。

如果关闭 prefix cache 后精度正常，重点检查：

- `num_computed_tokens` 是否被正确带到 D 侧。
- P/D block size 是否一致。
- compress ratio 是否影响 token per block。
- SWA tail block 裁剪是否正确。

### 4.7 检查 TP 不一致后的 reformat

当 `tp_num_need_pulls > 1` 时，D 从多个 P rank 拉到的 KV 需要重排。

相关函数：

```python
reformat_kv_cache()
reformat_kv_cache_with_fused_op()
reformat_kv_cache_hybrid_linear_torch()
_cat_kv_cache()
_nz_kv_cache()
```

检查日志：

```bash
grep -E "tp_num_need_pulls|remote_tp_offset|Mooncake kv transfer meta|transpose_kv_cache|reformat" decode.log
```

隔离 fused op：

```bash
export VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK=0
```

隔离 NZ：

- 使用不开启 NZ 的配置复现。
- 或对比 `enable_kv_nz` 前后输出。

判断方法：

- 关闭 fused op 后恢复：重点看 `torch.ops._C_ascend.transpose_kv_cache_by_block`。
- 关闭 NZ 后恢复：重点看 `_nz_kv_cache()` 和 NZ layout。
- 只有 P/D TP 不一致时异常：重点看 `_cat_kv_cache()` 和 head 拼接顺序。

### 4.8 检查 MLA / Sparse / GQA head 映射

普通 attention 会根据 `num_key_value_heads` 计算 P/D 每 rank 的 KV head。

风险点：

- P TP 大于 KV head 数。
- D TP 和 P TP 的 KV head 分组不能整除。
- MLA 和 sparse 路径把 KV head 视作特殊分组。

检查：

```bash
grep -E "num_key_value_heads|tp_num_need_pulls|use_mla|use_sparse|remote_tp_offset" decode.log
```

如果是 DeepSeek MLA：

- `_get_tp_num_need_pulls()` 中 MLA 场景通常按特殊逻辑处理。
- 不要直接套普通 GQA head 分组结论。

如果是 sparse：

- 检查模型 config 中是否存在 `index_topk`。
- sparse 场景同样走特殊分组。

### 4.9 检查 Mamba / Hybrid / SWA 特殊路径

Hybrid KV cache 下可能同时存在 attention KV 和 state group。

重点：

- Mamba state 不是普通 context block。
- D 侧通常需要从 P 侧拿 `h(N-1)`，再由 D 侧 recompute 最后一个 token。
- SWA 只传窗口尾部 block。
- Hybrid attention group 可能每个 group 的 pull 数不一样。

相关函数：

```python
_state_prefill_token_count()
_truncate_request_for_prefill()
_append_mamba_transfer_meta()
_get_hybrid_remote_rank_group_pulls()
_get_swa_transfer_block_ids()
```

排查步骤：

1. 确认是否启用 Hybrid KV cache manager。
2. 检查 `kv_group2layeridx` 是否包含 `MambaSpec`。
3. 检查 P 侧是否执行了 `_truncate_request_for_prefill()`。
4. 检查 D 侧 `num_external_tokens` 是否为 `prompt_len - 1`。
5. 检查 Mamba state 是否只在 final shard 传输。
6. 检查 SWA group 是否只保留窗口尾部 block。

### 4.10 精度异常决策树

```text
精度异常
|
+-- 非 PD baseline 是否正常?
|   |
|   +-- 否：先排模型、权重、量化、采样参数
|   |
|   +-- 是：继续
|
+-- PD 且 P/D TP 相同是否正常?
|   |
|   +-- 否：检查 P/D token、kv_transfer_params、block ids、KV group
|   |
|   +-- 是：继续
|
+-- 仅 P/D TP 不同时异常?
|   |
|   +-- 是：检查 tp_num_need_pulls、remote_tp_offset、reformat、GQA/MLA
|   |
|   +-- 否：继续
|
+-- 关闭 fused transpose 后是否恢复?
|   |
|   +-- 是：检查 fused transpose op
|   |
|   +-- 否：继续
|
+-- 关闭 NZ 后是否恢复?
|   |
|   +-- 是：检查 NZ layout 和 _nz_kv_cache
|   |
|   +-- 否：继续
|
+-- 关闭 prefix cache 后是否恢复?
|   |
|   +-- 是：检查 num_computed_tokens、remote_start_idx、SWA tail block
|   |
|   +-- 否：检查 Hybrid/Mamba、模型配置一致性、proxy token 流
```

## 5. 启动、连接与资源初始化异常定位流程

这类问题经常最终表现为请求不返回，但根因发生在服务启动或 KV cache 注册阶段。

### 5.1 检查 Mooncake 是否可导入

在容器内执行：

```bash
python - <<'PY'
from mooncake.engine import TransferEngine
print("Mooncake import ok:", TransferEngine)
PY
```

失败时检查：

- Mooncake 是否安装。
- Python 版本对应的 site-packages 路径是否正确。
- `LD_LIBRARY_PATH` 是否包含 Mooncake 动态库。
- `/usr/local/lib`、`/usr/local/lib64` 是否在 `LD_LIBRARY_PATH` 中。

### 5.2 检查 HCCN / RDMA 网络

A2 8 卡：

```bash
for i in {0..7}; do hccn_tool -i $i -lldp -g | grep Ifname; done
for i in {0..7}; do hccn_tool -i $i -link -g; done
for i in {0..7}; do hccn_tool -i $i -net_health -g; done
for i in {0..7}; do hccn_tool -i $i -netdetect -g; done
for i in {0..7}; do hccn_tool -i $i -gateway -g; done
for i in {0..7}; do hccn_tool -i $i -ip -g; done
for i in {0..7}; do hccn_tool -i $i -tls -g; done | grep switch
cat /etc/hccn.conf
```

A3 16 卡：

```bash
for i in {0..15}; do hccn_tool -i $i -lldp -g | grep Ifname; done
for i in {0..15}; do hccn_tool -i $i -link -g; done
for i in {0..15}; do hccn_tool -i $i -net_health -g; done
for i in {0..15}; do hccn_tool -i $i -netdetect -g; done
for i in {0..15}; do hccn_tool -i $i -gateway -g; done
for i in {0..15}; do hccn_tool -i $i -vnic -g; done
for i in {0..15}; do hccn_tool -i $i -tls -g; done | grep switch
cat /etc/hccn.conf
```

跨节点 ping：

```bash
# A2
for i in {0..7}; do hccn_tool -i $i -ping -g address <remote_npu_ip>; done

# A3
for i in {0..15}; do hccn_tool -i $i -hccs_ping -g address <remote_npu_ip>; done
```

### 5.3 检查端口规划

MooncakeConnector 至少涉及两类端口：

1. `kv_port` 起始的 ZMQ side channel 端口。
2. Mooncake AscendDirectTransport 使用的传输端口。

推荐：

| 每节点 NPU 数 | AscendDirectTransport 可能占用范围 | 推荐 `kv_port` |
|---|---|---|
| 8 | `20000 - 27999` | `>= 28000` |
| 16 | `20000 - 35999` | `>= 36000` |

检查端口占用：

```bash
ss -lntp | grep -E "28000|30000|36000|40000"
```

如果启动报：

```text
Address already in use
```

处理：

- 增大 `kv_port`。
- 不同实例错开 `kv_port`。
- 确认旧进程已退出。
- 避免多个 P/D 实例共用相同端口区间。

### 5.4 检查 `engine_id`

每个 P/D 实例必须使用唯一 `engine_id`。

如果 P/D 相同，会触发：

```text
Conflict engine id ... with local engine id
```

建议命名：

```text
prefill-0
prefill-1
decode-0
decode-1
```

### 5.5 检查 KV cache 注册

搜索：

```bash
grep -E "num_blocks|register kv caches metadata|block_len_per_addr|block_stride_per_addr|block_size_scale|register_buffer|validate_register_region_count" *.log
```

期望：

- 有 `num_blocks`。
- 有 `Mooncake register kv caches metadata`。
- 没有 register buffer 异常。

重点字段：

| 字段 | 含义 | 异常影响 |
|---|---|---|
| `kv_caches_base_addr` | 本地 KV tensor 基地址 | 错误会导致读写错误或精度异常 |
| `block_len_per_addr` | 一个 block 的字节长度 | 错误会导致传输长度错误 |
| `block_stride_per_addr` | 相邻 block 的字节 stride | 错误会导致读写偏移错误 |
| `block_size_scale` | tensor block 与 logical block 的比例 | 错误会导致 block id 展开错误 |
| `kv_group2layeridx` | KV group 到 layer 的映射 | P/D 不一致会导致错层 |

如果 register region 数量过多：

- 普通 attention 路径会尝试按 storage merge。
- Hybrid/Mamba 可能需要特殊注册。
- 检查是否有大量非连续 KV tensor。

### 5.6 检查 P/D 并行配置约束

启动或首次请求可能因为 P/D 并行配置不合法失败。

基本约束：

```text
prefill_tp_size >= decode_tp_size
```

Hybrid Mamba 约束：

```text
prefill_tp_size % decode_tp_size == 0
```

非 MLA / 非 sparse 场景还需要：

```text
d_node_heads_per_rank % p_node_heads_per_rank == 0
```

如果日志出现：

```text
prefill_tp_size ... must be greater than or equal to decode_tp_size
```

需要调整 P/D TP 配置。

如果日志出现：

```text
tp_num_need_pulls ...
chosen_rank_list ...
num_external_blocks ...
num_prompt_blocks ...
```

重点检查 PCP/DCP/TP 映射。

## 6. 常用隔离开关

### 6.1 关闭 fused KV transpose

用于判断精度异常是否由 fused transpose op 引起：

```bash
export VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK=0
```

### 6.2 切换 eager / 关闭图模式

用于隔离 ACLGraph 相关调度或 cache 行为：

```bash
--enforce-eager
```

### 6.3 关闭 prefix cache

用于隔离 prefix cache 命中导致的 block 裁剪问题。

移除或关闭：

```bash
--enable-prefix-caching
```

### 6.4 对比 P/D TP

建议先用 P/D TP 相同验证基础链路，再切换为异构 TP：

```text
case 1: P TP = D TP
case 2: P TP > D TP
```

如果 case 1 正常、case 2 异常，优先查 reformat 和 KV head 映射。

## 7. 问题报告模板

提交问题时建议包含：

~~~markdown
## 现象

- 请求不返回 / 精度异常 / 启动失败：
- 首次出现版本：
- 是否稳定复现：

## 拓扑

- P 节点数：
- D 节点数：
- 每节点 NPU 数：
- P TP/PP/DP/PCP/DCP：
- D TP/PP/DP/PCP/DCP：
- 是否跨机：

## 模型和配置

- 模型路径：
- quantization：
- block size：
- max model len：
- prefix cache：
- ACLGraph：
- NZ：
- MTP / Mamba / SWA / Hybrid：

## 启动命令

P:
```bash
...
```

D:
```bash
...
```

Proxy:
```bash
...
```

## 请求

```json
{
  "model": "...",
  "messages": []
}
```

## P 返回的 kv_transfer_params

```json
{}
```

## 关键日志

P:
```text
...
```

D:
```text
...
```

Proxy:
```text
...
```

## 已做隔离

- 非 PD baseline：
- PD 同 TP：
- PD 异 TP：
- 关闭 fused transpose：
- 关闭 NZ：
- 关闭 prefix cache：
- enforce eager：
~~~

## 8. 快速结论表

| 现象 | 优先检查 | 关键日志 |
|---|---|---|
| P 有返回，D 不返回 | D 是否收到 `do_remote_prefill` | `get_num_new_matched_tokens` |
| D 卡住等待 KV | ZMQ metadata 或 Mooncake transfer | `Receive failed`、`Mooncake transfer failed` |
| P 侧显存不释放 | DONE/ACK 未完成 | `DONE_RECVING_MSG`、`Force freed expired request` |
| 只有跨机不返回 | HCCN/RDMA/remote_host | `session id=<host>:<port>` |
| 只有异构 TP 精度异常 | reformat/head 映射 | `tp_num_need_pulls`、`remote_tp_offset` |
| 只有 prefix cache 场景异常 | block 裁剪 | `num_computed_tokens`、`num_prompt_blocks` |
| 只有 NZ 开启异常 | NZ layout | `_nz_kv_cache` 相关路径 |
| 只有 fused op 开启异常 | fused transpose op | `transpose_kv_cache_by_block` |
| Hybrid/Mamba 异常 | state group 特殊逻辑 | `MambaSpec`、`_truncate_request_for_prefill` |
