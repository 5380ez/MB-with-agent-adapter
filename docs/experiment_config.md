# MobilityBench 实验配置清单（补充材料）

> 用途：正文只讲“为什么选这些模型和 harness”，本文件集中列出**具体版本、运行参数、数据版本**，供逐条核对
> 7 个 harness × 3 个模型 = **21 个配置**。
>
> 记录时间：2026-11-05 · 机器：2×NVIDIA H20（97,871 MiB/卡）· 仓库 HEAD `1effd9b`
> 所有版本号与参数均从本机实际文件/进程日志读出，不是回忆值。

---

## 1. 数据集

| 用途 | 文件 | 规模 | md5（前 12 位） |
|---|---|---|---|
| 本次 21 配置所用评测集 | `data/datasets/pool_evaluable_20261028_sample5000.csv` | 5,000 行 × 46 列（39 MB） | `cfa139d96525e1e23b9362d26aa26495` |
| 字段完整池（抽样来源） | `data/datasets/pool_evaluable_20261028.csv` | 49,346 行（373 MB） | — |
| 语料分片 1 | `data/datasets/all_data_benchmark_50000.csv` | 50,000 行（288 MB） | `e588e6a44037` |
| 语料分片 2 | `data/datasets/all_data_benchmark_50000_2.csv` | 50,000 行（121 MB） | `ae98f982020e` |

抽样方式：按 `intent_family` 比例分层抽取（Basic Route Planning 2,524 / Basic Information Retrieval 1,465 /
Route-Dependent 547 / Preference 464），保留全部 46 列，`random_state=42`，抽后打乱。

## 2. 模型 checkpoint

| id | 模型 | 目录 | 体积 / 分片 | `model_type` | 层 / 隐层 / 头 / KV 头 | 原生最大长度 | 权重精度 | chat template |
|---|---|---|---|---|---|---|---|---|
| M1 | Qwen3.5-4B | `/opt/models/Qwen3.5-4B` | 8.8 G / 2 | `qwen3_5` | 32 / 2560 / 16 / 4 | 262,144 | bf16 | 自带 Jinja |
| M2 | Qwen3.8-27B | `/opt/models/Qwen3.8-27B` | 52 G / 18 | `qwen3_5` | 64 / 5120 / 24 / 4 | 262,144 | bf16 | 自带 Jinja |
| M3 | DeepSeek-V4-Flash-0731 | `/opt/models/DeepSeek-V4-Flash-0731` | 156 G / 48 | `deepseek_v4` | 43 / 4096 / 64 / 1，256 routed experts | 1,048,576 | fp8 `e4m3`，block 128×128 | **无 Jinja**，消息编码走 `encoding/` |

架构类：M1/M2 `Qwen3_5ForConditionalGeneration`；M3 `DeepseekV4ForCausalLM`。M3 为 0731 official release
（supersedes the preview checkpoint，48 分片 vs preview 的 46）。

## 3. 服务启动配置

推理栈：`/opt/vllm-env`，vLLM **0.31.0**，torch **2.13.0+cu130**，统一 `--port 8000`。

```bash
# M1 — Qwen3.5-4B（TP=1，仅 GPU0）
/opt/vllm-env/bin/vllm serve /opt/models/Qwen3.5-4B \
  --served-model-name qwen3.5-4b --dtype bfloat16 --max-model-len 32768 \
  --gpu-memory-utilization 0.7 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --default-chat-template-kwargs '{"enable_thinking": false}' --port 8000

# M2 — Qwen3.8-27B（TP=2）
/opt/vllm-env/bin/vllm serve /opt/models/Qwen3.8-27B \
  --served-model-name qwen3.8-27b --dtype bfloat16 --max-model-len 32768 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.95 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --default-chat-template-kwargs '{"enable_thinking": false}' --port 8000

# M3 — DeepSeek-V4-Flash-0731（TP=2；无 --dtype，由 fp8 checkpoint 决定）
TILELANG_DEFAULT_TARGET=cuda /opt/vllm-env/bin/vllm serve /opt/models/DeepSeek-V4-Flash-0731 \
  --served-model-name deepseek-v4-flash --max-model-len 32768 --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
  --default-chat-template-kwargs '{"enable_thinking": false}' --port 8000
```

| 生效参数 | M1 | M2 | M3 |
|---|---|---|---|
| `tensor_parallel_size` | 1（默认） | 2 | 2 |
| `gpu_memory_utilization` | 0.70 | 0.95 | 0.95 |
| `max_model_len` | 32,768 | 32,768 | 32,768 |
| `dtype` | bfloat16 | bfloat16 | 由 checkpoint 决定（fp8 `e4m3`；日志显示 `scale_fmt=ue8m0` → DeepGEMM） |
| `tool_call_parser` | `qwen3_xml` | `qwen3_xml` | `deepseek_v4` |
| `enable_auto_tool_choice` | true | true | true |
| `enable_thinking`（chat template kwarg） | false | false | false |
| 显存占用 | GPU0 66,465 MiB（GPU1 空闲） | 92,303 MiB ×2 | 92,611 MiB ×2 |
| **KV cache（引擎自报）** | **1,591,385 tok** | **1,723,733 tok** | **53,583 tok** |
| 32k 请求最大并发 | 48.57× | 52.60× | **1.64×** |
| 冷启动到 `/v1/models` 200 | ~65–90 s | ~105–160 s | ~150–220 s |

> **注意**：M3 的 KV 只能容纳 1.64 路 32k 请求，因此 `-p 48` 的 0731 配置必然长时间排队，其运行时长
> （2:31–3:03）不可与 Qwen 的 2:47/2:50 直接比较。

## 4. Harness（7 个）

| harness | 版本（本机实测） | 入口 / 二进制 | 协议 | 单 case 超时 | 上下文 | 说明 |
|---|---|---|---|---|---|---|
| `plan_and_execute` | 内置（HEAD `1effd9b`） | `mbench run -f plan_and_execute` | chat | 600 s（模型条目） | 32,768 | 自带 planner/worker/reporter 三段 prompt；planner 的 `intent` 为自由文本 |
| `react` | 内置（工作区，> `1effd9b`） | `mbench run -f react` | chat | 600 s | 32,768 | `agent.react.max_iterations = 10`；finish 输出 4 字段 JSON（intent/route_index/content/route_info） |
| `codex` | **codex-cli 0.155.1** | `.../plugin-packages/codex/latest/tools/codex_wrapper.sh`（在 PATH） | responses | 900 s | 32,768 | 沙箱 MCP；`strip_mcp_prefix` 归一化工具名 |
| `claude_code` | **2.1.293** | `/mnt/workspace/.local/share/agent-harnesses/bin/claude`（**不在 PATH**，需 `CLAUDE_BIN`） | anthropic | 900 s | 32,768 | `max_output_tokens 4096`；`--tools ""` 去掉内置工具 |
| `openclaw` | **2026.7.1**（build `2d2ddc4`） | `/opt/openclaw/bin/openclaw`（可 `OPENCLAW_BIN` 覆盖） | chat | 900 s | 32,768 | 每 run 生成 `openclaw.json`；`bootstrapMaxChars` 必须 >0 |
| `hermes` | **Hermes Agent v0.13.0**（插件 tag v2026.5.7，Python 3.11.15） | `/opt/hermes/hermes-agent/venv/bin/hermes` | chat | 900 s | 声明 65,536（**64K 下限**），实际服务窗口 32,768 | 进程内 `hermes_cli.oneshot` 驱动；provider 必须写 `custom` |
| `dsh` | **0.2.0-rc.2** | `/mnt/workspace/.local/share/agent-harnesses/bin/dsh`（需 `DSH_BIN`） | chat | 900 s | 32,768 | `--profile headless`，vLLM 路由经 `--patch` 注入 |

两层超时是不同概念：`configs/models/default.yaml` 的 `timeout: 600` 是**单次 LLM 调用**的客户端超时；
`configs/harness/default.yaml` 的 900 s 是 **CLI harness 单 case** 的预算。

## 5. 21 个配置清单

`-p` = `mbench run --parallel`；`not_run` 表示该配置**尚未在这份 5000 样本上跑过**（仅历史小规模冒烟）。

| # | harness | 版本 | 模型 | `-p` | 单 case 超时 | 状态 | run id |
|---|---|---|---|---|---|---|---|
| C01 | plan_and_execute | HEAD 1effd9b | M1 Qwen3.5-4B | 64 | 600 s | measured | `run_20261029_191826_653dbfe4` |
| C02 | plan_and_execute | HEAD 1effd9b | M2 Qwen3.8-27B | 64 | 600 s | measured | `run_20261029_164250_b483efe5` |
| C03 | plan_and_execute | HEAD 1effd9b | M3 DeepSeek-0731 | 48 | 600 s | measured | `run_20261030_094011_306a1589` |
| C04 | react | worktree > 1effd9b | M1 Qwen3.5-4B | 48 | 600 s | measured | `run_20261030_162851_4e13c080` |
| C05 | react | worktree > 1effd9b | M2 Qwen3.8-27B | 48 | 600 s | measured | `run_20261030_174512_fbfdc4b5` |
| C06 | react | worktree > 1effd9b | M3 DeepSeek-0731 | 48 | 600 s | measured | `run_20261031_102254_45245bea` |
| C07 | codex | 0.155.1 | M1 Qwen3.5-4B | 48 | 900 s | measured | `run_20261028_221739_6858a805` |
| C08 | codex | 0.155.1 | M2 Qwen3.8-27B | 48 | 900 s | measured | `run_20261028_145340_6d45b9c7` |
| C09 | codex | 0.155.1 | M3 DeepSeek-0731 | 48 | 900 s | measured | `run_20261029_233351_4508617a` |
| C10 | claude_code | 2.1.293 | M1 Qwen3.5-4B | 48 | 900 s | not_run | — |
| C11 | claude_code | 2.1.293 | M2 Qwen3.8-27B | 48 | 900 s | not_run | — |
| C12 | claude_code | 2.1.293 | M3 DeepSeek-0731 | 48 | 900 s | not_run | — |
| C13 | openclaw | 2026.7.1 | M1 Qwen3.5-4B | 48 | 900 s | not_run | — |
| C14 | openclaw | 2026.7.1 | M2 Qwen3.8-27B | 48 | 900 s | not_run | — |
| C15 | openclaw | 2026.7.1 | M3 DeepSeek-0731 | 48 | 900 s | not_run | — |
| C16 | hermes | v0.13.0 | M1 Qwen3.5-4B | 48 | 900 s | not_run | — |
| C17 | hermes | v0.13.0 | M2 Qwen3.8-27B | 48 | 900 s | not_run | — |
| C18 | hermes | v0.13.0 | M3 DeepSeek-0731 | 48 | 900 s | not_run | — |
| C19 | dsh | 0.2.0-rc.2 | M1 Qwen3.5-4B | 48 | 900 s | not_run | — |
| C20 | dsh | 0.2.0-rc.2 | M2 Qwen3.8-27B | 48 | 900 s | not_run | — |
| C21 | dsh | 0.2.0-rc.2 | M3 DeepSeek-0731 | 48 | 900 s | not_run | — |

机器可读版（可直接粘进表格）：

```csv
id,harness,harness_version,wire_api,model,checkpoint_id,parallel,case_timeout_s,status,run_id
C01,plan_and_execute,native(HEAD 1effd9b),chat,Qwen3.5-4B,M1,64,600,measured,run_20261029_191826_653dbfe4
C02,plan_and_execute,native(HEAD 1effd9b),chat,Qwen3.8-27B,M2,64,600,measured,run_20261029_164250_b483efe5
C03,plan_and_execute,native(HEAD 1effd9b),chat,DeepSeek-V4-Flash-0731,M3,48,600,measured,run_20261030_094011_306a1589
C04,react,native(worktree,>1effd9b),chat,Qwen3.5-4B,M1,48,600,measured,run_20261030_162851_4e13c080
C05,react,native(worktree,>1effd9b),chat,Qwen3.8-27B,M2,48,600,measured,run_20261030_174512_fbfdc4b5
C06,react,native(worktree,>1effd9b),chat,DeepSeek-V4-Flash-0731,M3,48,600,measured,run_20261031_102254_45245bea
C07,codex,codex-cli 0.155.1,responses,Qwen3.5-4B,M1,48,900,measured,run_20261028_221739_6858a805
C08,codex,codex-cli 0.155.1,responses,Qwen3.8-27B,M2,48,900,measured,run_20261028_145340_6d45b9c7
C09,codex,codex-cli 0.155.1,responses,DeepSeek-V4-Flash-0731,M3,48,900,measured,run_20261029_233351_4508617a
C10,claude_code,2.1.293,anthropic,Qwen3.5-4B,M1,48,900,not_run,
C11,claude_code,2.1.293,anthropic,Qwen3.8-27B,M2,48,900,not_run,
C12,claude_code,2.1.293,anthropic,DeepSeek-V4-Flash-0731,M3,48,900,not_run,
C13,openclaw,2026.7.1 (2d2ddc4),chat,Qwen3.5-4B,M1,48,900,not_run,
C14,openclaw,2026.7.1 (2d2ddc4),chat,Qwen3.8-27B,M2,48,900,not_run,
C15,openclaw,2026.7.1 (2d2ddc4),chat,DeepSeek-V4-Flash-0731,M3,48,900,not_run,
C16,hermes,v0.13.0 (plugin 2026.5.7),chat,Qwen3.5-4B,M1,48,900,not_run,
C17,hermes,v0.13.0 (plugin 2026.5.7),chat,Qwen3.8-27B,M2,48,900,not_run,
C18,hermes,v0.13.0 (plugin 2026.5.7),chat,DeepSeek-V4-Flash-0731,M3,48,900,not_run,
C19,dsh,0.2.0-rc.2 (headless),chat,Qwen3.5-4B,M1,48,900,not_run,
C20,dsh,0.2.0-rc.2 (headless),chat,Qwen3.8-27B,M2,48,900,not_run,
C21,dsh,0.2.0-rc.2 (headless),chat,DeepSeek-V4-Flash-0731,M3,48,900,not_run,
```

## 6. 评测配置

| 项目 | 取值 |
|---|---|
| 指标 | TS_P / TS_R / SC（tool）、ID / IE（instruction）、DEC_P / DEC_R（planning）、DR / FPR（answer）、IT / OT（efficiency） |
| ID 相似度判据 | `similarity_threshold = 0.7`；后端 `./embedding-model`（1.2 G，hidden 1024），加载失败时回退字符级 Jaccard |
| ID 标签空间（CLI 与 react） | 8 个封闭标签：地点查询 / 位置查询 / 附近周边搜索 / 天气查询 / 拥堵 / 距离 / 出发时间 / 路线规划 |
| ID 标签空间（plan_and_execute） | planner 自由文本，无枚举约束 |
| react 循环上限 | `max_iterations = 10` |
| CLI 工具调用上限 | 单任务 20 次（写进 CLI prompt） |
| 生成参数 | `temperature 0.1`、`max_tokens 8192` |
| 工具模式 | Sandbox（`--sandbox`，工具经 MCP 暴露 9 个沙箱工具） |
| 路由判分 | 路线类只读答案里的 `route_index`；`"-1"`（判定无法规划）记 0 分 |

## 7. 地图/回放数据版本（`data/sandbox/`）

| 文件 | 对应工具 | 条目 | 体积 | md5（前 12 位） |
|---|---|---|---|---|
| `nested_poi_data.json` | query_poi | 59,020 | 26.1 M | `faa2743a212f` |
| `driving_route_handle.json` | driving_route | 19,978 | 1,420.2 M | `82730b5d74ce` |
| `bus_route_mock.json` | bus_route | 8,522 | 332.5 M | `a31ea8015f72` |
| `bicycling_route_handle.json` | bicycling_route | 8,590 | 4.5 M | `a16576385266` |
| `search_around_poi_sandbox.json` | search_around_poi | 8,950 | 11.3 M | `926f6dd5ab49` |
| `reverse_geocoding_sandbox.json` | reverse_geocoding | 9,863 | 1.0 M | `ce0b2b2d19d5` |
| `weather_sandbox.json` | weather_query | 949 | 377 K | `4a798fd77ac1` |
| `traffic_info_mock.json` | traffic_info | 102 | 47 K | `6ed009924857` |
| `waypoint_mock.json` | 途经点 | 3,273 | 104 M | `ee991e7d5225` |

回放文件主体时间戳为 2026-09-21。`walking_route_handle.json`（2 条）与
`walking_route_handle_new.json`（1 条）是后补的**小体量占位文件**，不构成全量步行回放，引用时需单独说明。

## 8. 已知偏差（必须与表格一同发布）

1. **12 个配置未跑**：C10–C21（claude_code / openclaw / hermes / dsh × 3）在这份 5000 样本上没有结果，
   已在表中标 `not_run`，不得留空或按历史小规模数据推断。
2. **并行度不一致**：C01/C02 为 `-p 64`，其余已测配置为 `-p 48`。
3. **react 三行来自未提交代码**：`react/builder.py`、`react/nodes.py`、`agent/graph/state.py` 及两个
   react prompt 仍在工作区，发布前必须提交并回填 commit hash。
4. **0731 的评测口径**：C03 的 `evaluation/` 是停掉服务后用真实 embedding 重评的结果（ID 0.3261），
   其余已测配置多为 Jaccard 回退口径；跨行比较 ID/FPR 时需注明。
5. **M3 的 KV 容量**：53,583 tokens（1.64×），`-p 48` 下排队属预期，不是故障。

## 9. 复现要点

```bash
# 1) 起服务（三选一，见 §3），确认 /v1/models 的 root 指向正确目录
curl -s http://localhost:8000/v1/models | head -c 200
# 2) 跑一个配置（以 C05 为例）
.venv/bin/mbench run -m qwen3.8-27b \
  -d data/datasets/pool_evaluable_20261028_sample5000.csv -f react -p 48
# 3) 评测
.venv/bin/mbench eval --run-id <run_id>
```

前置条件（易漏）：`claude_code` 需 `CLAUDE_BIN`、`dsh` 需 `DSH_BIN`（两者不在 PATH）；
`hermes` 需 `HERMES_HOME` 与 `custom` provider、且必须声明 65,536 的 context 才能通过 64K 下限检查；
`openclaw` 需 node 在 PATH。
