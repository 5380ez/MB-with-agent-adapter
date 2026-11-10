# MobilityBench 已测配置总览（模型 × Harness）

> 记录时间：2026-11-10 · 机器：2×NVIDIA H20（97,871 MiB/卡，48 vCPU，256 GB RAM）· 仓库 HEAD `434905a`（react/plan_and_execute 修复提交）
> 本文档汇总**实际测过的全部配置**：3 个模型 + 7 个 harness 的版本与参数，以及每个组合的 run id。
> 版本号与参数均取自本机二进制/配置/日志，不是回忆值。补充材料用的 21 配置勾选表见 `docs/experiment_config.md`。

---

## 1. 模型（3 个）

| id | 模型 | 本地路径 | 体积 / 分片 | 架构 | 层 / 隐层 / 头 / KV 头 | 原生上下文 | 权重精度 | chat template |
|---|---|---|---|---|---|---|---|---|
| **M1** | Qwen3.5-4B | `/opt/models/Qwen3.5-4B` | 8.8 G / 2 | `qwen3_5` | 32 / 2560 / 16 / 4 | 262,144 | bf16 | 自带 Jinja |
| **M2** | Qwen3.8-27B | `/opt/models/Qwen3.8-27B` | 52 G / 18 | `qwen3_5` | 64 / 5120 / 24 / 4 | 262,144 | bf16 | 自带 Jinja |
| **M3** | DeepSeek-V4-Flash-**0731** | `/opt/models/DeepSeek-V4-Flash-0731` | 156 G / 48 | `deepseek_v4` | 43 / 4096 / 64 / 1，256 routed experts | 1,048,576 | fp8 `e4m3`，block 128×128 | **无 Jinja**（用 `encoding/` 脚本） |

M3 为 0731 official release（48 分片；preview 版是 46 分片、`/opt/models/DeepSeek-V4-Flash`）。
对外服务名分别是 `qwen3.5-4b` / `qwen3.8-27b` / `deepseek-v4-flash`。

### 服务启动命令（推理栈 `/opt/vllm-env`，vLLM 0.31.0 + torch 2.13.0+cu130）

```bash
# M1 —— Qwen3.5-4B（TP=1，仅 GPU0）
/opt/vllm-env/bin/vllm serve /opt/models/Qwen3.5-4B \
  --served-model-name qwen3.5-4b --dtype bfloat16 --max-model-len 32768 \
  --gpu-memory-utilization 0.7 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --default-chat-template-kwargs '{"enable_thinking": false}' --port 8000

# M2 —— Qwen3.8-27B（TP=2）
/opt/vllm-env/bin/vllm serve /opt/models/Qwen3.8-27B \
  --served-model-name qwen3.8-27b --dtype bfloat16 --max-model-len 32768 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.95 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --default-chat-template-kwargs '{"enable_thinking": false}' --port 8000

# M3 —— DeepSeek-V4-Flash-0731（TP=2；无 --dtype，由 fp8 checkpoint 决定）
TILELANG_DEFAULT_TARGET=cuda /opt/vllm-env/bin/vllm serve /opt/models/DeepSeek-V4-Flash-0731 \
  --served-model-name deepseek-v4-flash --max-model-len 32768 --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
  --default-chat-template-kwargs '{"enable_thinking": false}' --port 8000
```

| 生效参数 | M1 | M2 | M3 |
|---|---|---|---|
| 张量并行 | 1 | 2 | 2 |
| 显存占用 | GPU0 66,465 MiB | 92,303 MiB ×2 | 92,611 MiB ×2 |
| `max_model_len` | 32,768 | 32,768 | 32,768 |
| `tool_call_parser` | `qwen3_xml` | `qwen3_xml` | `deepseek_v4` |
| **KV cache** | 1,591,385 tok（48.57×） | 1,723,733 tok（52.60×） | **53,583 tok（1.64×）** |
| 冷启动到 healthy | ~65–90 s | ~105–160 s | ~150–220 s |

> M3 的 KV 只够 1.64 路 32k 请求，`-p 48` 下必然排队；`-p 16` 实测比 `-p 48` 慢 72 分钟。

---

## 2. Harness（7 个）

| harness | 版本（本机实测） | 入口 / 二进制 | 协议 | 单 case 超时 | 上下文 | 输出上限 | MCP |
|---|---|---|---|---|---|---|---|
| `react` | 内置（`434905a`） | `mbench run -f react` | chat | 600 s（模型条目） | 32,768 | 8192 | 进程内 ToolRegistry |
| `plan_and_execute` | 内置（`434905a`） | `mbench run -f plan_and_execute` | chat | 600 s（模型条目） | 32,768 | 8192 | 进程内 ToolRegistry |
| `codex` | **codex-cli 0.155.1** | `codex`（在 PATH，解析到 `plugin-packages/codex/latest/tools/codex_wrapper.sh`） | `openai` / **responses** | 900 s | 32,768 | 默认 | 是 |
| `claude_code` | **2.1.293** (Claude Code) | `/mnt/workspace/.local/share/agent-harnesses/bin/claude`（**不在 PATH**，需 `CLAUDE_BIN`） | **anthropic** | 900 s | 32,768 | 4096 | 是 |
| `openclaw` | **OpenClaw 2026.7.1 (2d2ddc4)** | `/opt/openclaw/bin/openclaw`（`OPENCLAW_BIN` 可覆盖） | chat | 900 s | 32,768 | 4096 | 是 |
| `dsh` | **0.2.0-rc.2** | `/mnt/workspace/.local/share/agent-harnesses/bin/dsh`（**不在 PATH**，需 `DSH_BIN`） | chat，`profile: headless` | 900 s | 32,768 | 默认 | 是 |
| `hermes` | **Hermes Agent v0.13.0 (2026.5.7)** | `/opt/hermes/hermes-agent/venv/bin/hermes`（Python 3.11.15） | chat，provider 必须写 `custom` | 900 s | 声明 65,536（**64K 下限**） | 4096 | 是 |

### 每个 harness 的特殊前提（复现时最容易漏）

- **react**：`agent.react.max_iterations = 10`；finish 输出四字段 JSON（`intent` / `route_index` / `content` / `route_info`），`plan_result` 末尾带 ```json 块供 answer 指标读取；对应提交 `434905a`。
- **plan_and_execute**：自带 planner/worker/reporter prompt；planner 的 `intent` 为**自由文本**（无 8 标签枚举），`RawPlan.steps` 需两层结构（已加裸字符串归一化）。
- **codex**：`wire_api: responses`（走 `/v1/responses`）；工具名带 `mcp__sandbox__` 前缀时由 `strip_mcp_prefix` 归一化。
- **claude_code**：`ANTHROPIC_BASE_URL` 不能带 `/v1`；用 `--tools ""` 关掉内置 33 个工具，否则每次请求多 1.7 万 token。
- **openclaw**：每 run 生成 `openclaw.json`，`agents.defaults.bootstrapMaxChars` 必须 >0，否则配置非法；需 node 在 PATH。
- **dsh**：`--profile headless --json`，模型路由由适配器用 `--patch` 注入（`llm-pi-ai` → OpenAI 兼容 vLLM）；**每个 case 起一个 `sandbox_tools.py`**，单进程加载约 1.9 GB 回放数据 → `-p 48` 会吃穿内存并连带 OOM 掉 vLLM，**上限 `-p 8`**。自带 `bash`/`read`/`todo_write` 等内置工具，会出现在工具轨迹里。
- **hermes**：进程内 `hermes_cli.oneshot` 驱动；`model.provider: custom`；必须同时给主模型与辅助压缩模型声明 `context_length: 65536` 才能过 64K 检查；需清掉 `OPENAI_BASE_URL`/`OPENAI_API_KEY`（本机指向 DeepSeek）。

---

## 3. 所有 run 共用的运行参数

| 项 | 取值 |
|---|---|
| 数据集 | `data/datasets/pool_evaluable_20261028_sample5000.csv`（5,000 行 × 46 列，md5 `cfa139d96525e1e23b9362d26aa26495`） |
| 并行度 | 默认 `-p 48`；例外：p&e 4b/27b 为 64，dsh 为 8，react·0731 有一次 16 |
| 生成参数 | `temperature 0.1`，`max_tokens 8192`（`configs/models/default.yaml`） |
| 模型侧超时 | `timeout: 600`（三个模型条目统一，单次 LLM 调用） |
| 工具模式 | Sandbox（9 个沙箱工具，经 MCP 或进程内注册） |
| 评测 | 相似度阈值 0.7；embedding 后端不可用时回退字符级 Jaccard；ID 标签空间见各 harness 说明 |

---

## 4. 已测组合（run 清单）

### 4.1 codex

| 模型 | `-p` | 时长 | run id | ID | DR | FPR |
|---|---|---|---|---|---|---|
| M1 Qwen3.5-4B | 48 | 2:58:00 | `run_20261028_221739_6858a805` | .6136 | .7036 | .4811 |
| M2 Qwen3.8-27B | 48 | 4:22:15 | `run_20261028_145340_6d45b9c7` | .7338 | .7679 | .6138 |
| M3 DeepSeek-0731 | 48 | 2:30:57 | `run_20261029_233351_4508617a` | .7312 | .8376 | .6498 |

### 4.2 plan_and_execute

| 模型 | `-p` | 时长 | run id | ID | DR | FPR |
|---|---|---|---|---|---|---|
| M1 Qwen3.5-4B | **64** | 0:33:33 | `run_20261029_191826_653dbfe4` | .7138 | .7222 | .5156 |
| M2 Qwen3.8-27B | **64** | 1:54:26 | `run_20261029_164250_b483efe5` | .7432 | .8444 | .5664 |
| M3 DeepSeek-0731 | 48 | 2:47:05 | `run_20261030_094011_306a1589` | **.3261**¹ | .7748 | .4748 |

¹ 该 run 的 `evaluation/` 是停服后用真实 embedding 后端重评的结果（Jaccard 口径备份在 `evaluation_jaccard/`，ID .3067）。其余组合为各自评时的后端。

### 4.3 react

| 模型 | `-p` | 时长 | run id | ID | DR | FPR |
|---|---|---|---|---|---|---|
| M1 Qwen3.5-4B | 48 | 0:58:57 | `run_20261030_162851_4e13c080` | .7270 | .8260 | .5090 |
| M2 Qwen3.8-27B | 48 | 2:50:09 | `run_20261030_174512_fbfdc4b5` | .7338 | .8574 | .4644 |
| M3 DeepSeek-0731 | 48 | 3:03:28 | `run_20261031_102254_45245bea` | .6712 | .7930 | .2476 |
| M3 DeepSeek-0731 | 16 | 4:15:18 | `run_20261109_093103_95e085e6` | .6690 | .7788 | .2470 |

### 4.4 dsh

| 模型 | `-p` | 时长 | run id | ID | DR | FPR |
|---|---|---|---|---|---|---|
| M3 DeepSeek-0731 | **8** | 6:45:59 | `run_20261109_144004_b5bf615a` | .7268 | .8498 | .6402 |
| M3 DeepSeek-0731 | 48 | ✗ 两次崩溃 | `run_20261109_141325_46df65cb` / `run_20261109_142044_d33f101f` | — | — | — |

`-p 48` 两轮都在 ~90 秒内把可用内存从 258 GB 压到 72 GB，随后 mbench 与 vLLM 一起被 OOM kill，零可用数据。

### 4.5 尚未在 5000 样本上测的组合（11 个）

| harness | 未测模型 |
|---|---|
| `claude_code` | M1 / M2 / M3（仅 2026-10-09 的 115 条 qwen3-4b 冒烟） |
| `openclaw` | M1 / M2 / M3（仅 2026-10-12 的 4 条冒烟） |
| `hermes` | M1 / M2 / M3（仅 2026-10-12 的 4 条冒烟） |
| `dsh` | M1 / M2（0731 已测） |

---

## 5. 口径注意事项（进对比表前必须一并说明）

1. **并行度不统一**：p&e 的 4b/27b 是 `-p 64`，dsh 是 `-p 8`，react·0731 有一轮 `-p 16`，其余 `-p 48`。时长不可直接横比。
2. **react 代码版本**：`graph/state.py`、`react/nodes.py`、`react/builder.py`、`prompts/react/*.md` 共 +269/−60，已提交为 `434905a`；引用这批结果时用该 commit。
3. **dsh 的工具维度含内置工具**：24,300 次调用里 1,973 次（8.1%）是 `bash` / `todo_write` / `read` / `create_goal` 等 dsh 自带工具，TS_P/TS_R/SC 与其他 harness 不完全可比。
4. **ID 的 Basic Information Retrieval 分族恒为 0.15 左右**：`POI Search` / `Nearby Search` 未映射进指标表，是该族的固定损失，不是模型能力差异。
5. **p&e·0731 的 ID 是散文标签惩罚**：planner 输出描述句（中位 32 字符），逐字包含判据命中率低；同模型 codex 为 .7312、dsh 为 .7268。
6. **路线族 FPR 只认 `route_index`**：react / codex / dsh 的答案带该字段；`"-1"`（判定无法规划）仍记 0 分。
7. **服务被 dsh 的 `-p 48` 打挂过两次**：跑完后需重启 vLLM（命令见 §1），并确认 `/v1/models` 的 `root` 指向正确 checkpoint 目录。

---

## 6. 复现一条配置的命令模板

```bash
cd /mnt/workspace/project/MobilityBench

# 1) 起对应模型（见 §1），确认服务与模型名
curl -s http://localhost:8000/v1/models | head -c 200

# 2) 跑一条配置（以 codex × 0731 为例）
.venv/bin/mbench run -m deepseek-v4-flash \
  -d data/datasets/pool_evaluable_20261028_sample5000.csv \
  -f codex --sandbox -p 48

# 3) 评测
.venv/bin/mbench eval --run-id <run_id>

# 需要环境变量的 harness：
#   claude_code → CLAUDE_BIN=/mnt/workspace/.local/share/agent-harnesses/bin/claude
#   dsh         → DSH_BIN=/mnt/workspace/.local/share/agent-harnesses/bin/dsh
#   hermes      → HERMES_HOME=<临时目录>，provider=custom，context_length=65536
```
