---
name: vllm-torch-profile-capture
description: "How to capture a non-empty, steady-state, size-bounded torch profile from a running vLLM server on ROCm. Use when asked to collect (not analyze) a vLLM profile, when a capture produced an empty or tiny trace, when the capture script can't detect steady state, or when writing/fixing a profile collection script. Covers the --profiler-config startup contract, iteration-based windowing, the /metrics signal that actually works for steady-state detection, and verification. For analysis of an existing trace, use sglang-torch-profiler-analysis instead (same trace format)."
---

# vLLM torch profile 采集

只讲**采集**。拿到 trace 之后的分析用 `sglang-torch-profiler-analysis`（trace 格式相同）。

目标是三件事同时成立：**非空**（里面真有 GPU kernel）、**稳态**（不是冷启动或空转）、
**大小可控**（几十 MB 量级，不是 1KB 也不是几百 MB）。

采集失败的典型形态**不是报错**，而是"一切正常但 trace 里什么都没有"。
所以每一步都要观测真实信号，最后必须解开 trace 验一遍。

---

## 1. profiler 是启动时配的，不是运行时

```bash
vllm serve ... --profiler-config '{
  "profiler": "torch",
  "torch_profiler_dir": "'"$PROFILE_DIR"'",
  "torch_profiler_with_stack": false,
  "torch_profiler_use_gzip": true,
  "torch_profiler_record_shapes": false,
  "ignore_frontend": true,
  "delay_iterations": 3,
  "max_iterations": 10
}'
```

- **不认 `VLLM_TORCH_PROFILER_DIR` 环境变量**（较新的版本）。按旧写法设环境变量，
  服务照样起得来，但 profiler 没开 —— 静默失效。
- 改这里的任何一个值**都要重启服务**。
- `with_stack: true` 会让 trace 膨胀好几倍，采集时默认关掉。

### 采集窗口用引擎迭代数定界

`POST /start_profile` **不接受任何参数**，没法像 sglang 那样传 `num_steps`。
窗口长度在启动配置里定：`delay_iterations` + `max_iterations`。

这比按时间采好两层：

1. **体积可预测**：≈ `0.375 MB × 迭代数 × rank 数`（`with_stack=false` + gzip）。
   不受机器快慢影响。
2. **天然躲开空转**：`delay_iterations` 数的是 **worker step**，而 worker step
   只在有请求在跑的时候才发生。引擎空转根本不计数。

`delay/max` 必须和 **`ignore_frontend: true`** 一起给，否则前端 profiler
不跟迭代数走，会把整个区间都记下来（vLLM 自己在 `_validate_profiler_config` 里会 warn）。

**副作用**：`ignore_frontend: true` 会让启动日志里的 `Torch profiler enabled` 那行
**消失**（它来自 `not ignore_frontend` 的分支）。任何 grep 这行来确认 profiler
开了的就绪检查都会**假阴性**。用下面的主动探针，不要扒日志。

### 零代价探针：profiler 到底开没开

```bash
curl -sf -X POST "$BASE/stop_profile" -o /dev/null
```

- 没配 `--profiler-config` → `worker.profile()` 抛 RuntimeError → **HTTP 500**
- 配了但还没 start 过 → `self.profiler is None`，只打一句
  `Profiler was not started, nothing to stop.` 就返回 **200**，不建 wrapper、不落文件

**采集前先探一次，探不过立刻退出。** 否则要等 warm up + bench 跑完好几分钟才暴露。

---

## 2. 判稳态：`num_requests_running` 不能用

这是最容易踩的一个，因为它看起来正是为此设计的。

**实测**（MiniMax-M3 / TP4 / MI355X，CONC=10、ISL=75000、OSL=1）：引擎稳稳地
2 iter/s、每迭代 ~18900 token 在跑 prefill 时，`vllm:num_requests_running`
读出来在 **0~5 之间乱跳**，好几个采样点是 0。

原因：它是某个瞬间调度器里的请求数，而 OSL=1 的请求 prefill 完就走，这个 gauge
稳不下来。`running + waiting >= CONC` 也不行 —— 请求在 API 进程里 tokenize
75000 个 token 要花不少时间，还没进调度器。

用 `running >= CONC * 0.8` 当判据的后果：**永远等不到，超时误判成"没进稳态"**。

### 能用的信号

`vllm:iteration_tokens_total` 是个 histogram：

| | |
|---|---|
| `_count` | 引擎迭代次数（累计） |
| `_sum` | 这些迭代一共处理了多少 token |

两次采样求差：

```
iter/s     = Δ_count / Δt          引擎在持续步进（不是空转等数据集）
token/iter = Δ_sum / Δ_count       batch 里真有 prefill（不是零星 decode）
```

它直接测"引擎在不在干活、干的活肥不肥"，和请求怎么排队无关。

**阈值按工况算，不要拍脑袋**。prefill 场景可以用
`token/iter >= 2 × ISL × (1 - 缓存命中率)`，即"至少两条请求的未命中部分"。

**轮询间隔必须 ≥ 一个引擎迭代的耗时**，否则相邻两次采样的 `Δ_count` 经常是 0，
会把稳态误判成空闲。上面那个工况一个迭代 ~0.8s，用 0.25s 采样就会看到一片零。
拿不准就先采样一段再按秒聚合，看清楚形状再定阈值。

---

## 3. 三个会静默出错的地方

### 空 trace

`vllm bench serve` **在发请求之前要先生成数据集**。CONC=10 / ISL=75000 时
这段有十几秒，而真正的 benchmark 可能只有几秒。按"bench 进程起来了"计时开采，
采到的是引擎空转。

典型产物：**1.4KB / 57 个事件 / 唯一的 kernel 是 `hipDeviceSynchronize`**。
文件在，大小非零，`ls` 看着一切正常。

### "设个超大 num_prompts 让它跑够久"会反噬

数据集生成耗时 **∝ num_prompts × token 数**，而这段时间 GPU 全程空转。
3000 条 × 75k token 光 tokenize 就 > 2 分钟 —— 把上面那个坑放大了。

**条数应该按"需要多少个引擎迭代"反推**：

```
需要的条数 ≈ 迭代数 × max_num_batched_tokens / (ISL × (1 - 缓存命中率)) × 安全系数
```

### 采到的不是目标工况

prefix cache 命中率跑偏时（缓存没清 → ~99.8%，前缀对不上 → ~85%），
**trace 本身看不出任何异常** —— kernel 名字、事件数、GPU busy 全都正常，
但 kernel 分布已经是另一个工况的了。

所以要在采集脚本里断言命中率，用 `vllm:prefix_cache_queries_total` /
`vllm:prefix_cache_hits_total` 的增量算。

**预期值是 block 对齐后的天花板，不是你设的前缀比例**：

```
期望命中率 = floor(prefix_len / block_size) * block_size / ISL
```

`block_size=128`、`prefix_len=67500` 时是 `67456/75000 = 89.941%`，不是 90%。

另外 `POST /reset_prefix_cache` **只在 `VLLM_SERVER_DEV_MODE=1` 时才注册**，
否则是 404。清缓存失败必须硬退出 —— 缓存没清的话同 seed 第二次跑会得到
~99.8% 命中，测的根本不是 prefill。

---

## 4. 收产物

**用 before/after 差集，不要 glob。** trace 文件名带随机后缀
（`...rank0.1789443997556062053.pt.trace.json.gz`），而 `PROFILE_DIR` 里可能
已经有上一次的残留、`capture_traces/` 子目录、`profiler_out_*.txt`。

**"文件出现了" ≠ "文件写完了"。** trace 是异步 flush 的，几十 MB 的序列化要好几秒
到几十秒。做法：差集非空后，**连续两次采样总字节数不变**才算收敛。

顺手把 `profiler_out_<rank>.txt` 一起收走 —— 那是 vLLM 自己 dump 的
`key_averages` 表，白送的交叉验证。

`PROFILE_DIR` 必须落在**容器的 bind mount 里**。给一个宿主侧路径的话，
服务端会往容器内的同名路径写，产物在宿主上根本不存在，而且不会报错。

---

## 5. 验证：解开看，不看文件存不存在

参考实现：[scripts/verify_trace.py](scripts/verify_trace.py)

三层逐层收紧：

| 层 | 查什么 |
|---|---|
| 文件 | 每个 rank 的 `.pt.trace.json.gz` ≥ 64KB |
| 事件 | `cat=="kernel"` 的事件数够多，且**去重后的 kernel 名字**够多种。只有一两个名字基本就是空转 + 同步原语 |
| 语义 | 从 `execute_context_N(T)_generation_M(G)` 标注里数引擎迭代，看有多少个真带 prefill |

一份健康的 prefill trace 长这样（TP4 的单个 rank）：

```
大小        10.90 MB
traceEvents 759,301
GPU kernel  50,998 个事件 / 84 种
GPU busy    13,603 ms / 窗口 13,701 ms  (99.3%)
kernel 分类 moe=18183 gemm=12151 quant=11919 norm=5249 comm=3538 attention=1827
引擎迭代    30 个，其中 29 个真带 prefill，token/迭代 中位数 22,632
```

**`max_iterations` 数的是所有 worker step，包括空调度步。** 实测有过 30 个迭代里
15 个是空的（前端还在 tokenize 75000 token 的 prompt，调度器这步没排到活），
预算白费一半 —— 而文件大小、kernel 数全都过关。**把
`真带 prefill 的迭代 / 总迭代` 这个比值打出来**，它是判断"要不要把窗口开大"的唯一线索。

> `verify_trace.py` 是参考实现，不是共享库。每个 recipe 自己带一份并按工况改
> （kernel 家族正则是后端相关的，ROCm 和 CUDA 的名字完全不同）。

---

## 6. 收尾：把服务杀干净

**不要用 `pkill -f 'vllm serve'`。** 实测它会把自己那个 `bash -lc` 也匹配掉
（cmdline 里含这个串），结果 `pkill` 先杀了自己，vLLM 活得好好的、显存一直占着，
**而命令退出码是 0**。

正确做法：

1. 按记录下来的 PID 的**进程组**杀
2. 杀完**查显存确认真的放开**，不要信 `kill` 的返回值
3. 宿主机上 `rocm-smi` 通常不在 PATH（只在容器里和 `/opt/rocm*/bin/`），
   这个检查得**进容器**跑
4. 在宿主机上杀 `docker exec` 客户端**不会**终止容器里的进程

容器内进程落的文件属主是 root，宿主侧清理或移动会 `Permission denied` ——
收产物和清理要么在容器里做，要么先改属主。

---

## 7. 动手前

- **估算体积**：`0.375 MB × 迭代数 × rank 数`，开 `with_stack` 翻好几倍。
  共享机器上磁盘经常是紧的，先 `df -h` 看一眼，别跑到一半写爆盘。
- **看看卡被谁占着**：`rocm-smi --showmemuse --csv`。注意**查不到 ≠ 空闲** ——
  这条命令在有些机器上返回空，不能当作"没人用"。
- **冷启动加载权重约 10 分钟**，同 session 内重启因为 page cache 只要约 1 分钟。
  超时要按冷启动配，但"重启贵不贵"的判断要按热启动算。
