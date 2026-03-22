---
name: sglang-amd-bench
description: >
  Benchmark sglang serving performance on AMD Instinct GPUs (MI355X, MI300X, MI308X) with
  various parallel configurations (TP, DP, EP). Use this skill whenever the user wants to
  benchmark, profile, or measure sglang inference performance on AMD GPUs, compare different
  parallelism strategies, sweep ISL/OSL/concurrency combinations, or establish a performance
  baseline for LLM serving. Trigger this skill when the user mentions sglang benchmarking,
  serving throughput testing, TTFT/TPOT measurement, AMD GPU inference performance, parallel
  config comparison (TP vs DP vs EP), or wants to find the best sglang configuration for a
  specific model on AMD hardware. This skill covers mix mode (non-disaggregated) serving only.
  For PD-disaggregation benchmarking, a separate skill is needed.
---

# SGLang AMD Benchmark

Benchmark sglang LLM serving on AMD Instinct GPUs across parallel configurations (TP/DP/EP) and workload shapes (ISL/OSL/Concurrency). This skill runs in **mix mode** (non-disaggregated) — prefill and decode happen on the same GPUs. It produces a performance baseline and suggests config-level optimizations.

## Key Metrics

Every benchmark collects these metrics per (ISL, OSL, Concurrency) combination:


| Metric             | Unit  | Description                                               |
| ------------------ | ----- | --------------------------------------------------------- |
| TTFT               | ms    | Time To First Token — latency from request to first token |
| TPOT               | ms    | Time Per Output Token — average inter-token latency       |
| Input throughput   | tok/s | Input tokens processed per second across all requests     |
| Output throughput  | tok/s | Output tokens generated per second across all requests    |
| Total throughput   | tok/s | Input + Output token throughput combined                  |
| Per-GPU throughput | tok/s | Total throughput / number of GPUs                         |


Per-GPU throughput is the most important efficiency metric — it shows how well each GPU is utilized. Two configs might have similar total throughput, but the one using fewer GPUs has better per-GPU throughput and is more cost-efficient.

## Common Workspace Layout

The standard development environment uses `/sgl-workspace` as the root workspace inside Docker containers:

```
/sgl-workspace/
├── sglang/                    # sglang source (installed via pip -e, dev mode)
├── aiter/                     # AITER source (AMD AI Tensor Engine)
├── mori/                      # Mori (communication library)
└── <model_short>_<YYYYMMDD>/  # benchmark output directories (created by this skill)
```

All benchmark artifacts (logs, reports) are saved under `/sgl-workspace/` by default. If the user specifies a different workspace, use that instead.

## Core Principle: Ask First, Execute Later

**Do NOT guess or assume any configuration.** Every detail must be explicitly confirmed by the user before execution begins. The workflow has two distinct phases:

1. **Planning phase** (Steps 0–1): Gather ALL information through conversation. Ask questions, wait for answers. Do not proceed to the next question until the current one is answered.
2. **Confirmation gate** (Step 2): Present the complete plan as a summary. Get explicit "go ahead" from the user.
3. **Execution phase** (Steps 3–5): Only after full confirmation, run the benchmarks.

If at any point you're unsure about a parameter, **ask**. Never fill in a value the user hasn't confirmed.

## Workflow

### Step 0: Model & Environment Discovery

**Ask the user these questions one by one. Wait for each answer before asking the next.**

#### 0a. Model selection — ask this FIRST

**"Which model do you want to benchmark?"**

The user may respond with:

- A full HuggingFace model ID (e.g., `deepseek-ai/DeepSeek-R1-0528`)
- A short name (e.g., "DeepSeek R1", "Llama 70B", "Qwen 235B")
- A local path to the model weights

If the user gives a short name, confirm the exact model ID (e.g., "Do you mean `deepseek-ai/DeepSeek-R1-0528`?").

#### 0b. Single-node or multi-node?

**"Is this single-node or multi-node?"**

- Single-node: 1 node, typically 8 GPUs
- Multi-node: ask how many nodes and GPUs per node

If multi-node, also ask for:

- Network interface (`GLOO_SOCKET_IFNAME`)
- InfiniBand HCAs (`NCCL_IB_HCA`)
- Head node IP (`SGLANG_HOST_IP`)

#### 0c. Access the GPU node

**"How do I access the GPU node?"**

- SSH command? (e.g., `ssh user@gpu-node`)
- Docker container? (e.g., `docker exec -it <container> bash`)
- Already on the machine?
- For multi-node: ask about access to each node

#### 0d. Probe the environment

Once connected, probe automatically (no need to ask — just run and report back):

- Run `rocm-smi --showid` → report GPU count, model (MI355X, MI300X, MI308X), architecture
- Run `pip show sgl-kernel 2>/dev/null && python3 -c "import sglang; print('sglang version:', sglang.__version__)"` → report sglang version
- Run `pip list | grep -i aiter` → report AITER status
- Check common paths: `/sgl-workspace/sglang`, `/sgl-workspace/aiter`, `/sgl-workspace/mori`

#### 0e. Locate model weights

The user may or may not have specified where the model weights are stored. If they haven't provided a path, do a quick search — but don't waste time on this:

```bash
# Check HuggingFace cache env var
echo $HUGGINGFACE_HUB_CACHE

# Check common cache locations
ls ~/.cache/huggingface/hub/ 2>/dev/null | head -20

# Check common mount points for model storage
for dir in /mnt /raid /data; do
  find "$dir" -maxdepth 3 -type d -name "*$(echo MODEL_NAME | tr '/' '-')*" 2>/dev/null | head -5
done
```

Replace `MODEL_NAME` with the model name (e.g., `DeepSeek-R1-0528`). If you find a match, confirm with the user:

> "I found what looks like the model weights at `/data/models/DeepSeek-R1-0528/`. Is this the right location?"

If nothing turns up quickly, ask:

> "I couldn't find the model weights on this machine. Where are they stored?"

The final `--model-path` for sglang can be either a local path or a HuggingFace model ID (sglang will download if not cached).

#### 0f. Report findings and confirm

Present everything you found to the user:

> "Here's what I have so far:
>
> - **Model**: deepseek-ai/DeepSeek-R1-0528
> - **Weights**: /data/models/DeepSeek-R1-0528/
> - **GPUs**: 8x MI355X (gfx950)
> - **sglang**: v0.5.x at /sgl-workspace/sglang
> - **AITER**: installed
> - **Setup**: single-node
>
> Does this look right? Anything I should know about this environment?"

### Step 1: Configuration Planning

**Ask each of these questions explicitly. Do not move forward until you have clear answers for ALL of them.**

#### 1a. MTP decision (if applicable)

If the model is MTP-capable (DeepSeek-R1/V3, Qwen3), ask:

**"This model supports Multi-Token Prediction (MTP), which can improve decode throughput. By default we run without MTP for a clean baseline. What would you like to do?"**

1. Run without MTP (baseline only)
2. Run with MTP enabled
3. Run both and compare

#### 1b. Server setup

**"Do you already have a sglang server running, or should I launch one for each config?"**

- If running: **"What's the server URL?"** (e.g., `http://127.0.0.1:8000`)
- If launching: confirm the port, `--mem-fraction-static`, any extra sglang flags the user wants

Also ask: **"Any additional sglang launch flags you want to use?"** (e.g., `--quantization fp8`, `--chunked-prefill-size`, `--schedule-policy`, etc.)

Note: `--disable-radix-cache` is enabled by default in `serve.sh` for benchmarking. User can opt out with `DISABLE_RADIX_CACHE=0`.

#### 1c. Parallel configurations

This is the most important decision in the benchmark. Read `references/server_config.md` for the full reference on parallelism types, naming conventions, EP modes, and how to reason about config choices.

**Before asking the user**, do the following:

1. **Read the model's `config.json`** from the weights directory using the script in `references/server_config.md`. Extract KV heads, Q heads, expert count, and detect attention type (MLA/GQA/MHA).

2. **Analyze** the 4 factors described in `references/server_config.md` → "How to Reason About Parallel Config":
   - Weight size vs GPU HBM → which TP values fit?
   - Attention type + KV heads → TP or DP-attention?
   - MoE vs Dense → EP applicable?
   - EP mode → all-to-all or all-reduce?

3. **Present your analysis to the user** — show your reasoning (weight size calc, KV head implications, why certain configs are better). Then present a suggested config table and **ask the user to pick**.

4. **If EP is involved**, ask which EP mode (all-to-all or all-reduce), or suggest benchmarking both.

Wait for the user to respond. If they say "try all of them" or "you decide", confirm your suggested set before proceeding.

#### 1d. Benchmark sweep parameters

**"What ISL (input sequence length), OSL (output sequence length), and concurrency levels do you want to sweep?"**

If the user isn't sure, offer options but still ask them to pick:

> "Some common approaches:
>
> 1. **Specific pairs** — e.g., (ISL=512, OSL=256), (ISL=1024, OSL=512) — good for simulating real workloads
> 2. **Full sweep** — provide separate ISL, OSL, and CON lists, benchmark all combinations
>
> Which approach? And what values?"

If the user says "you pick" or "whatever makes sense", then suggest values and **ask for confirmation before proceeding**:

> "Here's what I'd suggest:
>
> - ISL: 128, 512, 1024, 2048, 4096
> - OSL: 128, 512, 1024, 2048
> - Concurrency: 1, 16, 64, 128, 256
>
> That's 5 × 4 × 5 = 100 runs per config, times 2 configs = 200 total runs.
> Estimated ~3+ hours. Want to proceed with these, or adjust?"

### Step 2: Confirmation Gate

**Do NOT start any benchmark until this step is complete.**

#### Naming convention

Use this pattern for directories:

```
BENCH_DIR=/sgl-workspace/<model_short>_<YYYYMMDD>
```

Per-config dirs: `<CONFIG>_mtp<0|1>` (e.g., `DP8EP8_mtp0`, `TP8_mtp0`)

#### Present the plan summary

> **Benchmark Plan Summary**
>
> | Item | Value |
> |------|-------|
> | Model | deepseek-ai/DeepSeek-R1-0528 |
> | GPU | 8x MI355X |
> | Mode | Mix (non-disaggregated) |
> | Bench dir | `/sgl-workspace/DeepSeek-R1_20260322/` |
>
> **Sweep:** ISL=[128, 512, 1024, 2048], OSL=[128, 512, 1024], CON=[1, 16, 64, 128, 256]

#### Confirm each config with dry-run

For each parallel config, run `scripts/serve.sh` in dry-run mode to show the user the exact sglang launch command. Present them **one by one** and get confirmation before moving on to the next.

```bash
export SGLANG_USE_AITER=1
BENCH_DIR=/sgl-workspace/<model_short>_$(date +%Y%m%d)

# Config 1 — dry run
MODEL_PATH=<MODEL_PATH> CONFIG=DP8EP8_A2A MTP=0 \
LOG_DIR=$BENCH_DIR/DP8EP8_A2A_mtp0 DRY_RUN=1 bash serve.sh
```

Show the output and ask: **"Config 1 (DP8EP8_A2A_mtp0) — does this look right?"**

Then the next config:

```bash
# Config 2 — dry run
MODEL_PATH=<MODEL_PATH> CONFIG=TP8 MTP=0 \
LOG_DIR=$BENCH_DIR/TP8_mtp0 DRY_RUN=1 bash serve.sh
```

**"Config 2 (TP8_mtp0) — does this look right?"**

Continue until all configs are confirmed. If the user wants changes to any config, adjust and re-run the dry run. Once all configs are confirmed, proceed to Step 3.

### Step 3: Benchmark Execution

Only proceed here after the user has confirmed ALL configs in Step 2.

For each parallel config:

**3a. Launch sglang server**

```bash
MODEL_PATH=<MODEL_PATH> CONFIG=<CONFIG> MTP=<0|1> \
LOG_DIR=$BENCH_DIR/<CONFIG>_mtp<0|1> \
bash serve.sh
```

If the user already has a running server, skip the launch and use their URL.

**3b. Wait for server ready**

```bash
timeout 600 bash -c 'until curl -s http://localhost:${PORT:-30000}/health > /dev/null 2>&1; do sleep 5; done'
```

**3c. Run benchmark**

```bash
MODEL_PATH=<MODEL_PATH> \
ISL=<ISL> OSL=<OSL> \
CONCURRENCY="<CON1> <CON2> <CON3>" \
LOG_DIR=$BENCH_DIR/<CONFIG>_mtp<0|1> \
bash bench.sh
```

For multiple ISL/OSL combinations, loop:

```bash
for ISL in 128 512 1024 2048; do
  for OSL in 128 512 1024; do
    ISL=$ISL OSL=$OSL \
    MODEL_PATH=<MODEL_PATH> \
    CONCURRENCY="1 16 64 128 256" \
    LOG_DIR=$BENCH_DIR/<CONFIG>_mtp<0|1> \
    bash bench.sh
  done
done
```

**3d. Stop server and repeat**

```bash
pkill -f "sglang.launch_server" || true
sleep 5
```

Repeat 3a–3d for each parallel config.

### Step 4: Report & Optimization Suggestions

After all configs are benchmarked, read the benchmark logs from each config directory, parse the metrics (TTFT, TPOT, throughput, etc.), and write a Markdown report directly to `$BENCH_DIR/benchmark_report.md`.

The report should include:

- Configuration summary (model, GPUs, mode, MTP status)
- Per-config results tables with all metrics + per-GPU throughput
- Cross-config comparison highlighting the best performer for each metric
- Optimization suggestions based on the patterns below

**Patterns to look for:**
- **Concurrency saturation** — throughput plateaus while latency degrades. Report the "knee" point.
- **Prefill vs decode bottleneck** — high TTFT = prefill-bound, high TPOT = decode-bound.
- **Per-GPU efficiency** — if per-GPU throughput drops at higher TP, communication overhead is the cost.
- **MTP impact** (if both runs exist) — MTP should primarily improve TPOT and output throughput.

**Suggest concrete next steps** (see `references/server_config.md` for flag details):
- Better TP/DP/EP ratio based on observed tradeoffs
- `--chunked-prefill-size` if prefill-bound
- `--mem-fraction-static` adjustment if OOM or underutilized
- Quantization, radix cache, schedule policy tuning
- AITER CK GEMM kernel tuning (`aiter-ck-gemm-tune` skill) for kernel-level optimization
- PD-disaggregation evaluation as a separate follow-up for production

Present the report to the user and walk them through the key findings.

## File Organization

```
/sgl-workspace/<model_short>_<YYYYMMDD>/
├── benchmark_report.md                          # final report
├── DP4EP4_mtp0/                                 # per-config directory
│   ├── server_DP4EP4_mtp0.log                   # sglang server log (from serve.sh)
│   ├── bench_ISL4096_OSL1024_CON64.log          # per-run benchmark log (from bench.sh)
│   ├── bench_ISL4096_OSL1024_CON128.log
│   └── ...
├── TP8_mtp0/
│   ├── server_TP8_mtp0.log
│   └── ...
└── DP8EP8_A2A_mtp1/
    └── ...
```

Each config gets its own directory. `serve.sh` writes `server_<LABEL>.log`, `bench.sh` writes `bench_ISL<X>_OSL<Y>_CON<Z>.log` — both into the same `LOG_DIR`.

## Important Notes

- This skill covers **mix mode only** (no PD-disaggregation). Prefill and decode run on the same GPUs.
- Always set `export SGLANG_USE_AITER=1` on AMD GPUs to enable AITER optimized kernels.
- `--random-range-ratio 1.0` ensures exact ISL/OSL lengths (no variation) for reproducible benchmarks.
- `bench.sh` uses `num_prompts = concurrency * 2` — this is handled by the script automatically.
- Between configs, fully kill the sglang server and wait for GPU memory to be freed before relaunching.
- If a benchmark run fails or hangs, check GPU memory usage with `rocm-smi` and server health with the `/health` endpoint.

