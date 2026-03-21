---
name: aiter-ck-gemm-tune
description: >
  Tune AITER's CK GEMM kernels for better performance with specific model shapes.
  Use this skill whenever the user wants to tune or optimize CK GEMM kernels
  in the AITER project. This includes tasks like: parsing inference logs for untuned GEMM shapes,
  running baseline benchmarks, tuning kernels for new shapes, comparing before/after performance,
  or any workflow involving aiter's ck_gemm tuning pipeline. Trigger this skill when the user
  mentions aiter gemm tuning, ck gemm performance, kernel tuning, untuned gemm shapes,
  or wants to optimize GEMM operations for specific model configurations.
---

# AITER CK GEMM Tune

A skill for tuning AITER's Composable Kernel (CK) GEMM kernels to achieve better performance for specific model shapes. The tuning workflow is a multi-step process: discover the environment, capture shapes, run baseline benchmarks, tune kernels, and compare results.

## Background

**AITER** (AI Tensor Engine for ROCm) is AMD's high-performance operator library for LLM inference on ROCm/AMD GPUs. It provides optimized kernels for common operations in transformer models — most critically, GEMM (General Matrix Multiply), which dominates the compute in LLM inference (linear projections, attention, MLP/FFN layers, MoE expert computations).

**Composable Kernel (CK)** is AMD's open-source library of GPU kernel primitives. CK provides templated, composable building blocks for writing high-performance GPU kernels. AITER uses CK to implement its GEMM kernels, with many kernel variants optimized for different quantization schemes (INT8, FP4, BF16) and memory layouts (blockscale, byte-pair reshuffle, batched, MoE).

**Why tuning matters:** Each CK GEMM kernel has many implementation variants (tile sizes, pipeline configurations, split-K strategies). The optimal variant depends on the specific GEMM shape (M, N, K) and the GPU hardware (number of compute units). AITER's tuning process benchmarks all candidate kernel configurations for each shape and selects the fastest one. Shapes come from specific model architectures — for example, a Llama 70B model produces different (N, K) pairs than a DeepSeek V3 model. The M dimension corresponds to the batch/token count and varies at runtime, so tuning sweeps M as powers of 2 to cover all realistic batch sizes.

**How it fits into the inference stack:** Inference frameworks like sglang and vllm call into AITER for their GEMM operations. When AITER encounters a shape that hasn't been tuned, it falls back to a default kernel configuration and logs a warning. The tuning workflow in this skill captures those untuned shapes and finds optimal kernel configurations for them.

## Supported Kernel Variants

Each variant follows the same tuning workflow pattern. The table below maps each variant to its key files (all paths relative to the aiter root):

| Variant | Tune Script | Untuned CSV | Tuned CSV | Test File | README |
|---------|-------------|-------------|-----------|-----------|--------|
| `a8w8` | `csrc/ck_gemm_a8w8/gemm_a8w8_tune.py` | `aiter/configs/a8w8_untuned_gemm.csv` | `aiter/configs/a8w8_tuned_gemm.csv` | `op_tests/test_gemm_a8w8.py` | `csrc/ck_gemm_a8w8/README.md` |
| `a8w8_blockscale` | `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py` | `aiter/configs/a8w8_blockscale_untuned_gemm.csv` | `aiter/configs/a8w8_blockscale_tuned_gemm.csv` | `op_tests/test_gemm_a8w8_blockscale.py` | `csrc/ck_gemm_a8w8_blockscale/README.md` |
| `a8w8_bpreshuffle` | `csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py` | `aiter/configs/a8w8_bpreshuffle_untuned_gemm.csv` | `aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv` | *(none)* | `csrc/ck_gemm_a8w8_bpreshuffle/README.md` |
| `a8w8_blockscale_bpreshuffle` | `csrc/ck_gemm_a8w8_blockscale_bpreshuffle/gemm_a8w8_blockscale_bpreshuffle_tune.py` | `aiter/configs/a8w8_blockscale_bpreshuffle_untuned_gemm.csv` | `aiter/configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv` | *(none)* | `csrc/ck_gemm_a8w8_blockscale_bpreshuffle/README.md` |
| `a4w4_blockscale` | `csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py` | `aiter/configs/a4w4_blockscale_untuned_gemm.csv` | `aiter/configs/a4w4_blockscale_tuned_gemm.csv` | `op_tests/test_gemm_a4w4.py` | `csrc/ck_gemm_a4w4_blockscale/README.md` |
| `batched_a8w8` | `csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py` | `aiter/configs/a8w8_untuned_batched_gemm.csv` | `aiter/configs/a8w8_tuned_batched_gemm.csv` | `op_tests/test_batched_gemm_a8w8.py` | `csrc/ck_batched_gemm_a8w8/README.md` |
| `batched_bf16` | `csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.py` | `aiter/configs/bf16_untuned_batched_gemm.csv` | `aiter/configs/bf16_tuned_batched_gemm.csv` | `op_tests/test_batched_gemm_bf16.py` | `csrc/ck_batched_gemm_bf16/README.md` |
| `moe_2stages` | `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py` | `aiter/configs/untuned_fmoe.csv` | `aiter/configs/tuned_fmoe.csv` | `op_tests/test_moe_2stage.py` | `csrc/ck_gemm_moe_2stages_codegen/README.md` |

## Log Files

The skill records outputs from Steps 2, 3, and 4 to log files under `$AITER_PATH/tune_logs/`. Use this naming convention:

```
$AITER_PATH/tune_logs/<variant>_bench_before_<YYYYMMDD_HHMMSS>.log  # Step 2: baseline benchmark
$AITER_PATH/tune_logs/<variant>_tuning_<YYYYMMDD_HHMMSS>.log        # Step 3: tuning process
$AITER_PATH/tune_logs/<variant>_bench_after_<YYYYMMDD_HHMMSS>.log   # Step 4: post-tune benchmark
```

For example:
```
tune_logs/a8w8_blockscale_bench_before_20260321_143022.log
tune_logs/a8w8_blockscale_tuning_20260321_150000.log
tune_logs/a8w8_blockscale_bench_after_20260321_160515.log
```

Create the `tune_logs/` directory if it doesn't exist. For interactive commands (Steps 2 and 4), use `2>&1 | tee <log>` to show output in real time while logging. For long-running background jobs (Step 3), redirect output to file directly (`> <log> 2>&1`).

## Workflow

Follow these steps in order. At each step, communicate clearly with the user about what is happening, what you found, and what you plan to do next.

---

### Step 0: Environment Discovery

Before anything else, establish the working environment. Tuning typically runs inside a **Docker container on a remote node** with AMD GPUs. Ask the user to provide access details upfront:

1. **Target environment access**: Ask the user how to reach the tuning environment:
   - **Node access**: How to SSH into the node (e.g., `ssh user@node-hostname`)
   - **Docker container**: The container name or ID to exec into (e.g., `docker exec -it <container_name> bash`)
   - If the user is already inside the target environment (local or already SSH'd in), that's fine too — just confirm.
   - All subsequent commands (Steps 1–4) should be run inside this environment.
2. **Locate aiter**: The pip package may be named `aiter` or `amd-aiter`, so use `pip list | grep -i aiter` to find the exact package name, then `pip show <package_name> | grep Location` to get its installed path. Do not guess common locations — there may be multiple aiter copies on the system, and only the one registered in pip is the active installation. Verify by checking that `csrc/` and `aiter/configs/` exist under that path.
3. **Log location**: Ask the user where the inference logs are. These could be from sglang, vllm, or another framework. Logs could also be provided directly. Logs may be on the node, inside the container, or on the user's local machine.
4. **Verify aiter installation**: Check if aiter is installed in dev mode. If not, warn the user that `python3 setup.py develop` from the aiter root may be needed.

---

### Step 1: Capture Shapes & Identify Kernel Type

The goal is to extract the unique (N, K) pairs that need tuning, and determine which kernel variant to tune.

#### Option A: Parse from aiter logs (preferred)

AITER itself logs untuned shapes with this pattern:

```
shape is M:<value>, N:<value>, K:<value>, not found tuned config in /tmp/aiter_configs/<variant>_tuned_gemm.csv, will use default config!
```

Use the bundled script `scripts/parse_untuned_shapes.py` to parse the log file. It extracts unique (N, K) pairs grouped by kernel variant.

**Step 1a: Run the parser to see what's in the log:**
```bash
python3 <skill_path>/scripts/parse_untuned_shapes.py <log_file>
```

This prints all variants found and their unique (N, K) pairs. A log may contain multiple kernel variants (e.g., both `a8w8_blockscale` and `a8w8` shapes).

**Step 1b: If multiple variants are found, ask the user which to tune.** They may want to tune one specific variant or all of them. Each variant must be tuned separately (different tune scripts, CSVs, and test files).

**Step 1c: Generate the untuned CSV for the chosen variant(s):**
```bash
# Single variant:
python3 <skill_path>/scripts/parse_untuned_shapes.py <log_file> --variant a8w8_blockscale --csv <output.csv> --m-sweep

# All variants (separate tuning runs needed per variant):
python3 <skill_path>/scripts/parse_untuned_shapes.py <log_file> --variant all --csv <output.csv> --m-sweep
```

Present the results to the user for confirmation before proceeding. If tuning multiple variants, repeat Steps 2–4 for each variant separately.

#### Option B: Direct user input

The user provides (N, K) pairs and specifies the kernel variant directly.

#### Generating M values for tuning

For each unique (N, K) pair, generate tuning rows by sweeping M as powers of 2:

```
M = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768
```

This produces `16 × number_of_unique_NK_pairs` rows for the untuned CSV.

**Note:** The M sweep for tuning (powers of 2) is separate from the M values used for benchmarking in Steps 2/4. Benchmarking typically uses the test script's default M list, which may include non-power-of-2 values (e.g., 96, 160, 224, etc.). This is normal — we tune with powers of 2 to cover the key points, but benchmark with a broader M range to see how the tuned kernels perform across all realistic batch sizes.

#### Write the untuned CSV

Write the generated shapes into the variant's untuned CSV file (e.g., `aiter/configs/a8w8_blockscale_untuned_gemm.csv`). The CSV format is simply:

```csv
M,N,K
1,12288,4096
2,12288,4096
4,12288,4096
...
32768,12288,4096
```

Present the full shape list to the user before writing.

---

### Step 2: Baseline Benchmark

Run the unit test for the target kernel variant with the **specific shapes from Step 1** to establish baseline performance **before** tuning. No rebuild is needed at this point.

#### Pre-benchmark checklist: ck_preshuffle

Some test scripts have a `--ck_preshuffle` or `--preshuffle` flag (currently only `a8w8_blockscale` and `moe_2stages`). The correct setting can be inferred from the kernel variant detected in Step 1:

- If the log shows `a8w8_blockscale_tuned_gemm.csv` (no "bpreshuffle" in the name) → use `--ck_preshuffle False`
- If the log shows `a8w8_blockscale_bpreshuffle_tuned_gemm.csv` → use `--ck_preshuffle True`

Mention the inferred setting in your response to the user for confirmation, but no need to ask them to specify — the log already tells you.

Other variants (e.g., `a8w8`, `a4w4_blockscale`, batched variants) do not have this flag — skip this for them.

#### Handling test script `choices` constraints

Test scripts may have argparse `choices` restrictions on `-m` and/or `-nk` that reject values not in their hardcoded lists. Before running, read the argparse section at the bottom of the test file to check for `choices` constraints. If the shapes or M values you need are not in the `choices` list, you must modify the test script:
- For `-m`: add missing M values (e.g., 16384, 32768) to both the `choices` and `default` lists.
- For `-nk`: remove the `choices` parameter entirely (keep `default`) so any (N,K) pair can be passed.

#### CLI argument formats by variant

| Variant | Test File | Shape Args | Example |
|---------|-----------|------------|---------|
| `a8w8_blockscale` | `test_gemm_a8w8_blockscale.py` | `-m M1 M2 ... -nk N1,K1 N2,K2 ...` | `-m 1 2 4 ... 32768 -nk 12288,4096 24576,1536` |
| `a8w8` | `test_gemm_a8w8.py` | `-mnk M1,N1,K1 M2,N2,K2 ...` | `-mnk 1,12288,4096 2,12288,4096 4,12288,4096` |
| `a4w4_blockscale` | `test_gemm_a4w4.py` | `-mnk M1,N1,K1 M2,N2,K2 ...` | `-mnk 1,12288,4096 2,12288,4096 4,12288,4096` |
| `batched_a8w8` | `test_batched_gemm_a8w8.py` | `-s M1,N1,K1 M2,N2,K2 ...` | `-s 1,12288,4096 2,12288,4096 4,12288,4096` |
| `batched_bf16` | `test_batched_gemm_bf16.py` | `-s M1,N1,K1 M2,N2,K2 ...` | `-s 1,12288,4096 2,12288,4096 4,12288,4096` |

For variants that use `-mnk` or `-s` (combined M,N,K tuples), generate all combinations of the M sweep with each (N,K) pair. For `a8w8_blockscale` which takes `-m` and `-nk` separately, pass all M values once and all (N,K) pairs once — the test script handles the cross product internally.

**Example for `a8w8_blockscale` with (N,K) pairs (12288,4096) and (24576,1536):**
```bash
cd $AITER_PATH
mkdir -p tune_logs
python3 op_tests/test_gemm_a8w8_blockscale.py \
  -m 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 \
  -nk 12288,4096 24576,1536 \
  --ck_preshuffle False \
  2>&1 | tee tune_logs/a8w8_blockscale_bench_before_$(date +%Y%m%d_%H%M%S).log
```

**Example for `a8w8` with (N,K) pair (12288,4096):**
```bash
cd $AITER_PATH
mkdir -p tune_logs
python3 op_tests/test_gemm_a8w8.py \
  -mnk 1,12288,4096 2,12288,4096 4,12288,4096 8,12288,4096 \
  16,12288,4096 32,12288,4096 64,12288,4096 128,12288,4096 \
  256,12288,4096 512,12288,4096 1024,12288,4096 2048,12288,4096 \
  4096,12288,4096 8192,12288,4096 16384,12288,4096 32768,12288,4096 \
  2>&1 | tee tune_logs/a8w8_bench_before_$(date +%Y%m%d_%H%M%S).log
```

Record the baseline log file path — you will need it in Step 4 for comparison.

If the variant has no test file (e.g., `a8w8_bpreshuffle`), inform the user and ask how they'd like to benchmark.

---

### Step 3: Tune

#### Check available GPUs

Before tuning, run `rocm-smi` to check how many GPUs are free. Use `--mp <num_free_gpus>` to parallelize tuning across all available GPUs — this can dramatically reduce tuning time (e.g., 8x faster with 8 GPUs vs 1).

```bash
rocm-smi --showuse | grep "GPU use"
```

#### Run the tuning script

The general command pattern is:

```bash
cd $AITER_PATH
python3 <tune_script> -i <untuned_csv> -o <tuned_csv> [options]
```

Tuning is a long-running job (potentially hours). Run it in the background with output redirected to a log file. Use `nohup` to ensure the process survives if the SSH session disconnects:

**Example for `a8w8_blockscale` with 8 free GPUs:**
```bash
cd $AITER_PATH
mkdir -p tune_logs
nohup python3 csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py \
  -i aiter/configs/a8w8_blockscale_untuned_gemm.csv \
  -o aiter/configs/a8w8_blockscale_tuned_gemm.csv \
  --libtype both --mp 8 --timeout 600 \
  > tune_logs/a8w8_blockscale_tuning_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

After launching, verify the process is running and monitor progress:
```bash
# Verify tuning process started (do NOT rely on $! — it doesn't work reliably through docker exec layers)
ps aux | grep tune.py | grep -v grep

# Monitor progress by tailing the log file
tail -f tune_logs/a8w8_blockscale_tuning_*.log
```

#### Key flags to consider

| Flag | Default | Description |
|------|---------|-------------|
| `--libtype` | — | `ck`, `cktile`, or `both` (recommend `both` for best results) |
| `--mp N` | all GPUs | Number of parallel GPU processes — set to number of free GPUs |
| `--batch N` | 100 | Shapes per tuning batch |
| `--errRatio` | 0.05 | Error tolerance threshold |
| `-k` / `--splitK` | off | Enable split-K optimization |
| `--warmup N` | 5 | Warmup iterations before profiling |
| `--iters N` | 101 | Profiling iterations |
| `--timeout N` | none | Timeout in seconds per task group (recommend `600`) |
| `-v` | off | Verbose output |
| `--all` | off | Retune all shapes |

**Important warnings to communicate to the user:**
- Tuning can take a very long time (potentially hours) depending on the number of shapes and options
- Using `--libtype both` is slower but produces better results
- Use `--mp` with all available GPUs to maximize parallelism
- `--timeout` is recommended to prevent individual shapes from hanging
- The first run includes a JIT compilation step that can take several minutes before actual tuning begins

---

### Step 4: Rerun & Compare

After tuning completes, rerun the benchmark to measure improvement. **Reuse the exact same command from Step 2** with only two changes:
1. Prepend `AITER_REBUILD=1` to force aiter to rebuild kernels using the newly tuned CSV
2. Change the log filename from `bench_before` to `bench_after`

This ensures the same shapes, M values, and flags are used for an apples-to-apples comparison. Do not re-type the command manually — copy the Step 2 command and apply the two changes above.

**Example — if Step 2 command was:**
```bash
python3 op_tests/test_gemm_a8w8_blockscale.py \
  -m 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 \
  -nk 512,4096 4096,256 8192,4096 12288,4096 17408,4096 \
  --ck_preshuffle False \
  2>&1 | tee tune_logs/a8w8_blockscale_bench_before_20260321_143022.log
```

**Then Step 4 command is:**
```bash
AITER_REBUILD=1 python3 op_tests/test_gemm_a8w8_blockscale.py \
  -m 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 \
  -nk 512,4096 4096,256 8192,4096 12288,4096 17408,4096 \
  --ck_preshuffle False \
  2>&1 | tee tune_logs/a8w8_blockscale_bench_after_$(date +%Y%m%d_%H%M%S).log
```

The `AITER_REBUILD=1` flag is essential — without it, old cached kernels will be used and you won't see improvements. The first run after tuning will take extra time for JIT rebuilding.

**Compare results** using the bundled comparison script:

```bash
python3 <skill_path>/scripts/compare_results.py \
  tune_logs/<variant>_bench_before_<timestamp>.log \
  tune_logs/<variant>_bench_after_<timestamp>.log
```

The script parses both log files, matches shapes by (M, N, K), and produces:
- A per-shape comparison table with before/after TFLOPS and speedup %
- A summary with average/min/max speedup and improved/regressed counts
- A per-(N, K) breakdown

You can also compare a different metric with `--metric "ck us"` (latency) or `--metric "asm TFLOPS"`.

Present the comparison results to the user and tell them where both log files are stored.

---

### Step 5: Generate Report

After completing the comparison, generate a tuning report and save it to `$AITER_PATH/tune_logs/<variant>_report_<YYYYMMDD_HHMMSS>.md`. The report should contain:

1. **Environment summary**: GPU model, aiter version, aiter path
2. **Shapes tuned**: the (N, K) pairs and kernel variant
3. **Tuning configuration**: flags used (`--libtype`, `--mp`, `--timeout`, etc.)
4. **Full comparison table**: the complete output from `compare_results.py` — include every shape, not a summary. This is the primary content of the report.
5. **Summary statistics**: average/min/max speedup, improved/regressed counts, per-(N,K) breakdown grouped by M category:
   - **Small M (1-63)**: decode-like workloads
   - **Medium M (64-512)**: mixed workloads
   - **Large M (>512)**: prefill-like workloads
6. **Log file locations**: paths to all log files (bench_before, tuning, bench_after)

Generate the report by running the comparison script and capturing its output:

```bash
python3 <skill_path>/scripts/compare_results.py \
  tune_logs/<variant>_bench_before_<timestamp>.log \
  tune_logs/<variant>_bench_after_<timestamp>.log \
  > /tmp/compare_output.txt
```

Then assemble the full report as a markdown file. Save the report in two locations:
1. **Remote**: `$AITER_PATH/tune_logs/<variant>_report_<YYYYMMDD_HHMMSS>.md` (inside the tuning environment, alongside the log files)
2. **Local**: a copy in the user's current working directory or a location they specify

Present the report to the user and tell them where both copies are saved.

---

## Troubleshooting

If anything fails at any step, check the variant's README at `$AITER_PATH/csrc/<kernel_dir>/README.md` — it contains variant-specific guidance, known issues, and examples.

Common issues:
- **JIT build fails**: The first run may take several minutes as kernels are built via JIT. Be patient.
- **`AITER_REBUILD=1` forgotten in Step 4**: Without this flag, old cached kernels will be used, and you won't see tuning improvements.
- **Stale builds with `PREBUILD_KERNELS=1`**: If aiter was installed with `PREBUILD_KERNELS=1`, you may need to remove `build/` and `*.so` in `aiter/jit/` and reinstall aiter to pick up new tuned kernels.
- **Tuning hangs on certain shapes**: Use `--timeout` to skip shapes that take too long.
- **Low accuracy (high errRatio)**: Tighten `--errRatio` (e.g., `0.01`) to filter out inaccurate kernel candidates.
