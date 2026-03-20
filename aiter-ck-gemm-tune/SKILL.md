---
name: aiter-ck-gemm-tune
description: >
  Tune AITER's CK GEMM kernels for better performance with specific model shapes.
  Use this skill whenever the user wants to tune, benchmark, or optimize CK GEMM kernels
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

The skill records benchmark results to log files so Step 2 (baseline) and Step 4 (after tuning) outputs can be compared. Store logs under `$AITER_PATH/tune_logs/` using this naming convention:

```
$AITER_PATH/tune_logs/<variant>_baseline_<YYYYMMDD_HHMMSS>.log   # Step 2 output
$AITER_PATH/tune_logs/<variant>_tuned_<YYYYMMDD_HHMMSS>.log      # Step 4 output
```

For example:
```
tune_logs/a8w8_blockscale_baseline_20260321_143022.log
tune_logs/a8w8_blockscale_tuned_20260321_160515.log
```

Create the `tune_logs/` directory if it doesn't exist. Use `tee` to write to the log file while also showing output to the user in real time.

## Workflow

Follow these steps in order. At each step, communicate clearly with the user about what is happening, what you found, and what you plan to do next.

---

### Step 0: Environment Discovery

Before anything else, establish the working environment:

1. **Locate aiter**: Ask the user for the aiter installation path, or try common locations (`~/projects/aiter`, `~/aiter`, etc.). Verify by checking that `csrc/` and `aiter/configs/` exist under that path.
2. **GPU info**: Run `rocm-smi` or `rocminfo` to determine `cu_num` (number of compute units) and GPU model. This matters because tuned configs are GPU-specific.
3. **Log location**: Ask the user where the inference logs are. These could be from sglang, vllm, or another framework. Logs could also be provided directly.
4. **Verify aiter installation**: Check if aiter is installed in dev mode. If not, warn the user that `python3 setup.py develop` from the aiter root may be needed.

---

### Step 1: Capture Shapes & Identify Kernel Type

The goal is to extract the unique (N, K) pairs that need tuning, and determine which kernel variant to tune.

#### Option A: Parse from aiter logs (preferred)

AITER itself logs untuned shapes with this pattern:

```
shape is M:<value>, N:<value>, K:<value>, not found tuned config in /tmp/aiter_configs/<variant>_tuned_gemm.csv, will use default config!
```

Parse these lines to extract:
- **Unique (N, K) pairs** — ignore M values, as M varies per forward pass
- **Kernel variant** — inferred from the CSV filename in the log (e.g., `a8w8_blockscale_tuned_gemm.csv` → variant is `a8w8_blockscale`)

Use grep/regex to find all matching lines, deduplicate by (N, K), and present the results to the user for confirmation.

#### Option B: Direct user input

The user provides (N, K) pairs and specifies the kernel variant directly.

#### Option C: Model config files

Pull shapes from existing model configs in `aiter/configs/model_configs/`. The user specifies which model and variant.

#### Generating M values for tuning

For each unique (N, K) pair, generate tuning rows by sweeping M as powers of 2:

```
M = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
```

This produces `15 × number_of_unique_NK_pairs` rows for the untuned CSV.

#### Write the untuned CSV

Write the generated shapes into the variant's untuned CSV file (e.g., `aiter/configs/a8w8_blockscale_untuned_gemm.csv`). The CSV format is simply:

```csv
M,N,K
1,12288,4096
2,12288,4096
4,12288,4096
...
16384,12288,4096
```

Present the full shape list to the user before writing.

---

### Step 2: Baseline Benchmark

Run the unit test for the target kernel variant with the **specific shapes from Step 1** to establish baseline performance **before** tuning. No rebuild is needed at this point.

The test scripts accept shapes via CLI arguments, but the argument format differs per variant. You must pass the captured shapes using the correct format:

#### CLI argument formats by variant

| Variant | Test File | Shape Args | Example |
|---------|-----------|------------|---------|
| `a8w8_blockscale` | `test_gemm_a8w8_blockscale.py` | `-m M1 M2 ... -nk N1,K1 N2,K2 ...` | `-m 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 -nk 12288,4096 24576,1536` |
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
  -m 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 \
  -nk 12288,4096 24576,1536 \
  2>&1 | tee tune_logs/a8w8_blockscale_baseline_$(date +%Y%m%d_%H%M%S).log
```

**Example for `a8w8` with (N,K) pair (12288,4096):**
```bash
cd $AITER_PATH
mkdir -p tune_logs
python3 op_tests/test_gemm_a8w8.py \
  -mnk 1,12288,4096 2,12288,4096 4,12288,4096 8,12288,4096 \
  16,12288,4096 32,12288,4096 64,12288,4096 128,12288,4096 \
  256,12288,4096 512,12288,4096 1024,12288,4096 2048,12288,4096 \
  4096,12288,4096 8192,12288,4096 16384,12288,4096 \
  2>&1 | tee tune_logs/a8w8_baseline_$(date +%Y%m%d_%H%M%S).log
```

Record the baseline log file path — you will need it in Step 4 for comparison.

If the variant has no test file (e.g., `a8w8_bpreshuffle`), inform the user and ask how they'd like to benchmark. If the test script's CLI format is unfamiliar or has changed, read the argparse section at the bottom of the test file to determine the correct arguments.

---

### Step 3: Tune

Run the tuning script for the target variant. The general command pattern is:

```bash
cd $AITER_PATH
python3 <tune_script> -i <untuned_csv> -o <tuned_csv> [options]
```

**Example for `a8w8_blockscale`:**
```bash
python3 csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py \
  -i aiter/configs/a8w8_blockscale_untuned_gemm.csv \
  -o aiter/configs/a8w8_blockscale_tuned_gemm.csv \
  --libtype both
```

#### Key flags to consider

| Flag | Default | Description |
|------|---------|-------------|
| `--libtype` | — | `ck`, `cktile`, or `both` (recommend `both` for best results) |
| `--mp N` | all GPUs | Number of parallel GPU processes |
| `--batch N` | 100 | Shapes per tuning batch |
| `--errRatio` | 0.05 | Error tolerance threshold |
| `-k` / `--splitK` | off | Enable split-K optimization |
| `--warmup N` | 5 | Warmup iterations before profiling |
| `--iters N` | 101 | Profiling iterations |
| `--timeout N` | none | Timeout in seconds per task group |
| `-v` | off | Verbose output |
| `--all` | off | Retune all shapes |

**Important warnings to communicate to the user:**
- Tuning can take a very long time (potentially hours) depending on the number of shapes and options
- Using `--libtype both` is slower but produces better results
- `--mp` can parallelize across GPUs to speed things up
- `--timeout` is recommended to prevent individual shapes from hanging

---

### Step 4: Rerun & Compare

After tuning completes, rebuild and rerun the benchmark with the **same shapes and CLI arguments from Step 2**, but with `AITER_REBUILD=1` to pick up the newly tuned kernels:

**Example for `a8w8_blockscale`:**
```bash
cd $AITER_PATH
AITER_REBUILD=1 python3 op_tests/test_gemm_a8w8_blockscale.py \
  -m 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 \
  -nk 12288,4096 24576,1536 \
  2>&1 | tee tune_logs/a8w8_blockscale_tuned_$(date +%Y%m%d_%H%M%S).log
```

**Example for `a8w8`:**
```bash
cd $AITER_PATH
AITER_REBUILD=1 python3 op_tests/test_gemm_a8w8.py \
  -mnk 1,12288,4096 2,12288,4096 4,12288,4096 8,12288,4096 \
  16,12288,4096 32,12288,4096 64,12288,4096 128,12288,4096 \
  256,12288,4096 512,12288,4096 1024,12288,4096 2048,12288,4096 \
  4096,12288,4096 8192,12288,4096 16384,12288,4096 \
  2>&1 | tee tune_logs/a8w8_tuned_$(date +%Y%m%d_%H%M%S).log
```

The `AITER_REBUILD=1` flag is essential — it forces aiter to rebuild kernels using the newly tuned CSV. Without it, old cached kernels will be used and you won't see improvements.

Use the exact same shape arguments as Step 2 so results are directly comparable. Refer to the CLI argument format table in Step 2 for the correct flags per variant.

**Compare results** by reading both log files (baseline from Step 2 and tuned from this step):
- For each shape (M, N, K), show before/after TFLOPS, latency (us), and speedup %
- Highlight shapes with significant improvement or regression
- Present a summary table to the user
- Tell the user where both log files are stored

Example comparison format:
```
Baseline log: tune_logs/a8w8_blockscale_baseline_20260321_143022.log
Tuned log:    tune_logs/a8w8_blockscale_tuned_20260321_160515.log

| M     | N     | K    | Before (TFLOPS) | After (TFLOPS) | Speedup |
|-------|-------|------|-----------------|----------------|---------|
| 128   | 12288 | 4096 | 85.2            | 125.4          | +47.2%  |
| 256   | 12288 | 4096 | 92.1            | 130.8          | +42.0%  |
```

---

## Troubleshooting

If anything fails at any step, check the variant's README at `$AITER_PATH/csrc/<kernel_dir>/README.md` — it contains variant-specific guidance, known issues, and examples.

Common issues:
- **JIT build fails**: The first run may take several minutes as kernels are built via JIT. Be patient.
- **`AITER_REBUILD=1` forgotten in Step 4**: Without this flag, old cached kernels will be used, and you won't see tuning improvements.
- **Stale builds with `PREBUILD_KERNELS=1`**: If aiter was installed with `PREBUILD_KERNELS=1`, you may need to remove `build/` and `*.so` in `aiter/jit/` and reinstall aiter to pick up new tuned kernels.
- **Tuning hangs on certain shapes**: Use `--timeout` to skip shapes that take too long.
- **Low accuracy (high errRatio)**: Tighten `--errRatio` (e.g., `0.01`) to filter out inaccurate kernel candidates.
