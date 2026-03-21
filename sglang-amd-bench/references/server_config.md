# SGLang Server Configuration & Parallel Strategy

This is the reference for sglang server launch configuration, parallelism strategies, and how to reason about the right config for a given model and GPU setup.

## GPU HBM Reference

| GPU | HBM per GPU | Architecture |
|-----|-------------|-------------|
| MI355X | 288 GB | gfx950 |
| MI300X | 192 GB | gfx942 |
| MI308X | 192 GB | gfx942 |

## Parallel Config Naming Convention

For **dense models**, config is simply TP/DP (e.g., `TP8`, `TP4_DP2`).

For **MoE models**, the convention is `<attention_parallel>+<MoE_parallel>` because the attention layers and MoE expert layers can use different parallelism:

- `TP8+TP8` — TP=8 for both attention and MoE
- `TP4+EP4` — TP=4 for attention, EP=4 for MoE experts
- `DP4+EP4` — DP-attention=4 for attention, EP=4 for MoE experts
- `DP4+TP4` — DP-attention=4 for attention, TP=4 for MoE

## Parallelism Types

### TP (Tensor Parallelism)

Splits each layer's weight tensors across GPUs. Every GPU participates in computing every token, then results are all-reduced.

- **Effect**: Reduces per-token latency (faster single-request), but adds all-reduce communication overhead per layer.
- **sglang flag**: `--tp <N>`
- **Use for**: Attention layers (if KV heads >= TP), MoE layers (as alternative to EP).

### DP (Data Parallelism / DP-Attention)

For **dense models**: Creates multiple independent replicas. Each replica handles different requests.
- **sglang flag**: `--dp <N>` (creates N replicas, each with TP = total_gpus / N)

For **MoE models**: DP means **dp-attention** — parallelizes the attention computation across GPUs while MoE layers use a different strategy (usually EP).
- **sglang flag**: `--dp <N>` (dp-attention for MoE)
- **Key rule**: DP-attention is most commonly paired with EP. Can also pair with TP for MoE. Do NOT combine TP+DP for the attention side unless user explicitly asks.

### EP (Expert Parallelism)

Only for MoE models. Distributes experts across GPUs.

- **sglang flag**: `--ep <N>`
- **Two modes** (must choose one):

| EP Mode | Communication | When to use |
|---------|--------------|-------------|
| **All-to-All EP** | All-to-all | Each GPU holds a subset of experts. Tokens routed to the GPU owning the target expert. |
| **All-Reduce EP** | All-reduce | Each GPU computes on a subset of experts, results combined via all-reduce. |

**Always ask the user which EP mode they want.** The best mode depends on expert count, network topology, and batch size.

## Reading Model Architecture from config.json

To make informed config suggestions, read the model's `config.json` from the weights directory. Do not rely on memorized numbers — always check the actual model.

```bash
cat <MODEL_WEIGHTS_PATH>/config.json | python3 -c "
import json, sys
c = json.load(sys.stdin)
print('Architecture fields:')
# Attention
print(f'  num_attention_heads (Q heads): {c.get(\"num_attention_heads\", \"N/A\")}')
print(f'  num_key_value_heads (KV heads): {c.get(\"num_key_value_heads\", \"N/A\")}')
# MoE
print(f'  num_experts: {c.get(\"num_local_experts\", c.get(\"n_routed_experts\", \"N/A (dense model)\"))}')
print(f'  num_experts_per_tok: {c.get(\"num_experts_per_tok\", c.get(\"num_selected_experts\", \"N/A\"))}')
# Size
print(f'  hidden_size: {c.get(\"hidden_size\", \"N/A\")}')
print(f'  num_hidden_layers: {c.get(\"num_hidden_layers\", \"N/A\")}')
# Attention type hints
if 'qk_nope_head_dim' in c or 'kv_lora_rank' in c:
    print('  attention_type: MLA (Multi-Latent Attention)')
elif c.get('num_key_value_heads', 0) < c.get('num_attention_heads', 0):
    print('  attention_type: GQA (Grouped-Query Attention)')
else:
    print('  attention_type: MHA (Multi-Head Attention)')
"
```

## How to Reason About Parallel Config

Use the config.json values and GPU specs to analyze these factors:

### 1. Weight size vs GPU memory — which TP values fit?

`weight_size / TP` must be less than GPU HBM (use ~70-80% of HBM for weights, rest for KV cache + activations).

Estimate weight size: for FP8 models, roughly `total_params_in_billions * 1 GB` (1 byte per param). For BF16, `total_params * 2 GB`.

Example: 400GB FP8 model on MI355X (288GB): TP=2 (200GB/GPU), TP=4 (100GB/GPU), TP=8 (50GB/GPU) — all fit.

### 2. Attention type and KV heads — TP vs DP for attention

Read `num_key_value_heads` and detect attention type from config.json.

- **MLA** (detected by `qk_nope_head_dim` or `kv_lora_rank` in config): Compresses KV into a latent representation (effectively 1 KV head). High TP for attention wastes KV cache — each GPU replicates the full KV cache. **DP-attention is strongly preferred for MLA models.**
- **GQA** (`num_key_value_heads` < `num_attention_heads`): If `num_key_value_heads >= TP`, TP is efficient. If `num_key_value_heads < TP`, KV cache gets replicated wastefully on some GPUs.
- **MHA** (`num_key_value_heads == num_attention_heads`): TP splits heads evenly. Usually fine up to TP = num_heads.

### 3. MoE vs Dense — which parallelism strategies apply?

Read `num_local_experts` or `n_routed_experts` from config.json.

- **Dense models** (no expert fields): Only TP/DP. Higher TP = lower latency; higher DP = more replicas = higher throughput.
- **MoE models**: DP+EP is the most popular config. EP distributes experts. TP can also work for MoE layers but is less common. Expert count determines EP granularity (EP must evenly divide expert count).

### 4. Putting it together — example analyses

**MLA MoE model (e.g., DeepSeek-R1) on 8x MI355X:**
- MLA → DP-attention preferred (not TP for attention)
- MoE → EP for experts
- Suggestion: DP8+EP8, DP4+EP4, plus TP8+TP8 as baseline

**GQA MoE model (e.g., Qwen3.5-397B) on 8x MI355X:**
- GQA with N KV heads → check if TP <= KV heads for attention efficiency
- MoE → DP+EP most popular
- ~400GB FP8 → TP2/TP4/TP8 all fit
- Suggestion: DP+EP configs, plus TP+EP hybrid

**GQA Dense model (e.g., Llama-70B) on 8x MI355X:**
- ~70GB FP8 → fits 1 GPU
- 8 KV heads → TP up to 8 efficient
- Suggestion: TP1_DP8, TP2_DP4, TP4_DP2, TP8

## Configuration Constraints

- **TP * DP** must equal total GPU count (for dense models without EP)
- **TP * EP** or **DP * EP** must equal total GPU count (for MoE with EP)
- EP must evenly divide the model's expert count
- EP is only meaningful for MoE models
- For MoE: attention and MoE layers can use different parallelism
- DP-attention is most commonly paired with EP, but can also pair with TP for MoE
- Do NOT combine TP+DP for the attention side unless user explicitly asks
- For dense models, DP creates independent replicas

## Server Launch Flags Reference

### Core flags

| Flag | Description | Default |
|------|------------|---------|
| `--model-path <PATH>` | HuggingFace model ID or local path to weights | required |
| `--tp <N>` | Tensor parallelism degree | 1 |
| `--dp <N>` | Data parallelism / dp-attention degree | 1 |
| `--ep <N>` | Expert parallelism degree (MoE only) | 1 |
| `--host <ADDR>` | Listen address | `127.0.0.1` |
| `--port <PORT>` | Listen port | `30000` |
| `--mem-fraction-static <F>` | Fraction of GPU memory for static allocation (weights + KV cache) | `0.88` |
| `--trust-remote-code` | Allow running model-defined code | - |

### Performance flags

| Flag | Description | When to use |
|------|------------|-------------|
| `--enable-mtp` | Enable Multi-Token Prediction | MTP-capable models (DeepSeek, Qwen3) — ask user |
| `--quantization fp8` | Use FP8 quantization | When model is not already FP8 |
| `--chunked-prefill-size <N>` | Max prefill tokens per chunk | When TTFT is high at high concurrency |
| `--disable-radix-cache` | Disable radix attention cache | For benchmarking cache impact |
| `--schedule-policy <P>` | Scheduling policy: `lpm`, `fcfs`, `random` | For comparing scheduling strategies |

### Multi-node flags

| Flag | Description |
|------|------------|
| `--dist-init-addr <IP>:<PORT>` | Head node address for distributed init |
| `--nnodes <N>` | Total number of nodes |
| `--node-rank <R>` | This node's rank (0 for head) |

### MTP note

MTP uses additional GPU memory for the extra prediction heads. When enabling MTP, consider reducing `--mem-fraction-static` (e.g., from 0.85 to 0.80) to leave room.

## Server Launch Template

```bash
export SGLANG_USE_AITER=1

python3 -m sglang.launch_server \
  --model-path <MODEL_PATH> \
  --tp <TP> \
  --dp <DP> \
  --ep <EP> \
  --host 0.0.0.0 \
  --port 8000 \
  --mem-fraction-static 0.85 \
  --trust-remote-code \
  2>&1 | tee <LOG_PATH>
```

Omit `--dp` if DP=1, `--ep` if EP=1. Add `--enable-mtp` if user opted in.

### Multi-node launch

```bash
# On each node (adjust --node-rank):
python3 -m sglang.launch_server \
  --model-path <MODEL_PATH> \
  --tp <TOTAL_TP_ACROSS_NODES> \
  --dist-init-addr <HEAD_IP>:5000 \
  --nnodes <NUM_NODES> \
  --node-rank <RANK> \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code
```

## AMD-Specific Environment Variables

Always set these when benchmarking on AMD Instinct GPUs:

```bash
# Required: enable AITER optimized kernels
export SGLANG_USE_AITER=1

# Multi-node NCCL settings (adjust to your network)
export GLOO_SOCKET_IFNAME=<network_interface>     # e.g., enp193s0f1np1
export NCCL_IB_HCA=ionic_0,ionic_1,...,ionic_7     # InfiniBand HCAs
export NCCL_IB_TC=104
export NCCL_IB_FIFO_TC=184
export NCCL_IB_GID_INDEX=1
export NCCL_CROSS_NIC=0
```
