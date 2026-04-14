#!/bin/bash
#
# SGLang Benchmark Runner
#
# Usage:
#   MODEL_PATH=/data/models/DeepSeek-R1 LOG_DIR=/sgl-workspace/bench/logs bash bench.sh
#
# Override defaults with env vars:
#   ISL=2048 OSL=512 PORT=8000 CONCURRENCY="32 64 128" MODEL_PATH=... LOG_DIR=... bash bench.sh
#

set -euo pipefail

export PYTHONPATH=/sgl-workspace/sglang/python:${PYTHONPATH:-}

: "${MODEL_PATH:?Error: MODEL_PATH must be set}"
: "${LOG_DIR:?Error: LOG_DIR must be set}"

ISL="${ISL:-4096}"
OSL="${OSL:-1024}"
RATIO="${RATIO:-1.0}"
PORT="${PORT:-30000}"
CONCURRENCY="${CONCURRENCY:-64 128}"

mkdir -p "$LOG_DIR"

echo "======================================================================"
echo "SGLang Benchmark"
echo "  Model:       ${MODEL_PATH}"
echo "  ISL:         ${ISL}"
echo "  OSL:         ${OSL}"
echo "  Ratio:       ${RATIO}"
echo "  Concurrency: ${CONCURRENCY}"
echo "  Log dir:     ${LOG_DIR}"
echo "  Started:     $(date -Iseconds)"
echo "======================================================================"

for CON in $CONCURRENCY; do
  PROMPTS=$((CON * 2))
  LOG="${LOG_DIR}/bench_ISL${ISL}_OSL${OSL}_CON${CON}.log"

  echo ""
  echo "[$(date -Iseconds)] ISL=${ISL} OSL=${OSL} CON=${CON} PROMPTS=${PROMPTS} -> ${LOG}"

  python3 -m sglang.bench_serving \
    --dataset-name random \
    --model "$MODEL_PATH" \
    --random-input-len "$ISL" \
    --random-output-len "$OSL" \
    --num-prompt "$PROMPTS" \
    --random-range-ratio "$RATIO" \
    --max-concurrency "$CON" \
    --host 0.0.0.0 \
    --port "$PORT" \
    2>&1 | tee "$LOG"
done

echo ""
echo "Done at $(date -Iseconds). Logs: ${LOG_DIR}"
