#!/bin/bash
#
# Stop sglang server and verify GPU memory is freed
#
# Usage:
#   bash stop.sh
#

set -eu

echo "Stopping sglang server..."

# Kill all sglang processes
ps -ef | grep -i sglang | grep -v grep | awk '{print $2}' | xargs kill -9

sleep 6

# Verify no GPU processes and memory is freed
echo "Checking GPU processes:"
rocm-smi --showpids 2>/dev/null || true
echo ""
echo "GPU VRAM usage:"
rocm-smi 2>/dev/null || true

echo "Done."
