#!/bin/bash

# NCCL Health Check Launcher for 8 GPUs
set -e

WORLD_SIZE=8
SCRIPT="nccl_health_check.py"

echo "Starting NCCL health check across $WORLD_SIZE GPUs..."

# Check if the script exists
if [ ! -f "$SCRIPT" ]; then
    echo "Error: $SCRIPT not found!"
    exit 1
fi

# Make sure we have python3
PYTHON_CMD=$(which python3 2>/dev/null || echo "")
if [ -z "$PYTHON_CMD" ]; then
    echo "Error: python3 not found!"
    exit 1
fi

echo "Using Python: $PYTHON_CMD"
echo "PyTorch version: $($PYTHON_CMD -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $($PYTHON_CMD -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA device count: $($PYTHON_CMD -c 'import torch; print(torch.cuda.device_count())')"

# Set environment variables for better debugging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export CUDA_LAUNCH_BLOCKING=1

# Kill any existing processes on the master port
pkill -f "12355" 2>/dev/null || true
sleep 2

echo "Starting processes..."

# Launch all processes in background
pids=()
for rank in $(seq 0 $((WORLD_SIZE-1))); do
    echo "Launching rank $rank..."
    CUDA_VISIBLE_DEVICES=$rank $PYTHON_CMD $SCRIPT $rank $WORLD_SIZE > "nccl_test_rank_${rank}.log" 2>&1 &
    pids+=($!)
    sleep 0.5  # Small delay between launches
done

echo "All processes launched. PIDs: ${pids[@]}"
echo "Waiting for completion..."

# Wait for all processes to complete
failed=0
for i in "${!pids[@]}"; do
    pid=${pids[$i]}
    if wait $pid; then
        echo "Rank $i completed successfully"
    else
        echo "Rank $i failed!"
        failed=1
    fi
done

echo ""
echo "=== RESULTS ==="
for rank in $(seq 0 $((WORLD_SIZE-1))); do
    echo "--- Rank $rank log ---"
    cat "nccl_test_rank_${rank}.log"
    echo ""
done

if [ $failed -eq 0 ]; then
    echo "✓ All NCCL tests PASSED!"
    exit 0
else
    echo "✗ Some NCCL tests FAILED!"
    exit 1
fi 