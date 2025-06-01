#!/bin/bash

# Progressive NCCL testing script
# Don't exit on first error - we want to see all results
set +e

SCRIPT="simple_nccl_test.py"
PYTHON_CMD="python3"

echo "=== Progressive NCCL Testing ==="

# Test basic CUDA first
echo "Testing basic CUDA..."
$PYTHON_CMD -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

# Test individual GPU access
echo -e "\nTesting individual GPU access..."
for gpu in 0 1; do
    echo "Testing GPU $gpu..."
    timeout 10s $PYTHON_CMD -c "
import torch
torch.cuda.set_device($gpu)
x = torch.ones(10).cuda()
print(f'GPU $gpu: Created tensor {x[:3]}')
" || echo "GPU $gpu failed"
done

# Test 2 GPU communication
echo -e "\nTesting 2 GPU NCCL communication..."
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=0
export NCCL_NVLS_ENABLE=0  # Disable NVLS to avoid CUDA errors with 4+ GPUs

# Kill any existing processes
pkill -f "12356" 2>/dev/null || true
sleep 2

echo "Launching 2 GPU test..."
pids=()
for rank in 0 1; do
    $PYTHON_CMD $SCRIPT $rank 2 > "test_2gpu_rank_${rank}.log" 2>&1 &
    pids+=($!)
done

# Wait for processes 
wait_failed=0
echo "Waiting for 2 GPU test completion..."
for i in "${!pids[@]}"; do
    if wait ${pids[$i]}; then
        echo "Rank $i (2GPU test): PASSED"
    else
        echo "Rank $i (2GPU test): FAILED"
        wait_failed=1
    fi
done

echo -e "\n=== 2 GPU Test Results ==="
for rank in 0 1; do
    echo "--- Rank $rank ---"
    cat "test_2gpu_rank_${rank}.log"
done

if [ $wait_failed -eq 0 ]; then
    echo -e "\n✓ 2 GPU test PASSED! Trying 4 GPUs..."
    
    # Test 4 GPU communication
    pkill -f "12356" 2>/dev/null || true
    sleep 2
    
    echo "Launching 4 GPU test..."
    pids=()
    for rank in 0 1 2 3; do
        $PYTHON_CMD $SCRIPT $rank 4 > "test_4gpu_rank_${rank}.log" 2>&1 &
        pids+=($!)
    done
    
    wait_failed=0
    echo "Waiting for 4 GPU test completion..."
    for i in "${!pids[@]}"; do
        if wait ${pids[$i]}; then
            echo "Rank $i (4GPU test): PASSED"
        else
            echo "Rank $i (4GPU test): FAILED"
            wait_failed=1
        fi
    done
    
    echo -e "\n=== 4 GPU Test Results ==="
    for rank in 0 1 2 3; do
        echo "--- Rank $rank ---"
        cat "test_4gpu_rank_${rank}.log"
    done
    
    if [ $wait_failed -eq 0 ]; then
        echo -e "\n✓ 4 GPU test PASSED! Trying 8 GPUs..."
        
        # Test 8 GPU communication
        pkill -f "12356" 2>/dev/null || true
        sleep 2
        
        echo "Launching 8 GPU test..."
        pids=()
        for rank in 0 1 2 3 4 5 6 7; do
            $PYTHON_CMD $SCRIPT $rank 8 > "test_8gpu_rank_${rank}.log" 2>&1 &
            pids+=($!)
        done
        
        wait_failed=0
        echo "Waiting for 8 GPU test completion..."
        for i in "${!pids[@]}"; do
            if wait ${pids[$i]}; then
                echo "Rank $i (8GPU test): PASSED"
            else
                echo "Rank $i (8GPU test): FAILED"
                wait_failed=1
            fi
        done
        
        echo -e "\n=== 8 GPU Test Results ==="
        for rank in 0 1 2 3 4 5 6 7; do
            echo "--- Rank $rank ---"
            cat "test_8gpu_rank_${rank}.log"
        done
        
        if [ $wait_failed -eq 0 ]; then
            echo -e "\n✓ 8 GPU test PASSED! NCCL works perfectly across all GPUs."
        else
            echo -e "\n✗ 8 GPU test FAILED. Issue occurs at 8 GPU scale."
        fi
    else
        echo -e "\n✗ 4 GPU test FAILED. Issue occurs at 4+ GPUs."
    fi
else
    echo -e "\n✗ 2 GPU test FAILED. Basic NCCL communication issue."
fi 