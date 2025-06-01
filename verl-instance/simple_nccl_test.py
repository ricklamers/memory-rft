#!/usr/bin/env python3

import torch
import torch.distributed as dist
import os
import sys
import signal
import time
from contextlib import contextmanager

@contextmanager
def timeout_context(seconds):
    """Context manager for timeout handling."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def test_simple_nccl(rank, world_size):
    """Simple NCCL test with timeout and better error handling."""
    try:
        print(f"Rank {rank}: Starting test...")
        
        # Set environment
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'
        os.environ['NCCL_NVLS_ENABLE'] = '0'  # Disable NVLS to avoid CUDA errors
        
        # Initialize process group with timeout
        with timeout_context(30):
            print(f"Rank {rank}: Initializing process group...")
            dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=torch.distributed.default_pg_timeout)
        
        print(f"Rank {rank}: Process group initialized")
        
        # Set device
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        
        print(f"Rank {rank}: Using device {device}")
        
        # Test simple tensor creation
        tensor = torch.ones(2).to(device) * rank
        print(f"Rank {rank}: Created tensor {tensor}")
        
        # Test allreduce with timeout
        with timeout_context(30):
            print(f"Rank {rank}: Starting allreduce...")
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            print(f"Rank {rank}: Allreduce completed, result: {tensor}")
        
        # Verify result - sum of all ranks: 0+1+2+...+(world_size-1)
        expected = sum(range(world_size))  # 0+1+2+...+(world_size-1)
        expected_tensor = torch.full_like(tensor, expected)
        if torch.allclose(tensor, expected_tensor):
            print(f"Rank {rank}: ✓ Test PASSED (expected {expected}, got {tensor[0].item()})")
            success = True
        else:
            print(f"Rank {rank}: ✗ Test FAILED (expected {expected}, got {tensor[0].item()})")
            success = False
        
        # Cleanup
        dist.destroy_process_group()
        return success
        
    except TimeoutError as e:
        print(f"Rank {rank}: TIMEOUT - {e}")
        return False
    except Exception as e:
        print(f"Rank {rank}: ERROR - {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python simple_nccl_test.py <rank> <world_size>")
        sys.exit(1)
    
    rank = int(sys.argv[1])
    world_size = int(sys.argv[2])
    
    success = test_simple_nccl(rank, world_size)
    print(f"Rank {rank}: Final result: {'SUCCESS' if success else 'FAILED'}")
    sys.exit(0 if success else 1) 