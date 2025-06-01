#!/usr/bin/env python3

import torch
import torch.distributed as dist
import os
import sys
import time

def setup_distributed(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set the GPU for this process
    torch.cuda.set_device(rank)
    
    print(f"Process {rank}/{world_size} initialized on GPU {torch.cuda.current_device()}")

def test_nccl_communication(rank, world_size):
    """Test basic NCCL operations."""
    # Create a tensor on GPU
    device = torch.device(f"cuda:{rank}")
    tensor = torch.ones(10).to(device) * rank
    
    print(f"Rank {rank}: Initial tensor = {tensor[:5].tolist()}")
    
    # Test allreduce
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected_sum = sum(range(world_size)) * 10  # Each rank contributes rank * 10
    
    print(f"Rank {rank}: After allreduce = {tensor[:5].tolist()}")
    
    # Verify result
    if torch.allclose(tensor, torch.tensor(expected_sum).to(device)):
        print(f"Rank {rank}: ✓ Allreduce test PASSED")
    else:
        print(f"Rank {rank}: ✗ Allreduce test FAILED")
        return False
    
    # Test broadcast
    broadcast_tensor = torch.zeros(5).to(device)
    if rank == 0:
        broadcast_tensor.fill_(42.0)
    
    dist.broadcast(broadcast_tensor, src=0)
    
    if torch.allclose(broadcast_tensor, torch.tensor(42.0).to(device)):
        print(f"Rank {rank}: ✓ Broadcast test PASSED")
    else:
        print(f"Rank {rank}: ✗ Broadcast test FAILED")
        return False
    
    return True

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def run_test(rank, world_size):
    """Run the complete NCCL test."""
    try:
        print(f"Starting NCCL test on rank {rank}")
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            print(f"Rank {rank}: CUDA not available!")
            return False
        
        print(f"Rank {rank}: CUDA device count = {torch.cuda.device_count()}")
        
        # Setup distributed
        setup_distributed(rank, world_size)
        
        # Test communication
        success = test_nccl_communication(rank, world_size)
        
        # Cleanup
        cleanup()
        
        if success:
            print(f"Rank {rank}: ✓ ALL TESTS PASSED")
        else:
            print(f"Rank {rank}: ✗ SOME TESTS FAILED")
        
        return success
        
    except Exception as e:
        print(f"Rank {rank}: Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python nccl_health_check.py <rank> <world_size>")
        sys.exit(1)
    
    rank = int(sys.argv[1])
    world_size = int(sys.argv[2])
    
    success = run_test(rank, world_size)
    sys.exit(0 if success else 1) 