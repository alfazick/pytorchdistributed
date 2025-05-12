import os
import torch
import torch.distributed as dist 
import torch.multiprocessing as mp 

def setup(rank, world_size):
    """
    Initialize the distributed environment.
    
    Args:
        rank: Unique identifier for the process
        world_size: Total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
def cleanup():
    """
    Clean up the distributed environment.
    """
    dist.destroy_process_group()


def ring_allreduce(rank,world_size):
    """ Ring AllReduce algorithm"""

    setup(rank,world_size)

    tensor_size = 4
    tensor = torch.ones(tensor_size) * (rank + 1)

    print(f"Rank {rank} initial data: {tensor}")

    # Make a copy for verification
    original_tensor = tensor.clone()

    # Allocate memory for receiving data
    recv_buffer = torch.zeros_like(tensor)

    # Calculate the expected result for verification
    expected_result = sum([(i + 1) for i in range(world_size)]) * torch.ones_like(tensor)

    print(expected_result)

# TODO

if __name__ == "__main__":
    world_size = 4
    mp.spawn(ring_allreduce, args=(world_size,), nprocs=world_size, join=True)




