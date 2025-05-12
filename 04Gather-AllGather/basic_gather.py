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

def run_gather_example(rank,world_size):
    setup(rank,world_size)

    # Create a tensor with values specific to this rank
    tensor = torch.tensor([rank]*5, dtype = torch.float32)
    print(f"Rank {rank} has data: {tensor}")

    # Create a list to gather results(only on destination rank)
    if rank == 0:
        gather_list = [torch.zeros(5,dtype=torch.float32) for _ in range(world_size)]
    else:
        gather_list = None

    # Perform gather operation
    # Note: this is one-to-one communication
    dist.gather(tensor, gather_list, dst = 0)

    # Print results (only on destination rank)
    if rank == 0:
        print(f"Rank 0 gathered data: {gather_list}")

    cleanup()


if __name__ == "__main__":
    world_size = 4
    mp.spawn(run_gather_example, args=(world_size,), nprocs=world_size)

