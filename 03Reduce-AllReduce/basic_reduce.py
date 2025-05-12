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


def reduce_example(rank,world_size):
    setup(rank,world_size)

    # Create a tensor with a value based on the rank
    tensor = torch.tensor([float(rank+1)], device = torch.device('cpu'))
    print(f"Rank {rank} has data: {tensor.item()}")

    # Define the root rank where results will be collected
    root_rank = 0

    if rank == root_rank:
        reduced_tensor = torch.zeros_like(tensor)
    else:
        reduced_tensor = None 

    # Perform the reduce operation
    dist.reduce(tensor, dst=root_rank, op = dist.ReduceOp.SUM)

    if rank == root_rank:
        print(f"Rank {rank} has reduced result: {tensor.item()}")

    # Clean up
    cleanup()

if __name__ == "__main__":
    world_size = 4

    mp.spawn(reduce_example, args = (world_size,), nprocs=world_size, join=True)



