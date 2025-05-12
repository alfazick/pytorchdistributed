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


def allreduce_example(rank, world_size):
    setup(rank,world_size)
    tensor = torch.tensor([float(rank+1)],device = torch.device('cpu'))
    print(f"Rank {rank} has data: {tensor.item()}")

    # Perform the allreduce operation
    dist.all_reduce(tensor,op=dist.ReduceOp.SUM)

    print(f"After ALLReduce, rank {rank} has data: {tensor.item()}")

    # cleanup
    cleanup()


if __name__ == "__main__":
    world_size = 4
    mp.spawn(allreduce_example, args = (world_size,), nprocs=world_size, join=True)



