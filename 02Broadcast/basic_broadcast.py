import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank,world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("gloo",rank = rank, world_size = world_size)

    print(f"Rank {rank}/{world_size} initialized")

def cleanup():
    dist.destroy_process_group()


def broadcast(rank,world_size):
    setup(rank, world_size)

    if rank == 0:
        tensor = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    else:
        tensor = torch.zeros(5, dtype=torch.float32)

    print(f"[rank {rank}] before: {tensor}")
    dist.broadcast(tensor, src=0)
    print(f"[rank {rank}]  after: {tensor}")

    cleanup()

if __name__ == "__main__":
    world_size = 3
    mp.spawn(broadcast, args = (world_size,), nprocs = world_size, join = True)




