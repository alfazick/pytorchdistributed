import os
import torch
import torch.distributed as dist 

def setup(rank,world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'


    dist.init_process_group("gloo",rank=rank,world_size=world_size)
    print(f"Rank {rank}/{world_size} initialized")


def cleanup():
    dist.destroy_process_group()

# simple_send_receive.py
# poetry run python3 simple_send_receive.py
import torch.multiprocessing as mp 

def simple_send_recv(rank,world_size):
    setup(rank,world_size)

    tensor_size = 10

    if rank == 0:
        # Process 0 sends data
        tensor = torch.randn(tensor_size)
        print(f"Rank {rank} sending tensor: {tensor}")
        dist.send(tensor=tensor,dst=1)
    elif rank ==1:
        #Process 1 receives data
        tensor = torch.zeros(tensor_size)
        dist.recv(tensor=tensor,src=0)
        print(f"Rank {rank} received tensor: {tensor}")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(simple_send_recv,args=(world_size,),nprocs=world_size,join=True)

