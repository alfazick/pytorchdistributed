import os
import torch
import torch.distributed as dist

def setup(rank,world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("gloo",rank = rank, world_size = world_size)

    print(f"Rank {rank}/{world_size} initialized")

def cleanup():
    dist.destroy_process_group()

def non_blocking_send_recv(rank, world_size):
    setup(rank, world_size)

    tensor_size = 10

    if rank == 0:
        # Process 0 sends data
        send_tensor = torch.randn(tensor_size)
        print(f"Rank {rank} sending tensor: {send_tensor}")

        # Non-blocking send returns a work handle
        req = dist.isend(tensor = send_tensor, dst = 1)

        # Do other work here while communication happens in background
        print(f"Rank {rank} doing other work while sending...")

        # Wait for the communication to complete
        req.wait()
        print(f"Rank {rank} send completed")

    elif rank == 1:
        # Process 1 receives data 
        recv_tensor = torch.zeros(tensor_size)

        # Non-blocking receive returns a work handle
        req = dist.irecv(tensor = recv_tensor, src = 0)

        # Do other work here while waiting for data
        print(f"Rank {rank} doing other work while receiving...")

        # Wait for the communication to complete
        req.wait()

        print(f"Rank {rank} received tensor: {recv_tensor}")

    # clean up
    cleanup()

import torch.multiprocessing as mp 

if __name__ == "__main__":
    world_size = 2
    mp.spawn(non_blocking_send_recv, args = (world_size,), nprocs = world_size, join = True)









