from torch import distributed as dist
from contextlib import contextmanager

_LOCAL_PROCESS_GROUP = None

def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_local_rank(single_gpu_id:int=None) -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if _LOCAL_PROCESS_GROUP is None:
        return get_rank()

    if single_gpu_id is None:
        single_gpu_id = 0
    
    if not dist.is_available():
        return single_gpu_id
    if not dist.is_initialized():
        return single_gpu_id
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)

@contextmanager
def wait_for_the_master(local_rank: int):
    """
    Make all processes waiting for the master to do some task.
    """
    if local_rank > 0:
        dist.barrier()
    yield
    if local_rank == 0:
        if not dist.is_available():
            return
        if not dist.is_initialized():
            return
        else:
            dist.barrier()