import time
import torch
import os
from loguru import logger

def get_total_and_free_memory_in_Mb(cuda_device):
    devices_info_str = os.popen(
        "nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader"
    )
    devices_info = devices_info_str.read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(",")
    return int(total), int(used)

def occupy_mem(cuda_device, mem_ratio=0.9):
    """
    pre-allocate gpu memory for training to avoid memory Fragmentation.
    """
    total, used = get_total_and_free_memory_in_Mb(cuda_device)
    max_mem = int(total * mem_ratio)
    block_mem = max_mem - used
    logger.info(f'Occupy memory for {block_mem} MB')
    x = torch.cuda.FloatTensor(256, 1024, block_mem)
    del x
    time.sleep(5)