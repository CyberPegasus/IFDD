import os
import cv2
from loguru import logger

def get_num_devices():
    gpu_list = os.getenv('CUDA_VISIBLE_DEVICES', None)
    if gpu_list is not None:
        if '.' in gpu_list:
            return len(gpu_list.split('.'))
        else:
            return len(gpu_list.split(','))
    else:
        devices_list_info = os.popen("nvidia-smi -L")
        devices_list_info = devices_list_info.read().strip().split("\n")
        return len(devices_list_info)

def configure_module(ulimit_value=8192):
    """
    Configure pytorch module environment. setting of ulimit and cv2 will be set.
    Soft RLIMIT_NOFILE is reset by ulimit_value

    Args:
        ulimit_value(int): default open file number on linux. Default value: 8192.
    """
    # system setting
    try:
        import resource

        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (ulimit_value, rlimit[1]))
    except Exception:
        # Exception might be raised in Windows OS or rlimit reaches max limit number.
        # However, set rlimit value might not be necessary.
        pass

    # cv2
    # multiprocess might be harmful on performance of torch dataloader
    os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
    try:
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        # cv2 version mismatch might rasie exceptions.
        pass