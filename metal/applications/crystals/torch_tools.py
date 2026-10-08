import os
from typing import List

import torch


def cuda_devices_sorted_by_free_mem() -> List[int]:
    """
    List available CUDA devices sorted by increasing available memory.

    To get the device with the most free memory, use the last list item.
    """

    free_memories = []
    deviceCount = torch.cuda.device_count()
    for i in range(deviceCount):
        info = torch.cuda.mem_get_info(i)
        free_memories.append(info[0])

    return sorted(range(len(free_memories)), key=lambda x: free_memories[x])


def get_torch_device(
        potential: str = 'CHGNet',
        gpu_id: int = None,
        threads_per_process: int = None) -> str:
    """
    Get a string representing the torch device that should be used

    Parameters
    ----------
    potential
        a potential chosen from 'CHGNet', 'M3GNet' and 'MatGL'. Note that the
        latter corresponds to the M3GNet implemenation in the Materials Graph
        Library (MatGL), which is based on pytorch rather than tensorflow.
        Defaults to 'CHGNet'.
    gpu_id
        GPU id, selected automatically if negative number
    threads_per_process
        number of threads to be used by each process. If unspecified it will be
        set equal to the environment variable `OMP_NUM_THREADS`, if available,
        otherwise, this will be handled by the MLP. In the latter case,
        multiple processes will typically compete for all available threads.
        Since this is not an issue when using Tensorflow `threads_per_process`
        is ignored when using GPUs, or the M3GNet potential.
    """
    if potential.lower() == 'm3gnet':

        raise Exception('Other functionalities should be used for setting the'
                        ' device when using M3GNet')

    if gpu_id is not None:

        if potential.lower() == 'chgnet' and torch.backends.mps.is_available():
            device = 'mps'
        else:
            if (gpu_id < 0 and torch.cuda.is_available() and
                    torch.cuda.device_count() > 0):
                gpu_id = cuda_devices_sorted_by_free_mem()[-1]
            if gpu_id > 0:
                device = f"cuda:{gpu_id}"
            else:
                device = None
    else:
        if threads_per_process is None: 
            try:
                threads_per_process = int(os.getenv('OMP_NUM_THREADS')) 
            except Exception:
                pass
        if threads_per_process is not None:
            torch.set_num_threads(threads_per_process)
        device = 'cpu'

    return device

