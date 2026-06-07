import torch


def _resolve_device(device_arg):
    if isinstance(device_arg, torch.device):
        return device_arg
    if int(device_arg) < 0 or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{int(device_arg)}")


def _set_cuda_device(device):
    if not isinstance(device, torch.device) or device.type != "cuda" or not torch.cuda.is_available():
        return None
    torch.cuda.set_device(device)
    return torch.cuda.current_device()


def _reset_cuda_peak_memory_stats(device):
    cuda_index = _set_cuda_device(device)
    if cuda_index is not None:
        torch.cuda.reset_peak_memory_stats(cuda_index)


def _max_cuda_memory_allocated(device):
    cuda_index = _set_cuda_device(device)
    if cuda_index is not None:
        return int(torch.cuda.max_memory_allocated(cuda_index))
    return None
