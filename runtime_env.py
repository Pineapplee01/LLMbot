import torch


def _resolve_device(device_arg):
    if isinstance(device_arg, torch.device):
        return device_arg
    if int(device_arg) < 0 or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{int(device_arg)}")


def _set_cuda_device(device):
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.set_device(device)


def _reset_cuda_peak_memory_stats(device):
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _max_cuda_memory_allocated(device):
    if isinstance(device, torch.device) and device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return None
