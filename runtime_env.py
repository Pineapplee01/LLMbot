import os
from pathlib import Path


def clean_process_env(source_env=None):
    """Return an environment dict with case-insensitive duplicate keys removed."""
    source_env = source_env or os.environ
    env = {}
    seen = set()
    for key, value in source_env.items():
        lower = key.lower()
        if lower in seen:
            continue
        seen.add(lower)
        env["Path" if lower == "path" else key] = value
    return env


def configure_model_cache_env(env, repo_root, *, offline=True):
    """Configure HuggingFace cache variables under the BotDetection model root."""
    repo_root = Path(repo_root)
    hf_home = repo_root / "models" / "huggingface"
    hf_hub_cache = hf_home / "hub"
    env["HF_HOME"] = str(hf_home)
    env["HF_HUB_CACHE"] = str(hf_hub_cache)
    env["TRANSFORMERS_CACHE"] = str(hf_hub_cache)
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    return {
        "hf_home": hf_home,
        "hf_hub_cache": hf_hub_cache,
        "transformers_cache": hf_hub_cache,
    }


def build_offline_model_env(repo_root, source_env=None):
    env = clean_process_env(source_env)
    configure_model_cache_env(env, repo_root=repo_root, offline=True)
    return env


def _torch():
    import torch

    return torch


def _resolve_device(device_arg):
    torch = _torch()
    if isinstance(device_arg, torch.device):
        return device_arg
    if int(device_arg) < 0 or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{int(device_arg)}")


def _set_cuda_device(device):
    torch = _torch()
    if not isinstance(device, torch.device) or device.type != "cuda" or not torch.cuda.is_available():
        return None
    torch.cuda.set_device(device)
    return torch.cuda.current_device()


def _reset_cuda_peak_memory_stats(device):
    torch = _torch()
    cuda_index = _set_cuda_device(device)
    if cuda_index is not None:
        torch.cuda.reset_peak_memory_stats(cuda_index)


def _max_cuda_memory_allocated(device):
    torch = _torch()
    cuda_index = _set_cuda_device(device)
    if cuda_index is not None:
        return int(torch.cuda.max_memory_allocated(cuda_index))
    return None
