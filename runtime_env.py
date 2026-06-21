import os
import json
import subprocess
from datetime import datetime, timezone
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


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat()


def write_json_file(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json_file(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_command(command):
    """Return a subprocess-safe command list without changing argument order."""
    return [str(item) for item in command]


def run_logged_command(command, *, cwd, env, log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as handle:
        return subprocess.run(
            normalize_command(command),
            cwd=str(cwd),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )


def start_manifest_run(manifest, manifest_path, entry):
    """Append a running queue entry and persist the queue manifest."""
    entry = dict(entry)
    entry.setdefault("status", "running")
    entry.setdefault("started_at", now_iso())
    manifest.setdefault("runs", []).append(entry)
    write_json_file(manifest_path, manifest)
    return entry


def finish_manifest_run(manifest, manifest_path, entry, returncode):
    """Record queue command completion without changing manifest schema."""
    entry["finished_at"] = now_iso()
    entry["returncode"] = int(returncode)
    entry["status"] = "completed" if returncode == 0 else "failed"
    write_json_file(manifest_path, manifest)
    return entry


def fail_manifest_run(manifest, manifest_path, entry, exc):
    """Record launch-time failures that happen before a process return code exists."""
    entry["finished_at"] = now_iso()
    entry["status"] = "failed"
    entry["error_type"] = type(exc).__name__
    entry["error"] = str(exc)
    write_json_file(manifest_path, manifest)
    return entry


def mark_queue_manifest_failed(manifest_path, exc, manifest=None):
    """Mark a queue-level manifest failed while preserving existing fields."""
    if manifest is None:
        try:
            manifest = read_json_file(manifest_path, default={}) or {}
        except Exception as read_exc:
            manifest = {
                "manifest_read_error_type": type(read_exc).__name__,
                "manifest_read_error": str(read_exc),
            }
    manifest["status"] = "failed"
    manifest["failed_at"] = now_iso()
    manifest["error_type"] = type(exc).__name__
    manifest["error"] = str(exc)
    write_json_file(manifest_path, manifest)
    return manifest


def run_manifest_command(command, *, cwd, env, log_path, manifest, manifest_path, entry):
    """Run a command while recording the standard queue manifest lifecycle."""
    entry = start_manifest_run(manifest, manifest_path, entry)
    try:
        proc = run_logged_command(command, cwd=cwd, env=env, log_path=log_path)
    except Exception as exc:
        fail_manifest_run(manifest, manifest_path, entry, exc)
        raise
    finish_manifest_run(manifest, manifest_path, entry, proc.returncode)
    return proc


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
