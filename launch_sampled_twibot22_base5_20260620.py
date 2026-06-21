import subprocess
from pathlib import Path

from runtime_env import build_offline_model_env, resolve_botdetection_root


repo_root = resolve_botdetection_root()
work_dir = repo_root / "LLMbot"
script = work_dir / "run_sampled_twibot22_base_5seed_20260620.ps1"
log_dir = work_dir / "server_logs"
stdout_path = log_dir / "sampled_twibot22_official_prior_base5_20260620_queue.stdout.log"
stderr_path = log_dir / "sampled_twibot22_official_prior_base5_20260620_queue.stderr.log"
pid_path = log_dir / "sampled_twibot22_official_prior_base5_20260620_queue.pid"


def clean_env():
    return build_offline_model_env(repo_root)


def main():
    log_dir.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        proc = subprocess.Popen(
            [
                r"C:\windows\System32\WindowsPowerShell\v1.0\powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script),
            ],
            cwd=str(work_dir),
            stdout=stdout,
            stderr=stderr,
            stdin=subprocess.DEVNULL,
            env=clean_env(),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    pid_path.write_text(str(proc.pid), encoding="ascii")
    print(f"STARTED pid={proc.pid}")
    print(f"stdout={stdout_path}")
    print(f"stderr={stderr_path}")


if __name__ == "__main__":
    main()
