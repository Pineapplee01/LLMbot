import json
import sys
from pathlib import Path

from runtime_env import build_offline_model_env, now_iso, run_logged_command, write_json_file

REPO_ROOT = Path(r"G:\Research\BotDetection")
WORK_DIR = REPO_ROOT / "LLMbot"
PYTHON = Path(r"D:\Anaconda\envs\llmbot\python.exe")
LOG_DIR = WORK_DIR / "server_logs"
EXPERIMENT_NAME = r"experiments\sampled_twibot22_official_prior_base5_20260620"
DATASET = "TwiBot-22-official-prior-sampled-v1"
SEEDS = [1, 2, 3, 4, 5]


def clean_env():
    return build_offline_model_env(REPO_ROOT)


def write_json(path, payload):
    write_json_file(path, payload)


def run_logged(command, log_path, manifest, seed, stage):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    manifest["runs"].append(
        {
            "seed": int(seed),
            "stage": stage,
            "status": "running",
            "started_at": now_iso(),
            "command": command,
            "log_path": str(log_path),
        }
    )
    write_json(MANIFEST_PATH, manifest)
    proc = run_logged_command(command, cwd=WORK_DIR, env=clean_env(), log_path=log_path)
    manifest["runs"][-1]["finished_at"] = now_iso()
    manifest["runs"][-1]["returncode"] = int(proc.returncode)
    manifest["runs"][-1]["status"] = "completed" if proc.returncode == 0 else "failed"
    write_json(MANIFEST_PATH, manifest)
    if proc.returncode != 0:
        raise RuntimeError(f"{stage} failed for seed {seed}; see {log_path}")


LOG_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = WORK_DIR / "experiments" / "sampled_twibot22_official_prior_base5_20260620_queue_manifest.json"


def main():
    manifest = {
        "created_at": now_iso(),
        "dataset": DATASET,
        "experiment_name": EXPERIMENT_NAME,
        "python": str(PYTHON),
        "model_root": str(REPO_ROOT / "models" / "huggingface"),
        "seeds": SEEDS,
        "stages": ["semantic_encoder_finetune", "graph_detector_prepare"],
        "notes": "Full-text/drop-no-tweet sampled TwiBot-22 base 5-seed queue. Uses local G: model cache in offline mode.",
        "runs": [],
        "status": "running",
    }
    write_json(MANIFEST_PATH, manifest)

    for seed in SEEDS:
        embedding_path = WORK_DIR / EXPERIMENT_NAME / f"seed_{seed}" / "preparation" / "semantic_encoder" / "embeddings.pt"
        semantic_log = LOG_DIR / f"sampled_twibot22_official_prior_base5_20260620_seed{seed}_semantic.log"
        graph_log = LOG_DIR / f"sampled_twibot22_official_prior_base5_20260620_seed{seed}_graph.log"

        if not embedding_path.exists():
            semantic_cmd = [
                str(PYTHON),
                "main.py",
                "--experiment_task",
                "semantic_encoder_finetune",
                "--dataset",
                DATASET,
                "--reset_split",
                "-1",
                "--semantic_encoder",
                "roberta",
                "--lm_batch_size",
                "4",
                "--max_length",
                "512",
                "--seeds",
                str(seed),
                "--device",
                "0",
                "--disable_wandb",
                "--experiment_name",
                EXPERIMENT_NAME,
            ]
            run_logged(semantic_cmd, semantic_log, manifest, seed, "semantic_encoder_finetune")

        graph_cmd = [
            str(PYTHON),
            "main.py",
            "--experiment_task",
            "graph_detector_prepare",
            "--dataset",
            DATASET,
            "--reset_split",
            "-1",
            "--graph_data_variant",
            "labeled",
            "--graph_backbone",
            "rgcn",
            "--embedding_path",
            str(embedding_path),
            "--graph_detector_epochs",
            "200",
            "--seeds",
            str(seed),
            "--device",
            "0",
            "--disable_wandb",
            "--experiment_name",
            EXPERIMENT_NAME,
        ]
        run_logged(graph_cmd, graph_log, manifest, seed, "graph_detector_prepare")

    manifest["status"] = "completed"
    manifest["completed_at"] = now_iso()
    write_json(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        existing = {}
        if MANIFEST_PATH.exists():
            existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        existing["status"] = "failed"
        existing["failed_at"] = now_iso()
        existing["error"] = str(exc)
        write_json(MANIFEST_PATH, existing)
        print(str(exc), file=sys.stderr)
        raise
