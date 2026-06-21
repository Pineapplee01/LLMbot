import sys
from pathlib import Path

from runtime_env import (
    build_offline_model_env,
    mark_queue_manifest_failed,
    now_iso,
    resolve_botdetection_root,
    resolve_python_executable,
    run_manifest_command,
    write_json_file as write_json,
)

REPO_ROOT = resolve_botdetection_root()
WORK_DIR = REPO_ROOT / "LLMbot"
PYTHON = resolve_python_executable(r"D:\Anaconda\envs\llmbot\python.exe")
LOG_DIR = WORK_DIR / "server_logs"
EXPERIMENT_DIR = Path("experiments") / "sampled_twibot22_official_prior_base5_20260620"
EXPERIMENT_NAME = r"experiments\sampled_twibot22_official_prior_base5_20260620"
DATASET = "TwiBot-22-official-prior-sampled-v1"
SEEDS = [1, 2, 3, 4, 5]


def clean_env():
    return build_offline_model_env(REPO_ROOT)


def run_logged(command, log_path, manifest, seed, stage):
    proc = run_manifest_command(
        command,
        cwd=WORK_DIR,
        env=clean_env(),
        log_path=log_path,
        manifest=manifest,
        manifest_path=MANIFEST_PATH,
        entry={
            "seed": int(seed),
            "stage": stage,
            "command": command,
            "log_path": str(log_path),
        },
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{stage} failed for seed {seed}; see {log_path}")


MANIFEST_PATH = WORK_DIR / "experiments" / "sampled_twibot22_official_prior_base5_20260620_queue_manifest.json"


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
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
        embedding_path = WORK_DIR / EXPERIMENT_DIR / f"seed_{seed}" / "preparation" / "semantic_encoder" / "embeddings.pt"
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
        mark_queue_manifest_failed(MANIFEST_PATH, exc)
        print(str(exc), file=sys.stderr)
        raise
