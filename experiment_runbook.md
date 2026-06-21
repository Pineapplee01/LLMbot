# LLMbot Experiment Runbook

This runbook is the operator entry point for active LLMbot experiments. It
keeps experiment launch rules, queue-script boundaries, and artifact contracts
discoverable without turning `README.md` or `experiments.md` into the only
source of operational truth.

## Scope

- Active code mainline: this `LLMbot/` repository.
- Experiment registry: `experiments.md`.
- Public CLI entry point: `python main.py ...`.
- Local Windows queue surfaces: root-level dated queue scripts listed below.
- Evidence zones: `experiments/`, `server_logs/`, checkpoints, model caches,
  and datasets are generated or external evidence and must not be hand-edited.

## Required Reading Order

1. `AGENTS.md` for edit zones, validation rules, and research-claim boundaries.
2. This runbook for operational experiment rules.
3. `experiments.md` for the active queue index and detailed experiment records.
4. `README.md` for CLI flags, module boundaries, and artifact contracts.

For parent-workspace or paper-facing NLPCC work, use `../NLPCC/README.md` and
`../NLPCC/docs/llmbot_dependency_boundary.md` before citing experiment evidence.

## Queue Script Contract

Queue scripts are operational launch surfaces, not research conclusions. They
must preserve these rules:

- Use stable semantic experiment IDs and stable manifest paths.
- Record every queued command in a queue manifest before launch.
- Record `finished_at`, `returncode`, and normalized `status` after completion.
- Use `runtime_env.build_offline_model_env(...)` for local HuggingFace and
  Transformers cache variables.
- Use `runtime_env.run_manifest_command(...)` for ordinary subprocess queue
  steps.
- Use `runtime_env.mark_queue_manifest_failed(...)` for top-level queue
  failures so manifests consistently retain `failed_at`, `error_type`, and
  `error`.
- Keep custom process-control loops local when they have extra behavior, such
  as early stopping after an embedding artifact materializes.
- Do not change training flags while doing queue-helper cleanup.

## Active Local Queue Scripts

| Script | Role | Manifest or log contract |
| --- | --- | --- |
| `launch_sampled_twibot22_base5_20260620.py` | Windows launcher for the sampled TwiBot-22 queue | `server_logs/sampled_twibot22_official_prior_base5_20260620_queue.*.log` |
| `run_sampled_twibot22_base_5seed_20260620.py` | Sampled TwiBot-22 semantic and graph queue | `experiments/sampled_twibot22_official_prior_base5_20260620_queue_manifest.json` |
| `run_twibot20_formal5_ablation_completion_20260620.py` | Formal TwiBot-20 ablation completion queue | `experiments/twibot20_formal5_ablation_completion_20260620_queue_manifest.json` |
| `run_twibot20_full_selective_residual_5seed_20260620.py` | Full selective residual five-seed queue | `experiments/twibot20_full_selective_residual_5seed_20260620_queue_manifest.json` |
| `extract_raw_roberta_embeddings_20260620.py` | Support script for raw RoBERTa embeddings | Adjacent `manifest.json` in the requested output directory |

## Report Snapshot Helpers

These scripts summarize existing artifacts. They write report snapshots and
must not be treated as training or queue launch commands.

| Script | Role | Output contract |
| --- | --- | --- |
| `summarize_formal_runs_20260620.py` | Builds CSV summaries and a manifest from existing TwiBot-20/TwiBot-22 artifacts | `experiments/formal_result_snapshots_20260620/manifest.json` and adjacent CSV files |

## Local Model Cache

Local queue scripts should keep downloaded models under the parent workspace:

- `HF_HOME=G:\Research\BotDetection\models\huggingface`
- `HF_HUB_CACHE=G:\Research\BotDetection\models\huggingface\hub`
- `TRANSFORMERS_CACHE=G:\Research\BotDetection\models\huggingface\hub`

Do not add per-script cache rewrites. Reuse `runtime_env.build_offline_model_env`
so offline local runs and smoke checks share the same cache contract.

## Validation Commands

Run these checks after queue-helper, runbook, or report-helper changes:

```powershell
python -m py_compile runtime_env.py launch_sampled_twibot22_base5_20260620.py run_sampled_twibot22_base_5seed_20260620.py run_twibot20_formal5_ablation_completion_20260620.py run_twibot20_full_selective_residual_5seed_20260620.py extract_raw_roberta_embeddings_20260620.py summarize_formal_runs_20260620.py
@'
from pathlib import Path
from runtime_env import build_offline_model_env, mark_queue_manifest_failed, run_manifest_command
env = build_offline_model_env(Path(r"G:\Research\BotDetection"))
assert env["HF_HOME"].endswith(r"models\huggingface")
assert callable(run_manifest_command)
print("runtime_env queue helpers ok")
'@ | python -
@'
import json
import sys
import tempfile
from pathlib import Path
from runtime_env import build_offline_model_env, run_manifest_command
root = Path(tempfile.mkdtemp(prefix="llmbot_manifest_smoke_"))
manifest = {"runs": []}
env = build_offline_model_env(Path(r"G:\Research\BotDetection"))
proc = run_manifest_command(
    [sys.executable, "-c", "print('ok')"],
    cwd=Path.cwd(),
    env=env,
    log_path=root / "ok.log",
    manifest=manifest,
    manifest_path=root / "manifest.json",
    entry={"job": "ok", "command": [sys.executable, "-c", "print('ok')"], "log_path": str(root / "ok.log")},
)
assert proc.returncode == 0
payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
assert payload["runs"][0]["status"] == "completed"
assert payload["runs"][0]["returncode"] == 0
proc = run_manifest_command(
    [sys.executable, "-c", "import sys; sys.exit(3)"],
    cwd=Path.cwd(),
    env=env,
    log_path=root / "fail.log",
    manifest=payload,
    manifest_path=root / "manifest.json",
    entry={"job": "fail", "command": [sys.executable, "-c", "import sys; sys.exit(3)"], "log_path": str(root / "fail.log")},
)
assert proc.returncode == 3
payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
assert payload["runs"][1]["status"] == "failed"
assert payload["runs"][1]["returncode"] == 3
mark_queue_manifest_failed(root / "queue_failed.json", RuntimeError("queue failed"), manifest={"runs": []})
failed = json.loads((root / "queue_failed.json").read_text(encoding="utf-8"))
assert failed["status"] == "failed"
assert failed["error_type"] == "RuntimeError"
assert failed["error"] == "queue failed"
print("manifest command smoke ok")
'@ | python -
git diff --check
```

If `python main.py --help` cannot run because optional ML dependencies are not
available in the current shell, record the exact missing dependency instead of
changing the CLI or dependency policy.
