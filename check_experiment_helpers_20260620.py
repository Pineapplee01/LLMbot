"""Dry validation for root experiment helpers without launching experiments."""

import csv
import importlib
import json
import os
import py_compile
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

from runtime_env import (
    DEFAULT_BOTDETECTION_ROOT,
    build_offline_model_env,
    mark_queue_manifest_failed,
    resolve_powershell_executable,
    run_manifest_command,
    write_csv_rows_file,
)


EXPECTED_OFFLINE_ENV_KEYS = [
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_CACHE",
    "TRANSFORMERS_OFFLINE",
]

PY_COMPILE_TARGETS = [
    "runtime_env.py",
    "launch_sampled_twibot22_base5_20260620.py",
    "run_sampled_twibot22_base_5seed_20260620.py",
    "run_twibot20_formal5_ablation_completion_20260620.py",
    "run_twibot20_full_selective_residual_5seed_20260620.py",
    "extract_raw_roberta_embeddings_20260620.py",
    "summarize_formal_runs_20260620.py",
    "check_experiment_helpers_20260620.py",
]


def compile_targets():
    cache_dir = Path(tempfile.mkdtemp(prefix="llmbot_pycompile_smoke_"))
    for target in PY_COMPILE_TARGETS:
        output = cache_dir / f"{Path(target).stem}.pyc"
        py_compile.compile(target, cfile=str(output), doraise=True)


def check_powershell_resolver():
    assert resolve_powershell_executable(
        source_env={"LLMBOT_POWERSHELL": r"C:\Tools\pwsh.exe"}
    ) == Path(r"C:\Tools\pwsh.exe")
    fallback = resolve_powershell_executable(
        default_path=r"Z:\missing\powershell.exe",
        source_env={},
        which_func=(
            lambda name: (
                r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
                if name == "powershell.exe"
                else None
            )
        ),
    )
    assert str(fallback).endswith("powershell.exe")
    try:
        resolve_powershell_executable(
            default_path=r"Z:\missing\powershell.exe",
            source_env={},
            which_func=lambda name: None,
        )
    except FileNotFoundError as exc:
        assert "LLMBOT_POWERSHELL" in str(exc)
    else:
        raise AssertionError("expected missing PowerShell to fail fast")


def check_offline_env_empty_source_isolated():
    env = build_offline_model_env(Path(DEFAULT_BOTDETECTION_ROOT), source_env={})
    assert sorted(env) == EXPECTED_OFFLINE_ENV_KEYS
    assert env["HF_HOME"].endswith(r"models\huggingface")


def check_launcher_import_and_fail_fast():
    root = Path(tempfile.mkdtemp(prefix="llmbot_launcher_smoke_"))
    previous_root = os.environ.get("BOTDETECTION_ROOT")
    os.environ["BOTDETECTION_ROOT"] = str(root)
    try:
        module = importlib.import_module("launch_sampled_twibot22_base5_20260620")
        original_resolver = module.resolve_powershell_executable
        module.resolve_powershell_executable = lambda: (_ for _ in ()).throw(
            FileNotFoundError("missing powershell")
        )
        try:
            module.main()
        except FileNotFoundError:
            assert not module.log_dir.exists()
        else:
            raise AssertionError("expected launcher to fail before log creation")
    finally:
        if "module" in locals():
            module.resolve_powershell_executable = original_resolver
        if previous_root is None:
            os.environ.pop("BOTDETECTION_ROOT", None)
        else:
            os.environ["BOTDETECTION_ROOT"] = previous_root


def check_sampled_queue_import_side_effects():
    root = Path(tempfile.mkdtemp(prefix="llmbot_import_smoke_"))
    previous_root = os.environ.get("BOTDETECTION_ROOT")
    os.environ["BOTDETECTION_ROOT"] = str(root)
    try:
        module = importlib.import_module("run_sampled_twibot22_base_5seed_20260620")
        assert module.LOG_DIR == root / "LLMbot" / "server_logs"
        assert not module.LOG_DIR.exists()
    finally:
        if previous_root is None:
            os.environ.pop("BOTDETECTION_ROOT", None)
        else:
            os.environ["BOTDETECTION_ROOT"] = previous_root


def check_support_script_import_side_effects():
    module = importlib.import_module("extract_raw_roberta_embeddings_20260620")
    assert callable(module.tokenize)
    assert callable(module.extract)
    assert callable(module.main)


def check_shared_json_readers():
    root = Path(tempfile.mkdtemp(prefix="llmbot_json_smoke_"))
    path = root / "payload.json"
    path.write_text('{"status": "completed", "ok": true}', encoding="utf-8")
    formal = importlib.import_module("run_twibot20_formal5_ablation_completion_20260620")
    full = importlib.import_module("run_twibot20_full_selective_residual_5seed_20260620")
    expected = {"status": "completed", "ok": True}
    assert formal.read_json(path) == expected
    assert full.read_json(path) == expected


def check_routed_mask_helpers():
    module = importlib.import_module("run_twibot20_formal5_ablation_completion_20260620")
    root = Path(tempfile.mkdtemp(prefix="llmbot_routed_source_smoke_"))
    stage = root / "stage"
    stage.mkdir()
    (stage / "routed_nodes_budget100_shuffled.json").write_text("{}", encoding="utf-8")
    (stage / "only_shuffled_budget050_shuffled.json").write_text(
        '{"budget_name": "050"}',
        encoding="utf-8",
    )
    match = stage / "routed_nodes_budget100.json"
    match.write_text("{}", encoding="utf-8")
    metadata = stage / "metadata_only.json"
    metadata.write_text('{"selected_budget": "0.50"}', encoding="utf-8")
    assert module.select_routed_mask_source([stage / "z.json", match]) == match
    assert module.select_routed_mask_source([]) is None
    assert module.budget_matches_metadata({"budget_name": "050"}, "050")
    assert module.find_routed_mask_sources(stage, "100") == [match]
    assert module.find_routed_mask_sources(stage, "050") == [metadata]


def check_csv_helpers():
    formal = importlib.import_module("summarize_formal_runs_20260620")
    full = importlib.import_module("run_twibot20_full_selective_residual_5seed_20260620")
    root = Path(tempfile.mkdtemp(prefix="llmbot_csv_smoke_"))

    formal_path = root / "formal.csv"
    formal_empty = root / "formal_empty.csv"
    formal_auto_empty = root / "formal_auto_empty.csv"
    formal.write_csv(
        formal_path,
        [{"macro_f1_std": 0.2, "variant": "demo", "n": 1, "acc_mean": 0.9, "extra": "ignored"}],
        formal.SUMMARY_FIELDS,
    )
    formal.write_csv(formal_empty, [], formal.SUMMARY_FIELDS)
    formal.write_csv(formal_auto_empty, [])
    with formal_path.open("r", encoding="utf-8", newline="") as handle:
        formal_rows = list(csv.reader(handle))
    with formal_empty.open("r", encoding="utf-8", newline="") as handle:
        formal_empty_rows = list(csv.reader(handle))
    assert formal_rows[0] == formal.SUMMARY_FIELDS
    assert formal_rows[1][0] == "demo"
    assert "ignored" not in formal_rows[1]
    assert formal_empty_rows == [formal.SUMMARY_FIELDS]
    assert formal_auto_empty.read_text(encoding="utf-8") == ""

    full_path = root / "full.csv"
    full_empty = root / "full_empty.csv"
    runtime_path = root / "runtime.csv"
    full.write_summary_csv(
        full_path,
        [{"seed": 1, "test_count": 296, "acc": 0.9, "outputs_path": "outputs.pt", "extra": "ignored"}],
    )
    full.write_summary_csv(full_empty, [])
    write_csv_rows_file(
        runtime_path,
        [{"seed": 2, "test_count": 296, "extra": "ignored"}],
        full.SUMMARY_FIELDS,
        lineterminator="\n",
    )
    assert "\r\n" not in full_path.read_text(encoding="utf-8")
    with full_path.open("r", encoding="utf-8", newline="") as handle:
        full_rows = list(csv.reader(handle))
    with full_empty.open("r", encoding="utf-8", newline="") as handle:
        full_empty_rows = list(csv.reader(handle))
    with runtime_path.open("r", encoding="utf-8", newline="") as handle:
        runtime_rows = list(csv.reader(handle))
    assert full_rows[0] == full.SUMMARY_FIELDS
    assert full_rows[1][0] == "1"
    assert "ignored" not in full_rows[1]
    assert full_empty_rows == [full.SUMMARY_FIELDS]
    assert runtime_rows[0] == full.SUMMARY_FIELDS
    assert runtime_rows[1][0] == "2"
    assert "ignored" not in runtime_rows[1]


def check_manifest_command_helpers():
    root = Path(tempfile.mkdtemp(prefix="llmbot_manifest_smoke_"))
    manifest = {"runs": []}
    env = build_offline_model_env(Path(DEFAULT_BOTDETECTION_ROOT))
    proc = run_manifest_command(
        [sys.executable, "-c", "print('ok')"],
        cwd=Path.cwd(),
        env=env,
        log_path=root / "ok.log",
        manifest=manifest,
        manifest_path=root / "manifest.json",
        entry={
            "job": "ok",
            "command": [sys.executable, "-c", "print('ok')"],
            "log_path": str(root / "ok.log"),
        },
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
        entry={
            "job": "fail",
            "command": [sys.executable, "-c", "import sys; sys.exit(3)"],
            "log_path": str(root / "fail.log"),
        },
    )
    assert proc.returncode == 3
    payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert payload["runs"][1]["status"] == "failed"
    assert payload["runs"][1]["returncode"] == 3

    mark_queue_manifest_failed(
        root / "queue_failed.json",
        RuntimeError("queue failed"),
        manifest={"runs": []},
    )
    failed = json.loads((root / "queue_failed.json").read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["error_type"] == "RuntimeError"
    assert failed["error"] == "queue failed"


def main():
    checks = [
        compile_targets,
        check_powershell_resolver,
        check_offline_env_empty_source_isolated,
        check_launcher_import_and_fail_fast,
        check_sampled_queue_import_side_effects,
        check_support_script_import_side_effects,
        check_shared_json_readers,
        check_routed_mask_helpers,
        check_csv_helpers,
        check_manifest_command_helpers,
    ]
    for check in checks:
        check()
        print(f"{check.__name__} ok")


if __name__ == "__main__":
    main()
