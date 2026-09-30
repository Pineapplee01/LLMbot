import os
import shutil
import subprocess
import sys
from pathlib import Path


sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[1]
NLPCC_CODE = REPO_ROOT / "NLPCC" / "code"
LLMBOT_CODE = REPO_ROOT / "LLMbot" / "code"
TARGET_NLPCC_PY_FILES = {
    "main.py",
    "parser_args.py",
    "graph_detector.py",
    "router.py",
    "models.py",
    "data_io.py",
    "utils.py",
}
EXPECTED_PUBLIC_INTERFACES = {
    "main": {"main"},
    "parser_args": {"NLPCC_EXPERIMENT_TASKS", "build_parser", "parser_args"},
    "graph_detector": {"build_graph_detector_config", "describe_graph_detector_task", "run_graph_detector_task"},
    "router": {"rank_risk_scores", "select_top_budget"},
    "models": {"GraphDetectorConfig", "ModelConfig"},
    "data_io": {"read_json", "resolve_path", "write_json"},
    "utils": {"parse_seed_list"},
}
FORBIDDEN_NLPCC_TOKENS = {
    "joint_router_refinement",
    "trainer_glance",
    "GLANCE",
    "stage_runner",
    "stage_registry",
    "artifact_contracts",
    "dry_run",
    "ensure_list",
}


def _pythonpath_env(source_root):
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(source_root) if not existing else str(source_root) + os.pathsep + existing
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    return env


def test_nlpcc_code_is_trimmed_to_submission_modules():
    assert NLPCC_CODE.is_dir()
    actual_py_files = {path.name for path in NLPCC_CODE.glob("*.py")}
    assert actual_py_files == TARGET_NLPCC_PY_FILES

    forbidden_entries = {
        "trainer_glance.py",
        "trainer_preparation.py",
        "trainer_legacy_impl.py",
        "precompute.py",
        "preprocess.py",
        "prompt.py",
        "llm_evidence_refiner.py",
        "stage_registry.py",
        "stage_runner.py",
        "artifact_contracts.py",
        "runtime_env.py",
        "estimators.py",
        "GNNs.py",
        "model_building.py",
        "hypergnn.py",
        "dataloader.py",
        "runners",
        "claims",
        "utils",
    }
    for name in forbidden_entries:
        assert not (NLPCC_CODE / name).exists(), name


def test_nlpcc_code_imports_submission_modules_without_training_execution(monkeypatch):
    monkeypatch.syspath_prepend(str(NLPCC_CODE))

    from parser_args import parser_args
    import data_io
    import graph_detector
    import models
    import router
    import utils

    args = parser_args(["--experiment_task", "graph_detector_prepare"])

    assert args.experiment_task == "graph_detector_prepare"
    modules = {
        "graph_detector": graph_detector,
        "router": router,
        "models": models,
        "data_io": data_io,
        "utils": utils,
    }
    for module_name, module in modules.items():
        assert set(module.__all__) == EXPECTED_PUBLIC_INTERFACES[module_name]


def test_nlpcc_public_interfaces_are_explicit_and_submission_scoped(monkeypatch):
    monkeypatch.syspath_prepend(str(NLPCC_CODE))

    import data_io
    import graph_detector
    import main
    import models
    import parser_args
    import router
    import utils

    modules = {
        "main": main,
        "parser_args": parser_args,
        "graph_detector": graph_detector,
        "router": router,
        "models": models,
        "data_io": data_io,
        "utils": utils,
    }
    for module_name, module in modules.items():
        assert set(module.__all__) == EXPECTED_PUBLIC_INTERFACES[module_name]


def test_nlpcc_parser_rejects_legacy_and_non_submission_terms(monkeypatch):
    monkeypatch.syspath_prepend(str(NLPCC_CODE))

    from parser_args import parser_args

    for raw_args in (
        ["--stage", "graph_detector_prepare"],
        ["--experiment_task", "joint_router_refinement"],
        ["--dry_run"],
    ):
        try:
            parser_args(raw_args)
        except SystemExit as exc:
            assert exc.code != 0
        else:
            raise AssertionError(f"Expected parser rejection for {raw_args!r}")


def test_nlpcc_and_llmbot_help_entrypoints_remain_available():
    commands = [
        ([sys.executable, "main.py", "--help"], REPO_ROOT / "LLMbot", _pythonpath_env(LLMBOT_CODE)),
        ([sys.executable, str(NLPCC_CODE / "main.py"), "--help"], REPO_ROOT, _pythonpath_env(NLPCC_CODE)),
    ]

    for command, cwd, env in commands:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "--experiment_task" in result.stdout
    assert "joint_router_refinement" not in subprocess.run(
        [sys.executable, str(NLPCC_CODE / "main.py"), "--help"],
        cwd=str(REPO_ROOT),
        env=_pythonpath_env(NLPCC_CODE),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    ).stdout


def test_nlpcc_code_does_not_expose_forbidden_terms():
    for path in NLPCC_CODE.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_NLPCC_TOKENS:
            assert token not in text, f"{token} leaked into {path.name}"


def test_llmbot_claim_folders_remain_outside_nlpcc_trimmed_package():
    assert (LLMBOT_CODE / "claims").is_dir()
    assert not (NLPCC_CODE / "claims").exists()


def test_artifact_directories_are_not_moved_under_nlpcc_code():
    forbidden_names = {
        "experiments",
        "server_logs",
        "saved_artifacts",
        "checkpoints",
        "__pycache__",
        ".pytest_cache",
    }
    forbidden_suffixes = {
        ".pt",
        ".pth",
        ".ckpt",
        ".pkl",
        ".npy",
        ".npz",
        ".bin",
        ".safetensors",
    }

    pycache = NLPCC_CODE / "__pycache__"
    if pycache.exists():
        shutil.rmtree(pycache)

    assert NLPCC_CODE.is_dir()
    for directory in NLPCC_CODE.iterdir():
        if directory.is_dir():
            assert directory.name not in forbidden_names, directory.name
            assert directory.name not in {"claims", "runners", "utils"}, directory.name
    for name in forbidden_names:
        assert not (NLPCC_CODE / name).exists(), name
    for path in NLPCC_CODE.rglob("*"):
        assert path.suffix.lower() not in forbidden_suffixes, str(path)
