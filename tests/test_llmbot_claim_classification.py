import ast
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LLMBOT_CODE = REPO_ROOT / "LLMbot" / "code"

CLAIMS = {
    "phase_a_foundations",
    "high_order_graph_consumption",
    "conformal_risk_routing",
    "routed_llm_evidence_refinement",
    "local_graph_repair_diagnostics",
    "semantic_candidate_correction",
    "ablation_positioning_legacy",
}

MOVED_MODULES = {
    "router.py": "conformal_risk_routing/router.py",
    "llm_evidence_refiner.py": "routed_llm_evidence_refinement/llm_evidence_refiner.py",
    "trainer_dignn_conflict.py": "local_graph_repair_diagnostics/trainer_dignn_conflict.py",
}

DO_NOT_MOVE_TOP_LEVEL = {
    "precompute.py",
    "trainer_legacy_impl.py",
    "trainer_glance.py",
    "estimators.py",
    "trainer_preparation.py",
    "model_building.py",
    "GNNs.py",
    "prompt.py",
}

ORCHESTRATION_TOP_LEVEL = {
    "main.py",
    "parser_args.py",
    "stage_registry.py",
    "stage_runner.py",
    "artifact_contracts.py",
    "runtime_env.py",
    "trainer.py",
}

ALLOWED_CLAIM_IMPORTS = {
    ("routed_llm_evidence_refinement", "conformal_risk_routing"),
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _claim_map() -> dict:
    with (LLMBOT_CODE / "claim_map.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_claim_packages_exist_with_readmes_and_init_files():
    for claim in CLAIMS:
        package = LLMBOT_CODE / claim
        assert package.is_dir(), claim
        assert (package / "__init__.py").is_file(), claim
        readme = package / "README.md"
        assert readme.is_file(), claim
        text = _read(readme)
        for required in (
            "Claim role",
            "Canonical vocabulary",
            "Primary symbols",
            "Shared dependencies",
            "Do not move yet",
        ):
            assert required in text, f"{required} missing from {readme}"


def test_claim_map_covers_current_top_level_python_files():
    claim_map = _claim_map()
    assert set(claim_map) >= {
        "files",
        "symbols",
        "shared",
        "orchestration",
        "legacy_compat",
        "migration_status",
    }

    mapped_files = {entry["path"] for entry in claim_map["files"]}
    current_top_level = {
        path.name
        for path in LLMBOT_CODE.glob("*.py")
        if path.name != "__init__.py"
    }
    assert current_top_level <= mapped_files

    for entry in claim_map["files"]:
        assert {"path", "claim", "status"} <= set(entry)
        assert entry["status"] in {
            "owned",
            "shared",
            "compat_shim",
            "do_not_move_yet",
            "legacy_replay",
        }


def test_moved_modules_have_claim_implementation_and_top_level_shim():
    for shim_name, implementation_relpath in MOVED_MODULES.items():
        implementation = LLMBOT_CODE / implementation_relpath
        shim = LLMBOT_CODE / shim_name
        assert implementation.is_file(), implementation
        assert shim.is_file(), shim
        shim_text = _read(shim)
        assert "compatibility shim" in shim_text.lower()
        module_path = implementation_relpath.replace("/", ".").removesuffix(".py")
        assert module_path in shim_text


def test_do_not_move_large_files_remain_top_level():
    for name in DO_NOT_MOVE_TOP_LEVEL:
        assert (LLMBOT_CODE / name).is_file(), name


def test_claim_packages_do_not_cross_import_without_allowlist():
    for claim in CLAIMS:
        for path in (LLMBOT_CODE / claim).glob("*.py"):
            if path.name == "__init__.py":
                continue
            tree = ast.parse(_read(path))
            for node in ast.walk(tree):
                module = None
                if isinstance(node, ast.ImportFrom):
                    module = node.module
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_root = alias.name.split(".", 1)[0]
                        if imported_root in CLAIMS and imported_root != claim:
                            assert (claim, imported_root) in ALLOWED_CLAIM_IMPORTS
                    continue
                if not module:
                    continue
                imported_root = module.split(".", 1)[0]
                if imported_root in CLAIMS and imported_root != claim:
                    assert (claim, imported_root) in ALLOWED_CLAIM_IMPORTS


def test_claim_map_records_symbol_level_classification_for_moved_and_deferred_files():
    claim_map = _claim_map()
    symbol_keys = {
        (entry["owner_file"], entry["symbol"], entry["kind"], entry["status"])
        for entry in claim_map["symbols"]
    }
    required = {
        ("conformal_risk_routing/router.py", "GlanceReliabilityRouterMLP", "class", "owned"),
        ("routed_llm_evidence_refinement/llm_evidence_refiner.py", "LLMEvidenceRefiner", "class", "owned"),
        ("local_graph_repair_diagnostics/trainer_dignn_conflict.py", "DignnDualViewProxy", "class", "owned"),
        ("operators.py", "build_repair_operator", "function", "do_not_move_yet"),
    }
    assert required <= symbol_keys
