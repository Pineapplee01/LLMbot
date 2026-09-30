from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LLMBOT_CODE = REPO_ROOT / "LLMbot" / "code"
NLPCC_CODE = REPO_ROOT / "NLPCC" / "code"

COMMON_CANONICAL_TERMS = {
    "experiment_task",
    "task_spec",
    "graph_backbone",
    "text_encoder",
    "semantic_encoder",
    "embedding_path",
    "fused_x",
    "target_node_ids",
    "routed_node_ids",
    "center_node_ids",
    "risk_score",
    "router_score",
    "stage_dir",
    "artifact_namespace",
    "manifest",
}

LEGACY_ONLY_TERMS = {
    "stage",
    "stage_name",
    "GNN_model",
    "LM_model",
    "semantic_backbone",
    "emb_path",
    "node_repr",
    "frozen_g0",
}

CLAIM_VOCABULARY = {
    "phase_a_foundations": {
        "semantic_embeddings",
        "graph_detector_outputs",
        "fused_x",
        "train_idx",
        "valid_idx",
        "test_idx",
        "calibration_logits",
    },
    "high_order_graph_consumption": {
        "x_low",
        "x_new",
        "x_high",
        "second_view_edges",
        "routed_highpass_bundle",
        "center_node_ids",
    },
    "conformal_risk_routing": {
        "risk_score",
        "risk_scores",
        "router_score",
        "risk_budget",
        "selected_node_ids",
        "coverage_curve",
    },
    "routed_llm_evidence_refinement": {
        "prompt_expert_bundle",
        "evidence_card",
        "refiner_features",
        "oracle_advantage",
        "utility_reward",
        "gate_prob",
    },
    "local_graph_repair_diagnostics": {
        "conflict_score",
        "pruned_edge_index",
        "repair_delta",
        "structural_view_logits",
        "deferral_decision",
    },
    "semantic_candidate_correction": {
        "semantic_gate_score",
        "candidate_output_path",
        "correction_delta",
        "semantic_evidence",
    },
    "ablation_positioning_legacy": {
        "replay_compatibility",
        "legacy_alias",
        "historical_command",
    },
}

NLPCC_FORBIDDEN_TERMS = {
    "stage",
    "GLANCE",
    "trainer_glance",
    "node_repr",
    "joint_router_refinement",
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_common_llmbot_naming_governance_is_documented():
    glossary = _read(REPO_ROOT / "UBIQUITOUS_LANGUAGE.md")
    assert "LLMbot Naming Governance 2026-07-05" in glossary
    for term in COMMON_CANONICAL_TERMS:
        assert term in glossary
    for term in LEGACY_ONLY_TERMS:
        assert term in glossary
    for exception_term in ("StageSpec", "stage_registry.py", "stage_runner.py", "stage_dir"):
        assert exception_term in glossary


def test_claim_readmes_define_canonical_vocabulary():
    claim_root = LLMBOT_CODE / "claims"
    assert claim_root.is_dir()
    actual_claims = {path.name for path in claim_root.iterdir() if path.is_dir()}
    assert actual_claims == set(CLAIM_VOCABULARY)

    for claim_name, required_terms in CLAIM_VOCABULARY.items():
        readme = claim_root / claim_name / "README.md"
        text = _read(readme)
        assert "Canonical vocabulary" in text
        for term in required_terms:
            assert term in text, f"{term} missing from {readme}"


def test_legacy_aliases_are_not_advertised_as_claim_canonical_names():
    for readme in (LLMBOT_CODE / "claims").glob("*/README.md"):
        text = _read(readme)
        canonical_section = text.split("Canonical vocabulary", 1)[-1]
        canonical_section = canonical_section.split("Migration note", 1)[0]
        for legacy_term in LEGACY_ONLY_TERMS:
            assert f"`{legacy_term}`" not in canonical_section, (
                f"{legacy_term} advertised as canonical in {readme}"
            )


def test_nlpcc_package_stays_free_of_llmbot_only_terms():
    assert NLPCC_CODE.is_dir()
    for path in NLPCC_CODE.glob("*.py"):
        text = _read(path)
        for term in NLPCC_FORBIDDEN_TERMS:
            assert term not in text, f"{term} leaked into {path}"


def test_architecture_and_risk_log_reference_naming_governance():
    architecture = _read(REPO_ROOT / "docs" / "ARCHITECTURE.md")
    risk_log = _read(REPO_ROOT / "code.md")
    for text in (architecture, risk_log):
        assert "LLMbot Naming Governance 2026-07-05" in text
        assert "claim vocabulary" in text
