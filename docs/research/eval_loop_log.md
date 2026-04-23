# Eval Loop Log

This log records eval-driven research improvement iterations. Passing checks here only establish engineering or provenance status; they do not upgrade research claims without the result review checklist.

## 2026-04-23 - Loop 1: Legacy B2 Provenance Wrapping

### Baseline Checks Before Change

| Check | Command | Result | Evidence Meaning |
| --- | --- | --- | --- |
| Root governance gate | `bash scripts/run_research_gate.sh --task-type governance --paths AGENTS.md .agents docs/research scripts .github` | pass | Governance filenames, scope, and manifest discovery were valid for the governance surface. |
| Active implementation tests | `D:\Anaconda\python.exe -m pytest LLMbot/baseline/core/tests -q` | `21 passed` | Current `LLMbot/baseline/core` focused tests passed. |
| Legacy harness fast suite | `D:\Anaconda\python.exe -m pytest LMBot/tests/protocol/test_protocol_hardening.py LMBot/tests/frmi/test_frmi_backbone_contracts.py LMBot/tests/frmi/test_frmi_policy_logic.py LMBot/tests/frmi/test_frmi_three_action_gate.py LMBot/tests/harness/test_harness_core.py LMBot/tests/harness/test_harness_pipeline.py LMBot/tests/compatibility/test_legacy_wrapper_parity.py -q` | `25 passed` | Legacy/reference harness checks passed. |
| Legacy manifest probe | `python scripts/verify_experiment_manifest.py LLMbot/saved_artifacts/rw4_disagreement_local/seed_42/split_manifest.json --allow-missing-paths` | fail | Existing `split_manifest.json` is a legacy split-count file, not a flat claim-bearing experiment manifest. |

### What Changed

Added flat reproducibility manifests for the existing B2 single-seed mechanism-evidence pair:

| Manifest | Metrics Path | Artifact Root | Status |
| --- | --- | --- | --- |
| `docs/research/manifests/rw4_disagreement_local_seed_42_manifest.json` | `LLMbot/saved_artifacts/rw4_disagreement_local/seed_42/metrics.json` | `LLMbot/saved_artifacts/rw4_disagreement_local/seed_42` | Legacy artifact wrapped with caveats. |
| `docs/research/manifests/same_trigger_defer_baseline_seed_42_manifest.json` | `LLMbot/saved_artifacts/same_trigger_defer_baseline/seed_42/results.json` | `LLMbot/saved_artifacts/same_trigger_defer_baseline/seed_42` | Legacy baseline artifact wrapped with caveats. |

No saved artifacts, result files, datasets, or method code were edited.

### Evidence-Bounded Observation

The wrapped artifacts record single-seed B2 comparator metrics:

| Record | Global F1 | Global Accuracy | Disagreement-Slice F1 | Disagreement-Slice n |
| --- | --- | --- | --- | --- |
| `rw4_disagreement_local_seed_42` | 0.7603052747698057 | 0.7641589180050719 | 0.5881175636277678 | 264 |
| `same_trigger_defer_baseline_seed_42` | 0.7444491800024522 | 0.7497886728655959 | 0.43242669280405127 | 344 |

This is an auditability improvement only. It does not establish a publication-grade mechanism claim because the original run command, original run commit, full runtime, formal baseline comparability status, and rerun or multi-seed stability evidence remain incomplete.

### Post-Change Validation

| Check | Command | Result | Evidence Produced |
| --- | --- | --- | --- |
| Filename validation | `python scripts/validate_filenames.py --paths docs/research` | pass, 6 files checked | New manifest and log filenames follow repo naming rules. |
| Manifest validation | `python scripts/verify_experiment_manifest.py docs/research/manifests/rw4_disagreement_local_seed_42_manifest.json docs/research/manifests/same_trigger_defer_baseline_seed_42_manifest.json` | pass, 2 manifests validated | Both flat manifests contain required fields and point to existing config, metrics, and artifact paths. |
| Research gate | `bash scripts/run_research_gate.sh --task-type experiment --manifest docs/research/manifests/rw4_disagreement_local_seed_42_manifest.json --manifest docs/research/manifests/same_trigger_defer_baseline_seed_42_manifest.json --paths docs/research` | pass | Filename, changed-scope, and manifest checks pass together for the research-doc change. |

Operational note: in the current gate parser, put `--manifest` arguments before `--paths`; `--paths` consumes the remaining arguments.

### Status

- Code status: unchanged in this loop.
- Provenance status: improved from legacy-only split/result files to flat manifest wrappers that can be validated.
- Claim status: partial or pending; not upgraded.

### Required Next Review

Before using these numbers in claim-facing text, run `docs/research/result_review_checklist.md` and record:

- Baseline comparison status.
- Ablation status.
- Rerun or stability status.
- Whether the conclusion is directly supported by recorded artifacts.
- Any unsupported or downgraded claims.
