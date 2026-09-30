# LLMbot Governance Tracks

Date: 2026-07-04

## Active Tracks

| Track | Status | Scope | Exit Criteria |
| --- | --- | --- | --- |
| Phase 2 medium cleanup | in progress | Remove deprecated-only compatibility, consolidate owner/helper boundaries inside flat `LLMbot/code/`, and keep docs aligned | Static/CLI checks pass, no experiments or server logs changed, no `stages/` or `methods/` directories added |

## Completed Tracks

| Track | Completed | Result |
| --- | --- | --- |
| Phase 1 flat-source governance | 2026-07-04 | `conductor/`, `UBIQUITOUS_LANGUAGE.md`, architecture docs, and code-risk docs define the flat `LLMbot/code/` active source surface |
| GLANCE/helper tail owner transfer | 2026-07-04 | GLANCE runner/helper duplicates were removed from `trainer_legacy_impl.py`; `trainer_glance.py` is the active GLANCE owner surface |
| `StageSpec` runtime gate convergence | 2026-07-04 | `code/main.py` now consumes `runner_kind`, `claim_grade_allowed`, `requires_canonical_split`, `forces_use_gnn`, and `graph_data_mode` from `StageSpec` |
| Graph conflict diagnostic owner transfer | 2026-07-04 | `local_conflict_prune_diag`, `StructuralConflictRGCN`, structural-view forward, and conflict-router helpers now live in `trainer_graph.py`; duplicate legacy implementations were removed |

## Candidate Phase 2 Tracks

| Track | Priority | Scope | Guardrail |
| --- | --- | --- | --- |
| Local conformal helper owner transfer | high | Move remaining local conformal router/helper tails from `trainer_legacy_impl.py` into `trainer_graph.py` | Preserve local diagnostic outputs and do not rerun experiments |
| Runner script governance | medium | Define naming, manifest, and log-path rules for current `run_*.py` scripts and a future `runners/` target | Do not move all scripts at once; start with one low-risk runner |
| Algorithm boundary cleanup | medium | Plan pure algorithm boundaries for `estimators.py`, `model_building.py`, and `router.py` | Do not mix algorithm extraction with experiment reruns |
| Runtime-only artifact namespace audit | medium | Audit helper artifacts under runtime and stage sidecars for canonical names | Read/verify generated artifacts; do not hand-edit evidence |

## Source-Layout Decision 2026-07-04

The active source layout is deliberately flat:

```text
LLMbot/
  main.py, precompute.py, preprocess.py  # compatibility entrypoints
  code/                                  # active Python source
  run_*.py, launch_*.py, *.ps1, *.cmd    # runner scripts
  experiments/, server_logs/             # artifact surface
```

Do not introduce `stages/`, `methods/`, or similar taxonomy directories yet.
The next useful governance work is deletion, consolidation, and owner transfer
inside `LLMbot/code/`.

## Track Rules

- Each track must state allowed files, out-of-scope paths, validation commands,
  and documentation sync targets before implementation.
- Experiment execution, analysis writing, and review remain separate roles.
- Research claims must stay bounded to verified manifests, metrics, and traces.
