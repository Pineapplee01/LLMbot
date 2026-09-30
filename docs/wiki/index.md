# Research Wiki Index

Categorical index of the current project state.

Last updated: 2026-05-23

## Project Mission

- Primary task: social bot detection on TwiBot-20
- Primary metrics: Accuracy, Macro-F1
- Auxiliary metrics: ECE, NLL, Brier, AURC, and ranking diagnostics

## Current Project Framing

- current active mainline starts from `LLMbot/`
- `LLMbot/baseline/` is deprecated legacy baseline code
- `LLMbot/code/` is deprecated historical method code
- formal comparison rules still live under `docs/protocols/`

## Project Context

### Code Layers

- Current session default: `LLMbot/main.py`
- Current parser contract: `LLMbot/parser_args.py`
- Current stage/runtime orchestration: `LLMbot/trainer.py`
- Historical/deprecated baseline surface: `LLMbot/baseline/`
- Historical/deprecated active-method surface: `LLMbot/code/`

### Documentation Layers

- Authoritative docs root: `docs/`
- Persistent memory: `docs/wiki/`
- Current routing truth card: `docs/wiki/project/mainline_switch_context_2026-04-23.md`
- Active-mainline architecture note: `docs/ARCHITECTURE.md`
- Current parser contract note: `docs/code/parser.md`
- Current code risk note: `code.md`

### Research Phase Vocabulary

Use [project_phase_taxonomy.md](../research/project_phase_taxonomy.md) for A-G
phase naming. These are planning labels, not CLI task names.

| Phase | Canonical Name |
| --- | --- |
| A | semantic encoder -> GNN |
| B | GNN posterior -> post-hoc estimator/router |
| C | local semantic enhancement |
| D | local propagation repair |
| E | action selection |
| F | same-head refinement |
| G | positioning / stress test |

## Project Policies

1. mainline implementation policy: default to `LLMbot/`
2. auxiliary metrics policy: calibration and ranking metrics are diagnostic
3. comparability policy: formal comparison work must use canonical splits and
   explicit provenance
4. documentation policy: `docs/` is the only authoritative doc root
5. historical code-line policy: `LLMbot/baseline/` and `LLMbot/code/` are not
   current default routing surfaces
6. claim policy: internal implemented branches are not public methods unless
   parser, docs, and validation all expose them consistently

## Project Decisions

- dataset split: use `datasets/TwiBot-20/train_idx.pt`, `valid_idx.pt`,
  `test_idx.pt` as canonical comparison splits
- mainline routing: use `LLMbot/` for all new implementation and review work
- public CLI source: `LLMbot/parser_args.py`
- review summary source: `code.md`
- current mainline docs: `LLMbot/README.md`, `docs/ARCHITECTURE.md`,
  `docs/code/parser.md`

## Experiments And Notes

### Active Mainline

- [project:mainline_switch_context_2026-04-23](project/mainline_switch_context_2026-04-23.md)
- [project:dual_lane_mainline_status_2026-04-23](project/dual_lane_mainline_status_2026-04-23.md)

These notes are now historical routing records and should be read with the
current `LLMbot/` mainline status in mind.

### Historical Comparison Context

- [exp:baseline_audit_correction - Baseline Comparability Audit Correction](experiments/baseline_audit_correction.md)
- [project:baseline_migration_comparison_context_2026-04-17](project/baseline_migration_comparison_context_2026-04-17.md)

These remain historical context, not current active implementation routing.

## Current Mainline Review Status

- public parser surface exists and is usable
- internal handler surface is larger than public parser surface
- docs were recently being realigned to the active mainline
- `trainer.py` and `estimators.py` remain the main maintainability hotspots

## Quick Stats

- active implementation surface: `LLMbot/`
- deprecated implementation surfaces: `LLMbot/baseline/`, `LLMbot/code/`
- current public tasks: 15
- current known internal-only GLANCE branches: 5
- current review status: request changes before structural expansion
