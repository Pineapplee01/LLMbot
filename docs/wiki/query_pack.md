# Query Pack

Compressed project context for new sessions.

Last updated: 2026-06-14

## Project Direction

- Primary task: social bot detection on TwiBot-20
- Primary metrics: Accuracy, Macro-F1
- Auxiliary metrics: calibration and ranking diagnostics only

Current routing truth card:
- `docs/wiki/project/mainline_switch_context_2026-04-23.md`

## Current Mainline Truth

The only active implementation surface is `LLMbot/`.

- Default entrypoint: `LLMbot/main.py`
- Parser source of truth: `LLMbot/parser_args.py`
- Main execution surface: `LLMbot/trainer.py`
- Default operator docs: `LLMbot/README.md`

Deprecated and historical surfaces:
- `LLMbot/baseline/`: deprecated legacy baseline surface, forensic/reference only
- `LLMbot/code/`: deprecated historical active-method line, forensic/reference only

Do not route new sessions to `LLMbot/baseline/` or `LLMbot/code/` unless the
task explicitly asks for migration, archival cleanup, deletion, or historical
comparison.

## Code-Layer Truth

- `LLMbot/` is the current active mainline
- `LLMbot/baseline/` is not the current default execution surface
- `LLMbot/code/` is not the current default execution surface
- public CLI contract comes from `LLMbot/parser_args.py`
- internal implemented branches may still exist in `LLMbot/trainer.py`; treat
  them as internal unless the parser exposes them

Current public task surface:
- `legacy_distill`
- `semantic_finetune`
- `frozen_g0`
- `frozen_gats`
- `local_conformal_prune_diag`
- `glance_joint_router_refine`
- `vertical_minimal`
- `estimator_matrix`
- `semantic_matrix`
- `semantic_source_matrix`
- `repair_matrix`
- `selector_matrix`
- `positioning_matrix`
- `backbone_stress`
- `appendix`

## Research Boundary Reminder

Current active-mainline review found three important boundaries:

1. public parser tasks and internal stage handlers are not fully aligned
2. `trainer.py` and `estimators.py` remain refactor hotspots
3. some historical names such as `frozen_g0` remain public even though CLI
   flag names were partially canonicalized

Use:
- `docs/code/parser.md` for CLI contract
- `docs/ARCHITECTURE.md` for implementation map
- `code.md` for current code-review risks

## Comparison Context

Comparison lines that remain relevant but are not active mainline:
- `LMBot/`
- `botbr/`
- `HyperScan/`
- `SEBot/`

Formal comparison work must still follow:
- `docs/protocols/baseline_comparability.md`

Reported versus reproduced baselines must remain explicitly separated.

## Collaboration Preferences

- Prefer research quality over speed
- Distinguish implementation fact from research claim
- Keep code changes scoped and reviewable
- Use the active-mainline docs, not historical baseline routing notes, as the
  default source of truth

## Current Open Work

1. align public parser surface, internal stage registry, and docs
2. reduce naming drift between canonical CLI and legacy internal namespace
3. split `trainer.py` by responsibility boundary
4. tighten artifact provenance for reused calibration and external graph inputs
5. add lightweight validation for parser/docs/manifest drift

## Canonical Frozen SimTeG Baseline For Current Router/Prompt Work

- Baseline record: `docs/research/baselines/frozen_simteg_fullgraph_20260614/`
- Server artifact family:
  `/root/workspace/LMbot/LLMbot_active_knn_router_20260612/experiments/tmp_iterm1_forward_fullgraph/seed_{1,2,3}/preparation/graph_detector/outputs.pt`
- Contract: `frozen_g0_v1`, `rgcn`, `full_graph_support`,
  `train_only_supervised`, frozen `embeddings_iter_-1_seed_{1,2,3}.pt`
- Test aggregate: Accuracy `0.8631 +/- 0.0044`, Macro-F1
  `0.8613 +/- 0.0046`, Bot-F1 `0.8769 +/- 0.0039`
- Use this line for KNN conformal router, routed prompt/refiner, and
  GraphEdit-style edge-adjudication comparisons unless explicitly running a
  separate stronger-base ablation.
- Do not mix it with `finetuned_roberta_embeddings_iter_2_seed1.pt`,
  `train_supervised_plus_mhlgc`, labeled-only, or HyperScan-style detector
  ablation artifacts when claiming prompt/router gains.
# Method Boundary: KNN Router Evidence Semantics

- Do not frame the current KNN/conformal router as a three-class edge classifier with categories such as "trusted same-label support", "trusted opposite-label evidence", and "unreliable/conflict evidence" unless a future experiment explicitly implements and validates that target. Current evidence supports KNN hyperedges as risk/conflict signals, not as signed discriminative support.
- Prefer evidence-grounded wording: continuous edge/neighborhood reliability, graph denoising, selective prediction/rejection, risk calibration, and LLM adjudication for routed high-risk nodes.
- Treat "signed support edge" or "opposite-label evidence edge" as a blacklisted method framing for this project unless the user explicitly reopens it with a cited method and an experiment plan.

# Method Boundary: LLM Confidence And Override Semantics

- Do not claim that an LLM's self-reported confidence is a calibrated risk score or a sufficient basis for overriding the base detector. Current literature supports verbalized confidence only as an uncertainty signal that needs task-specific calibration, consistency checks, or external validation.
- Do not implement or describe the routed-node refiner as "LLM self-reports confidence and decides whether to override" unless a future experiment explicitly validates calibration and override safety on the target task.
- Prefer evidence-grounded wording: LLM produces structured evidence assessment and optional uncertainty rationale; the final override decision is made by an external calibrated router/refiner, threshold, or validated selection rule.
