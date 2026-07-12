# TwiBot-20 Paper-Formula And Same-Hyperedge Seed-1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run two isolated TwiBot-20 seed-1 experiments that compare the exact paper reference-tail router with an explicitly configured same-hyperedge router under one graph-training contract.

**Architecture:** A dedicated experiment runner reads the existing seed-1 low-only C2 artifact as the frozen routing source, generates one route mask from the exact paper equation and one from the built-in same-hyperedge estimator, then launches two fresh routed-only residual graph runs. It recomputes canonical test metrics from `outputs.pt`, records every command and status, and never modifies historical artifacts or paper tables.

**Tech Stack:** Python 3, PyTorch, NumPy, scikit-learn, existing `LLMbot/main.py` CLI, DHG/PyTorch Geometric graph stack.

## Global Constraints

- Dataset and split files remain the canonical local TwiBot-20 artifacts.
- Seed is fixed to `1`; routing budget is `0.10`; `K=8`; fanout is `64`; graph optimizer budget is `200` steps.
- P-Paper uses train-reference weighted-tail risk with leave-one-out for train targets.
- P-Hyperedge explicitly records `same_hyperedge` and `same_hyperedge_selected_tail`.
- Existing experiments, checkpoints, tables, and rebuttal evidence are read-only.
- No new dependency and no new test file.
- All generated outputs live under `LLMbot/experiments/twibot20_seed1_router_alignment_20260712/`.

---

### Task 1: Build The Isolated Experiment Runner

**Files:**
- Create: `LLMbot/run_twibot20_seed1_router_alignment_20260712.py`
- Modify: `LLMbot/README.md`
- Modify: `docs/experiment.md`

**Interfaces:**
- Consumes: C2 low-only `outputs.pt`, canonical split tensors, seed-1 fine-tuned RoBERTa embedding, and `LLMbot/main.py`.
- Produces: `experiment_manifest.json`, two route-mask JSON files, two fresh graph experiment roots, and canonical comparison reports.

- [ ] **Step 1: Add deterministic route and manifest helpers**

Implement `full_reference_tail_risk(...)`, `select_split_budget(...)`, `write_json(...)`, `artifact_entry(...)`, and `run_logged(...)`. `full_reference_tail_risk` must normalize `x_new`, apply `exp(-distance / 1.0)`, zero a train target's self-weight, and return `1 - tau` exactly as written in the paper.

- [ ] **Step 2: Add exact P-Paper mask generation**

Load the C2 low-only artifact:

```text
LLMbot/experiments/twibot20_seed1_component_ablation_20260619/
  C2_wo_residual_finetuned_low_only_empty_mask/seed_1/preparation/graph_detector/outputs.pt
```

Use its `logits`, `x_new`, and canonical train/valid/test indices. Select `floor(0.10 * split_size)` nodes independently per split and write `inputs/paper_reference_tail_budget100_seed1.json` with score provenance and SHA-256 hashes.

- [ ] **Step 3: Add explicit P-Hyperedge router execution**

Invoke the existing estimator CLI with all choices explicit:

```powershell
python main.py --experiment_task estimator_ablation --dataset TwiBot-20 --reset_split -1 `
  --graph_data_variant labeled --use_GNN --graph_backbone rgcn_hyperscan_dhg_nodeinput `
  --estimator_mode conformal_knn_risk_router --external_frozen_g0_root <C2-seed-root> `
  --conformal_knn_repr_source x_new --conformal_knn_candidate_scope labeled_full `
  --conformal_knn_k 8 --conformal_knn_learning_mode ncp_local `
  --conformal_knn_local_calibration_scope same_hyperedge `
  --conformal_knn_score_family_override same_hyperedge_selected_tail `
  --conformal_knn_ncp_lambda 1.0 --risk_budgets 0.1 --seeds 1 --device 0 `
  --disable_wandb --artifact_root <same-hyperedge-router-root>
```

Materialize `inputs/same_hyperedge_budget100_seed1.json` only after validating the resolved config and selected score family.

- [ ] **Step 4: Add fresh graph-training arms**

Run `graph_detector_prepare` twice with the existing paper-facing graph contract and distinct artifact roots. Both commands must include `routed_only`, `low_only` fallback, the arm-specific route mask, `K=8`, fanout 64, DHG backend, and `--force_retrain_backbone`.

- [ ] **Step 5: Add canonical reporting and dry-run mode**

Recompute Accuracy, Macro-Precision, Macro-F1, wrong-to-right, right-to-wrong, routed count, and route overlap from canonical `test_idx.pt`. Add `--dry-run` to emit commands and validate inputs without launching training.

- [ ] **Step 6: Document the operator command**

Document:

```powershell
cd G:\Research\BotDetection\LLMbot
python run_twibot20_seed1_router_alignment_20260712.py --dry-run
python run_twibot20_seed1_router_alignment_20260712.py
```

State that the run is seed-1 diagnostic evidence and does not repair the five-seed claim.

### Task 2: Validate The Runner Before Training

**Files:**
- Verify: `LLMbot/run_twibot20_seed1_router_alignment_20260712.py`
- Verify: `LLMbot/experiments/twibot20_seed1_router_alignment_20260712/experiment_manifest.json`

**Interfaces:**
- Consumes: completed Task 1 runner.
- Produces: syntax, dry-run, route-count, and config evidence required before expensive training.

- [ ] **Step 1: Compile the runner**

Run:

```powershell
python -m py_compile run_twibot20_seed1_router_alignment_20260712.py
```

Expected: exit code `0`.

- [ ] **Step 2: Execute dry-run validation**

Run:

```powershell
python run_twibot20_seed1_router_alignment_20260712.py --dry-run
```

Expected: route counts `827/236/118`, both training commands present, no historical artifact path used as an output root, and manifest status `dry_run_completed`.

- [ ] **Step 3: Validate exact formula behavior**

Check that train targets exclude their own reference weight and that all reference indices are from `train_idx.pt`. Verify deterministic reruns produce identical selected node IDs and hashes.

### Task 3: Run Both Seed-1 Experiments

**Files:**
- Generate: `LLMbot/experiments/twibot20_seed1_router_alignment_20260712/**`

**Interfaces:**
- Consumes: validated runner and local CUDA environment.
- Produces: P-Paper and P-Hyperedge checkpoints, outputs, manifests, logs, and reports.

- [ ] **Step 1: Launch the full runner**

Run:

```powershell
python run_twibot20_seed1_router_alignment_20260712.py
```

Do not terminate while either graph run is active. Failed or OOM states must remain recorded in `experiment_manifest.json`.

- [ ] **Step 2: Verify generated graph contracts**

For both graph manifests verify seed 1, routed-only consumption, low-only fallback, residual fusion, K=8, fanout 64, 200 optimizer steps, and the expected route-mask path.

- [ ] **Step 3: Verify the P-Hyperedge router contract**

Verify `resolved_config.json` records `same_hyperedge` and the risk manifest records `same_hyperedge_selected_tail`.

### Task 4: Register And Interpret Evidence

**Files:**
- Modify: `NLPCC/rebuttal/REBUTTAL_EXPERIMENTS.md`
- Modify: `NLPCC/rebuttal/evidence_registry.md`
- Modify: `NLPCC/rebuttal/REBUTTAL_STATE.md`
- Generate: `LLMbot/experiments/twibot20_seed1_router_alignment_20260712/reports/seed1_router_alignment.csv`
- Generate: `LLMbot/experiments/twibot20_seed1_router_alignment_20260712/reports/seed1_router_alignment.json`

**Interfaces:**
- Consumes: validated manifests and canonical metrics from Task 3.
- Produces: provenance-bounded rebuttal evidence and the final comparison summary.

- [ ] **Step 1: Validate artifact hashes and metric rows**

Confirm every reported number maps to an `outputs.pt`, route mask, graph manifest, and exact command in the experiment manifest.

- [ ] **Step 2: Register evidence without upgrading scope**

Register the result as seed-1 diagnostic evidence. Keep P-Paper, P-Hyperedge, archived NCP, and fixed-weight route-swap rows separate.

- [ ] **Step 3: Report performance changes**

Report absolute percentage-point changes versus archived NCP and each arm's low-only path. Do not claim five-seed significance, superiority, or final-paper support from this run.

## Plan Self-Review

- Spec coverage: both approved experiment arms, artifact isolation, exact formula, explicit same-hyperedge configuration, canonical metrics, and rebuttal provenance are covered.
- Placeholder scan: no deferred implementation placeholders remain.
- Interface consistency: route-mask paths produced in Task 1 are the exact inputs consumed by Task 3 and registered in Task 4.

