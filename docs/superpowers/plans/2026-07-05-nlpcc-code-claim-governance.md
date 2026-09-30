# NLPCC Code And Claim Governance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or inline TDD execution. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the NLPCC submission active code line into `NLPCC/code/` and add a shallow claim-classification structure for the remaining `LLMbot/code/` research code without running training or editing generated artifacts.

**Architecture:** `NLPCC/code/` is a complete source snapshot of the current active mainline needed to replay the NLPCC submission path. `LLMbot/code/` keeps its flat import-compatible modules for now, while `LLMbot/code/claims/` records claim ownership as a governance layer rather than moving modules and breaking absolute imports in the same slice.

**Tech Stack:** Python, pytest for structural tests, existing LLMbot flat-module imports, CodeGraph evidence from `LLMbot/.codegraph/codegraph.db`.

## Global Constraints

- Use TDD: add failing structure/import tests before implementation.
- Do not run training or experiments.
- Do not edit `LLMbot/experiments/`, `LLMbot/server_logs/`, checkpoints, saved artifacts, datasets, or generated manifests.
- Keep existing `LLMbot/code/*.py` import-compatible in this slice.
- Do not introduce `stages/` or `methods/` directories.
- Use clear claim and module names; legacy names appear only as compatibility references.

---

### Task 1: Add Migration Contract Tests

**Files:**
- Create: `tests/test_nlpcc_code_migration_contract.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `LLMbot/code/stage_registry.py`, `LLMbot/main.py`
- Produces: a failing RED contract for `NLPCC/code/` and claim folders

- [ ] **Step 1: Write the failing test**

Create tests that assert:

- `NLPCC/code/main.py`, `parser_args.py`, `stage_registry.py`, `stage_runner.py`, and `utils/` exist.
- `NLPCC/code` imports parser and registry without training imports.
- `python NLPCC/code/main.py --help` and `python LLMbot/main.py --help` expose `--experiment_task`.
- `NLPCC/code/claims/` and `LLMbot/code/claims/` contain shallow claim folders.
- generated artifact directories and binary artifacts are absent under `NLPCC/code/`.

- [ ] **Step 2: Verify RED**

Run:

```powershell
D:\Anaconda\python.exe -m pytest tests\test_nlpcc_code_migration_contract.py -q
```

Expected before implementation: FAIL because `NLPCC/code/` does not exist.

### Task 2: Build `NLPCC/code/` Source Snapshot

**Files:**
- Create: `NLPCC/code/`
- Create: `NLPCC/code/utils/`
- Create: `NLPCC/code/runners/`
- Create: `NLPCC/code/claims/`
- Copy: current active source files from `LLMbot/code/`
- Copy: NLPCC-relevant root runner/helper scripts from `LLMbot/`

**Interfaces:**
- Consumes: current flat `LLMbot/code` modules
- Produces: import-compatible `NLPCC/code` source root

- [ ] **Step 1: Copy active source only**

Copy `LLMbot/code/*.py` and `LLMbot/code/utils/*.py` to `NLPCC/code/`, excluding `__pycache__`.

- [ ] **Step 2: Copy NLPCC runner surface**

Copy root helper/runner scripts into `NLPCC/code/runners/`:

- `extract_raw_roberta_embeddings_20260620.py`
- `launch_sampled_twibot22_base5_20260620.py`
- `run_sampled_twibot22_base_5seed_20260620.py`
- `run_twibot20_formal5_ablation_completion_20260620.py`
- `run_twibot20_full_selective_residual_5seed_20260620.py`
- `summarize_formal_runs_20260620.py`

- [ ] **Step 3: Add source-boundary docs**

Create `NLPCC/code/README.md` and `NLPCC/code/CLAIM_MAP.md` describing the NLPCC active-mainline code path and why this is a source snapshot, not generated evidence.

### Task 3: Add Claim Classification Folders

**Files:**
- Create: `NLPCC/code/claims/*/README.md`
- Create: `LLMbot/code/claims/*/README.md`
- Create: `LLMbot/code/shared/README.md`

**Interfaces:**
- Consumes: `stage_registry.py` task families and current owner modules
- Produces: shallow claim ownership map without moving production modules

- [ ] **Step 1: Create shallow claim folders**

Use these folders:

- `phase_a_foundations`
- `high_order_graph_consumption`
- `conformal_risk_routing`
- `routed_llm_evidence_refinement`
- `local_graph_repair_diagnostics`
- `semantic_candidate_correction`
- `ablation_positioning_legacy`

- [ ] **Step 2: Record file-to-claim mapping**

Each folder README names its claim, owned modules, shared dependencies, and migration note.

### Task 4: Verify Green

**Files:**
- Modify only if tests fail: files from Tasks 1-3

- [ ] **Step 1: Run migration contract**

```powershell
D:\Anaconda\python.exe -m pytest tests\test_nlpcc_code_migration_contract.py -q
```

- [ ] **Step 2: Run import/help smoke checks**

```powershell
D:\Anaconda\python.exe LLMbot\main.py --help
D:\Anaconda\python.exe NLPCC\code\main.py --help
```

- [ ] **Step 3: Run artifact scope guard**

```powershell
git status --porcelain -- datasets results LLMbot\experiments LLMbot\server_logs LLMbot\saved_artifacts LLMbot\checkpoints
```

Expected: no newly modified artifact paths from this task.

## Self-Review

- Spec coverage: covers `NLPCC/code`, claim folders, TDD, source-only migration, and artifact exclusions.
- Placeholder scan: no TODO/TBD placeholders.
- Type consistency: tests consume existing parser and stage registry interfaces only.
