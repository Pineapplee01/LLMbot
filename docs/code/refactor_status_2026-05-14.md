# LLMbot Refactor Status - 2026-05-14

This is the stop/resume note for the current `LLMbot/` root-mainline cleanup. It records what has been changed, what was verified, and what remains intentionally unfinished.

## Current Task Boundary

The active implementation surface is `LLMbot/` root.

`LLMbot/baseline/` and `LLMbot/code/` are deprecated legacy/reference surfaces scheduled for deletion. They were treated as read-only during this cleanup except for pre-existing noisy git state.

The cleanup goal was:

- merge `frozen_g0.py` responsibilities into existing active files;
- remove stale helper imports from active root Python files;
- normalize parser aliases and boolean parsing;
- standardize stage/artifact return contracts;
- document current parser/mainline development rules;
- repair current guidance docs that still pointed future agents to `LLMbot/baseline/`.

## Completed Code Changes

Active code changes already applied:

- `LLMbot/frozen_g0.py` was deleted.
- Frozen G0 train/load/reuse logic now lives in `LLMbot/trainer.py`.
- `PhaseAInputAdapter`, feature resolution, scoring, projection, and selector compatibility helpers now live in `LLMbot/model_building.py`.
- Shared artifact/checkpoint/data helpers are exposed through the `LLMbot/utils/` package, especially `LLMbot/utils/__init__.py`.
- `LLMbot/main.py` no longer imports `frozen_g0`, `artifacts`, `checkpoint_io`, or `faithful_gates`.
- `LLMbot/trainer.py` owns the minimal faithful GATS gate/calibrator logic needed by active stages.
- `LLMbot/parser_args.py` now has `normalize_args(args)`.
- `--is_processed` uses compatibility boolean parsing instead of `type=bool`.
- `--emb_path` / `--g0_feature_path` and `--risk_budgets` / `--router_budgets` are normalized with the new parameter taking priority.
- `LLMbot/RGT.py` no longer prints `beta`; it stores detached attention as `self.last_beta`.
- `LLMbot/LM.py` removed the unused masked-LM import and now raises a model-specific `ValueError`.

No new source files or new test files were created for this cleanup.

## Completed Documentation Changes

New or updated current guidance:

- `docs/code/parser.md` added parser and active-mainline contract.
- `LLMbot/AGENTS.md` now declares `LLMbot/` root as the only active mainline and removes the deleted `frozen_g0.py` from editable source guidance.
- `LLMbot/README.md` documents root-mainline layout and deprecates `baseline/` / `code/`.
- `docs/README.md` points current implementation links to `LLMbot/`.
- `docs/CONVENTIONS.md` points current defaults to `LLMbot/`.
- `docs/guides/codex.md` and `docs/guides/claude.md` point agent adapters to `LLMbot/`.
- `docs/guides/model.md` was rewritten as a root-mainline model guide.
- `docs/ARCHITECTURE.md` was rewritten as a root-mainline architecture guide.

Historical docs such as `docs/wiki/**`, old `docs/superpowers/plans/**`, and implementation snapshots were not rewritten. They still contain historical `LLMbot/baseline/` references by design.

## Validation Already Passed

Compilation:

```powershell
D:\Anaconda\python.exe -m py_compile LLMbot\main.py LLMbot\parser_args.py LLMbot\trainer.py LLMbot\model_building.py LLMbot\GNNs.py LLMbot\RGT.py LLMbot\LM.py LLMbot\SimpleHGN.py LLMbot\utils\__init__.py LLMbot\llm_evidence_refiner.py
```

Import smoke:

```powershell
D:\Anaconda\python.exe -c "import sys; sys.path.insert(0, r'G:\Research\BotDetection\LLMbot'); import utils; print(utils.__file__); import main, trainer, model_building, GNNs, models; print('ok')"
```

CLI help:

```powershell
cd G:\Research\BotDetection\LLMbot
D:\Anaconda\python.exe main.py --help
```

Parser alias smoke:

```powershell
D:\Anaconda\python.exe -c "import sys; sys.path.insert(0, r'G:\Research\BotDetection\LLMbot'); import parser_args; sys.argv=['main.py','--g0_feature_path','x.pt','--router_budgets','0.2','--is_processed','false']; a=parser_args.parser_args(); print(a.seeds, a.emb_path, a.g0_feature_path, a.risk_budgets, a.router_budgets, a.is_processed)"
```

Observed output:

```text
1,2,3,4,5 x.pt x.pt 0.2 0.2 False
```

Recommended-arg smoke:

```powershell
D:\Anaconda\python.exe -c "import sys; sys.path.insert(0, r'G:\Research\BotDetection\LLMbot'); import parser_args; sys.argv=['main.py','--seeds','3','--emb_path','new.pt','--risk_budgets','0.3','--is_processed','true']; a=parser_args.parser_args(); print(a.seeds, a.emb_path, a.g0_feature_path, a.risk_budgets, a.router_budgets, a.is_processed)"
```

Observed output:

```text
3 new.pt new.pt 0.3 0.3 True
```

Stale active import search passed on all non-legacy Python files under `LLMbot/`:

- no `from frozen_g0` / `import frozen_g0`;
- no `checkpoint_io`, `artifacts`, `frmi_selectors`, or `faithful_gates` imports;
- no `print(beta)`;
- no `AutoModelForMaskedLM`.

No new non-legacy `test_*.py` files were found.

## Subagent Status

Subagent-driven development was used only as a sidecar review surface.

- One explore sidecar failed earlier due context-window exhaustion.
- One code-review sidecar completed and raised risks that were checked locally.
- A final reviewer sidecar was launched but timed out twice while the main lane continued verification. It was closed; the later notification reported the agent path as `shutdown`.

There are no known subagents still running for this cleanup.

## Known Remaining Work

### 1. `LLMbot/.gitignore` Reviewed And Updated

`LLMbot/.gitignore` was rewritten as an ASCII root-mainline whitelist. It no longer re-includes `baseline/**/*.py`, `baseline/**/*.md`, or `code/*.py`.

Next step: handle already tracked or staged legacy files as a separate deletion/archival task. `.gitignore` alone does not untrack files that were already tracked or staged.

### 2. Active Code Is Still Large And Needs Modular Refactor

The cleanup fixed boundary/import problems, but active files remain large:

- `LLMbot/trainer.py`
- `LLMbot/estimators.py`
- `LLMbot/operators.py`

Next refactor should be interface-preserving and one concern at a time. Do not create new source files unless a design record explicitly approves the path and ownership.

### 3. `utils.py` vs `utils/` Package Shadowing Resolved

Python imports the `LLMbot/utils/` package when `LLMbot` is on `sys.path`.

The active public utility surface is `LLMbot/utils/__init__.py`. The root `LLMbot/utils.py` file was removed to prevent future import ambiguity.

### 4. Legacy Directories Still Contain Stale Imports And Tests

Recursive searches under `LLMbot/baseline/` and `LLMbot/code/` still show old helper files, stale imports, and historical tests. This is expected because those directories are deprecated and were not edited as part of this cleanup.

Next step: handle legacy deletion/archival as a separate explicit task.

### 5. Historical Documentation Still Mentions `baseline/`

The current guidance docs were updated. Historical memory and old plan docs were not:

- `docs/wiki/**`
- `docs/superpowers/plans/**`
- old implementation snapshots under `docs/implementation/**`
- remote GPU notes under `docs/guides/REMOTE_GPU_SERVER.md`

Next step: if desired, add a short deprecation banner to those historical docs rather than rewriting their original facts.

### 6. Full Training Was Not Run

No full TwiBot training or GPU run was executed in this cleanup. Verification was limited to compile/import/CLI/parser/stale-import checks.

Next validation level should use a bounded smoke command with explicit args once data/GPU availability is confirmed.

## Suggested Resume Order

1. Re-run the validation commands in this document to confirm the stop point still holds.
2. Inspect `LLMbot/.gitignore` and nested git status before touching legacy directories.
3. Decide the fate of root `LLMbot/utils.py` versus `LLMbot/utils/`.
4. Plan the first size-reduction refactor for `trainer.py`, keeping public CLI and artifact contracts unchanged.
5. Only then start deleting or archiving `LLMbot/baseline/` and `LLMbot/code/`.
