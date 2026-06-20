# BotDetection NLPCC Reorganization Design

Date: 2026-06-20

## Summary

This design defines a staged reorganization for the BotDetection workspace.
The goal is to make `NLPCC/` a first-class, governed paper and task line
without importing generated artifacts, large files, or nested-repository
history into the parent repository by accident.

The accepted direction is a conservative split:

- `LLMbot/` remains the active code mainline and is governed as a nested
  repository / gitlink.
- `NLPCC/` becomes the active NLPCC paper and task directory at the parent
  workspace level.
- NLPCC consumes evidence from `LLMbot/` experiments and manifests, but does
  not copy training code, checkpoints, model caches, or experiment outputs.
- Large NLPCC PDFs, logs, preview pages, portable TeX environments, and build
  products are documented, not moved or tracked.

## Evidence

Repository inspection found:

- The parent repository worktree is
  `G:\Research\BotDetection\.worktrees\botdetection-nlpcc-reorg`.
- The branch is `codex-botdetection-nlpcc-reorg`.
- `LLMbot`, `LMBot`, `botbr`, `HyperScan`, and `SEBot` are tracked by the
  parent repository as gitlinks with mode `160000`, not ordinary directories.
- The parent repository has no `.gitmodules` file, so submodule metadata and
  source URLs are not currently encoded in a normal superproject mapping.
- The main checkout's nested `LLMbot` repository is on branch `llmbot` at
  commit `e769811`, while the parent repository records `LLMbot` at commit
  `c733dce`. This is an existing pointer drift and must not be silently
  resolved by unrelated reorganization work.
- The tracked parent tree contains no `NLPCC` / `nlpcc` files or references.
- The current NLPCC material lives under ignored local path
  `G:\Research\BotDetection\paper\NLPCC`.
- That NLPCC directory contains lightweight source files such as `main.tex`,
  `sections/*.tex`, `references.bib`, `NLPCC.md`, `PAPER_PLAN.md`,
  `claims_evidence_matrix.md`, figure scripts, figure source files, and
  table data.
- The same NLPCC directory also contains generated PDFs, `.log`, `.aux`,
  `.fls`, preview images, rendered pages, backup copies, and a large portable
  TeX environment under `.texenv/`.

## Non-Goals

This design does not:

- rewrite `LLMbot` source code inside the parent repository;
- update the parent repository's `LLMbot` gitlink pointer;
- import raw datasets, checkpoints, model downloads, experiment outputs, or
  generated paper PDFs;
- normalize historical wiki pages that intentionally record earlier routing
  states;
- replace the current active code mainline with `NLPCC/`.

## Target Structure

The parent repository should gain:

```text
NLPCC/
|-- AGENTS.md
|-- README.md
|-- manifest.json
|-- paper/
|   |-- main.tex
|   |-- references.bib
|   |-- llncs.cls
|   |-- splncs04.bst
|   |-- sections/
|   |-- figures/
|-- docs/
|   |-- artifact_inventory.md
|   |-- llmbot_dependency_boundary.md
|   |-- migration_notes.md
```

The exact file set may be narrowed during implementation if inspection shows a
candidate is generated or duplicate. The invariant is that tracked files must
be source, governance, scripts, lightweight metadata, or editable figure/table
inputs.

## NLPCC File Classification

Move or copy into `NLPCC/`:

- `main.tex`;
- `sections/*.tex`;
- `references.bib`;
- required LNCS template files used by the draft, currently `llncs.cls` and
  `splncs04.bst`;
- current paper-governance notes: `NLPCC.md`, `PAPER_PLAN.md`,
  `claims_evidence_matrix.md`, `NARRATIVE_REPORT.md`,
  `requirements_summary_2026.md`, and `PAPER_IMPROVEMENT_LOG.md`;
- figure/table source files that are required to rebuild or edit figures,
  including scripts, `.vsdx`, `.svg`, `.csv`, `.json`, and small generated
  table `.tex` includes when they are treated as editable paper inputs.

Document but do not move or track:

- generated PDF outputs, including compiled paper snapshots and figure exports;
- `.log`, `.aux`, `.bbl`, `.blg`, `.fls`, `.fdb_latexmk`, and other build
  products;
- preview and rendered-page image directories;
- `_tmp_*` extracts, probe files, and one-off build text dumps;
- `.texenv/` and any portable TeX / Perl / MiKTeX runtime content;
- `__pycache__/` and other Python bytecode;
- duplicate or stale revision copies unless a later plan explicitly promotes
  one as source.

## Governance Updates

Update the parent workspace governance so the structure is explicit:

- `AGENTS.md`: define `LLMbot/` as the active code mainline and `NLPCC/` as the
  active NLPCC paper/task line; add `NLPCC/` to normal edit zones; require
  artifact inventory for large local NLPCC files.
- `README.md`: add top-level onboarding for code work vs NLPCC paper work and
  remove stale claims that route default work to `LLMbot/baseline/`.
- `docs/README.md` and `docs/CONVENTIONS.md`: add `NLPCC/` as a first-class
  documented project surface.
- `docs/ARCHITECTURE.md`: describe the parent workspace as a collection of
  governed lines: active code mainline, NLPCC paper/task line, comparison
  baselines, datasets, and generated evidence zones.
- `.agents/PLANS.md`: add a planning trigger for NLPCC paper structure,
  evidence mapping, or cross-boundary `LLMbot` dependency changes.
- `code.md`: update maintainability tracking so it reflects the current
  active code mainline and the new NLPCC boundary instead of focusing only on
  stale `LLMbot/baseline` risk.

Historical wiki files may be left unchanged when they are clearly dated
records. If a wiki index is updated, it should label older `LLMbot/baseline`
and `LLMbot/code` references as historical routing notes rather than rewrite
their evidence claims.

## LLMbot Boundary

`LLMbot/` code-level refactoring is a separate nested-repository task.

Rules:

- Do not edit `LLMbot` internals through the parent repository worktree as if
  they were ordinary tracked files.
- If code must be changed, create or use a branch inside
  `G:\Research\BotDetection\LLMbot`.
- Commit code changes in the nested repository first.
- Update the parent gitlink pointer only after the nested commit is reviewed
  and intentionally selected.
- The NLPCC paper should cite `LLMbot` evidence through experiment paths,
  manifests, and reproducibility notes, not by copying executable pipeline
  code into `NLPCC/`.

## Artifact Policy

`NLPCC/docs/artifact_inventory.md` must record large local artifacts that are
not tracked. Each entry should include:

- local source path;
- artifact class;
- approximate size;
- whether it is generated, external, or source-like but intentionally untracked;
- regeneration or retrieval note;
- whether it is required to compile, inspect, or reproduce the NLPCC paper.

`NLPCC/AGENTS.md` must state that future work should not commit PDFs, TeX
build products, model checkpoints, datasets, generated previews, or local TeX
runtime directories. If a large artifact is essential, track a manifest or
inventory entry instead.

## Validation Strategy

Implementation should be validated by:

- `git status --short --branch`;
- `git ls-files NLPCC`;
- `git check-ignore -v` samples for generated NLPCC files that must remain
  ignored or untracked;
- search for stale default-mainline wording in root and active governance docs;
- inspection that no `.pdf`, `.log`, `.aux`, `.fls`, `.fdb_latexmk`, `.pt`,
  `.pth`, `.pkl`, `.ckpt`, `.safetensors`, `.npy`, `.npz`, `.bin`, `.tar`,
  `.zip`, `.7z`, or `.rar` files were added under `NLPCC/`;
- lightweight LaTeX source sanity check by confirming `NLPCC/paper/main.tex`
  references source files present under `NLPCC/paper/sections/` and
  `NLPCC/paper/references.bib`, while generated figure exports referenced by
  `\includegraphics` are listed in `NLPCC/docs/artifact_inventory.md`;
- governance drift checks for `LLMbot/baseline` as a default route.

No new test files are required for this parent-repository documentation and
layout change. Code behavior checks belong to the nested `LLMbot` task if and
when code is changed.

## Self-Review

- Placeholder scan: no unfinished placeholder markers are intentionally left.
- Consistency check: the design keeps code and paper ownership separate and
  does not claim `NLPCC/` replaces the active code mainline.
- Scope check: the parent-repository phase is structure, governance, and
  lightweight source migration only; nested `LLMbot` code refactoring is
  explicitly separated.
- Ambiguity check: large/generated files are documented by class and are not
  imported by default.
