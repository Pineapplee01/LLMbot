# BotDetection NLPCC Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a governed top-level `NLPCC/` paper/task line, migrate only lightweight NLPCC source assets, update workspace governance, and verify no large/generated artifacts were tracked.

**Architecture:** The parent repository owns governance, documentation, lightweight paper source, and artifact inventories. `LLMbot/` remains the active code mainline and nested-repository boundary; NLPCC references its evidence through documented paths and manifests rather than copying code or generated outputs.

**Tech Stack:** Git, Markdown, LaTeX source files, PowerShell validation commands, Python standard library for deterministic inventory checks.

---

## File Structure

Create:

- `NLPCC/AGENTS.md` - scoped rules for future NLPCC paper/task work.
- `NLPCC/README.md` - onboarding for the NLPCC line.
- `NLPCC/manifest.json` - machine-readable inventory of migrated source roots and excluded artifact classes.
- `NLPCC/paper/main.tex` - copied lightweight LaTeX entrypoint.
- `NLPCC/paper/references.bib` - copied bibliography source.
- `NLPCC/paper/llncs.cls` - copied LNCS class file required by the source.
- `NLPCC/paper/splncs04.bst` - copied bibliography style required by the source.
- `NLPCC/paper/sections/*.tex` - copied manuscript section sources.
- `NLPCC/paper/figures/README.md` - copied figure provenance guide with paths normalized to `NLPCC/paper/figures`.
- `NLPCC/paper/figures/*.py`, `*.ps1`, `*.vsdx`, `*.svg`, `*.csv`, `*.json`, and selected `data/*.tex` - copied editable figure/table source and lightweight generated table includes.
- `NLPCC/docs/artifact_inventory.md` - inventory for local large/generated NLPCC files not tracked.
- `NLPCC/docs/llmbot_dependency_boundary.md` - boundary contract between NLPCC paper work and `LLMbot` code/evidence.
- `NLPCC/docs/migration_notes.md` - record of what was migrated, omitted, and why.

Modify:

- `.gitignore` - add `NLPCC/` allowlist entries for lightweight sources and explicit ignores for NLPCC generated artifacts.
- `AGENTS.md` - declare `NLPCC/` as active paper/task line while keeping `LLMbot/` as active code mainline.
- `README.md` - replace stale `LLMbot/baseline/` default onboarding with current code/paper surface map.
- `docs/README.md` - add `NLPCC/` to documentation navigation.
- `docs/CONVENTIONS.md` - add NLPCC routing and artifact conventions.
- `docs/ARCHITECTURE.md` - describe the parent workspace surfaces, including `NLPCC/`.
- `.agents/PLANS.md` - add planning triggers for NLPCC paper/evidence-boundary changes.
- `code.md` - update maintainability risk tracking to mention current `LLMbot/` code mainline and NLPCC boundary.

Do not modify:

- `LLMbot/` gitlink pointer or nested repository internals.
- `paper/NLPCC/` original local source tree.
- `datasets/`, `results/`, model caches, checkpoints, or experiment outputs.
- Historical wiki evidence pages except navigation/index wording if a later review explicitly requires it.

## Task 1: Create NLPCC Governed Source Tree

**Files:**

- Create: `NLPCC/AGENTS.md`
- Create: `NLPCC/README.md`
- Create: `NLPCC/manifest.json`
- Create: `NLPCC/paper/main.tex`
- Create: `NLPCC/paper/references.bib`
- Create: `NLPCC/paper/llncs.cls`
- Create: `NLPCC/paper/splncs04.bst`
- Create: `NLPCC/paper/sections/*.tex`
- Create: `NLPCC/paper/figures/README.md`
- Create: selected lightweight files under `NLPCC/paper/figures/`

- [ ] **Step 1: Create directories**

Run:

```powershell
New-Item -ItemType Directory -Force -Path `
  'NLPCC', `
  'NLPCC\paper', `
  'NLPCC\paper\sections', `
  'NLPCC\paper\figures', `
  'NLPCC\paper\figures\data', `
  'NLPCC\docs'
```

Expected: all directories exist; no files are modified.

- [ ] **Step 2: Copy lightweight paper source**

Run from `G:\Research\BotDetection\.worktrees\botdetection-nlpcc-reorg`:

```powershell
$src = 'G:\Research\BotDetection\paper\NLPCC'
$dst = 'G:\Research\BotDetection\.worktrees\botdetection-nlpcc-reorg\NLPCC\paper'
Copy-Item -LiteralPath "$src\main.tex" -Destination "$dst\main.tex"
Copy-Item -LiteralPath "$src\references.bib" -Destination "$dst\references.bib"
Copy-Item -LiteralPath "$src\llncs.cls" -Destination "$dst\llncs.cls"
Copy-Item -LiteralPath "$src\splncs04.bst" -Destination "$dst\splncs04.bst"
Copy-Item -LiteralPath "$src\sections\*.tex" -Destination "$dst\sections"
```

Expected: `NLPCC/paper/main.tex`, `references.bib`, `llncs.cls`, `splncs04.bst`, and all section `.tex` files exist.

- [ ] **Step 3: Copy lightweight paper governance notes**

Run:

```powershell
$src = 'G:\Research\BotDetection\paper\NLPCC'
$docs = 'G:\Research\BotDetection\.worktrees\botdetection-nlpcc-reorg\NLPCC\docs'
Copy-Item -LiteralPath "$src\NLPCC.md" -Destination "$docs\writing_guidelines.md"
Copy-Item -LiteralPath "$src\PAPER_PLAN.md" -Destination "$docs\paper_plan.md"
Copy-Item -LiteralPath "$src\claims_evidence_matrix.md" -Destination "$docs\claims_evidence_matrix.md"
Copy-Item -LiteralPath "$src\NARRATIVE_REPORT.md" -Destination "$docs\narrative_report.md"
Copy-Item -LiteralPath "$src\requirements_summary_2026.md" -Destination "$docs\requirements_summary_2026.md"
Copy-Item -LiteralPath "$src\PAPER_IMPROVEMENT_LOG.md" -Destination "$docs\paper_improvement_log.md"
```

Expected: copied docs use lowercase descriptive names under `NLPCC/docs/`.

- [ ] **Step 4: Copy figure/table source inputs**

Run:

```powershell
$src = 'G:\Research\BotDetection\paper\NLPCC\figures'
$dst = 'G:\Research\BotDetection\.worktrees\botdetection-nlpcc-reorg\NLPCC\paper\figures'
$rootFiles = @(
  'README.md',
  'Architecture.vsdx',
  'Architecture_fit.svg',
  'casestudy.vsdx',
  'casestudy.svg',
  'feature_encoder.svg',
  'feature_encoder.vsdx',
  'feature encoder.svg',
  'feature encoder.vsdx',
  'framework_overview_risk_gated_residual.svg',
  'framework_overview_selective_residual_detector.svg',
  'latex_includes.tex',
  'paper_plot_style.py',
  'gen_architecture_fit.py',
  'gen_main_results_tables.py',
  'gen_twibot20_component_ablation.py',
  'gen_twibot20_hparam_sensitivity.py',
  'prepare_casestudy_node11654.py',
  'gen_casestudy_fallback.py',
  'gen_casestudy_vsdx_package.py',
  'gen_casestudy_visio.ps1'
)
foreach ($name in $rootFiles) {
  Copy-Item -LiteralPath (Join-Path $src $name) -Destination (Join-Path $dst $name)
}
$dataFiles = @(
  'main_results.csv',
  'main_results_combined_table.tex',
  'main_results_twibot20_compact_table.tex',
  'main_results_twibot20_full_table.tex',
  'main_results_twibot22_compact_table.tex',
  'twibot20_component_ablation.csv',
  'twibot20_component_ablation_table.tex',
  'twibot20_hparam_sensitivity.csv',
  'twibot20_hparam_sensitivity_table.tex',
  'twibot20_risk_quality.csv',
  'twibot20_risk_quality_table.tex',
  'case_study_selected_nodes.csv',
  'case_study_support_nodes.csv',
  'casestudy_node11654_evidence.json'
)
foreach ($name in $dataFiles) {
  Copy-Item -LiteralPath (Join-Path $src "data\$name") -Destination (Join-Path $dst "data\$name")
}
```

Expected: no PDF, PNG, XLSX, pyc, backup CSV, or preview files are copied.

- [ ] **Step 5: Add `NLPCC/AGENTS.md`**

Write:

```markdown
# AGENTS.md - NLPCC Paper And Task Line

This file governs `NLPCC/` and every child path under it.

## Scope

`NLPCC/` is the active NLPCC paper and task line for the BotDetection workspace.
It stores lightweight manuscript source, figure/table source, writing
guidelines, migration notes, and artifact inventories.

`NLPCC/` is not the active code mainline. Code behavior changes belong in the
`LLMbot/` nested repository first, then the parent repository may update the
gitlink pointer intentionally.

## Edit Rules

- Keep source and governance files under `NLPCC/`.
- Keep manuscript source under `NLPCC/paper/`.
- Keep paper/task documentation under `NLPCC/docs/`.
- Do not commit PDFs, TeX build products, previews, rendered pages, portable
  TeX environments, datasets, checkpoints, model caches, or experiment outputs.
- Record large or generated local artifacts in
  `NLPCC/docs/artifact_inventory.md` instead of tracking them.
- Do not hand-edit experiment evidence. Reference `LLMbot/` manifests,
  summaries, and reproducibility notes.

## Validation

Before closing NLPCC work, verify:

- no generated artifact classes were added under `NLPCC/`;
- `NLPCC/paper/main.tex` references existing section sources;
- generated figure exports referenced by LaTeX are listed in the artifact
  inventory or regenerated from tracked source scripts;
- documentation changes do not promote unverified research claims.
```

- [ ] **Step 6: Add `NLPCC/README.md`**

Write:

```markdown
# NLPCC Paper And Task Line

This directory contains the governed NLPCC paper/task surface for the
BotDetection workspace.

## What Lives Here

- `paper/` - lightweight LaTeX source, section files, bibliography, and
  editable figure/table source.
- `docs/` - writing guidelines, paper plan, evidence mapping, migration notes,
  artifact inventory, and boundary notes.
- `manifest.json` - machine-readable summary of migrated source and excluded
  artifact classes.

## What Does Not Live Here

Generated PDFs, TeX logs, preview images, portable TeX runtimes, datasets,
checkpoints, model caches, and experiment outputs are not tracked here. They
are recorded in `docs/artifact_inventory.md` when relevant.

## Code Boundary

The active code mainline remains `../LLMbot/`. NLPCC work references LLMbot
experiment evidence through documented paths and manifests. Code refactors
must be committed in the nested LLMbot repository before the parent repository
records any gitlink update.
```

- [ ] **Step 7: Add `NLPCC/manifest.json`**

Write:

```json
{
  "name": "botdetection-nlpcc",
  "created": "2026-06-20",
  "role": "active_nlpcc_paper_task_line",
  "source_origin": "G:\\Research\\BotDetection\\paper\\NLPCC",
  "tracked_roots": [
    "NLPCC/AGENTS.md",
    "NLPCC/README.md",
    "NLPCC/docs/",
    "NLPCC/paper/main.tex",
    "NLPCC/paper/sections/",
    "NLPCC/paper/references.bib",
    "NLPCC/paper/figures/"
  ],
  "excluded_artifact_classes": [
    "compiled_pdf",
    "tex_build_logs",
    "tex_auxiliary_files",
    "preview_images",
    "rendered_pages",
    "portable_tex_runtime",
    "datasets",
    "checkpoints",
    "model_caches",
    "experiment_outputs"
  ],
  "code_boundary": {
    "active_code_mainline": "LLMbot/",
    "parent_tracking_mode": "gitlink",
    "code_changes_policy": "commit in nested LLMbot repository before parent gitlink updates"
  }
}
```

- [ ] **Step 8: Commit Task 1**

Run:

```powershell
git add -- NLPCC
git status --short
```

Expected: only `NLPCC/` files are staged for this task.

Commit message:

```text
Create governed NLPCC source surface

NLPCC paper work previously lived only in an ignored local paper tree, which
made source, generated artifacts, and local build environments hard to separate.
This commit introduces a tracked lightweight NLPCC surface while preserving the
original local tree.

Constraint: Large/generated NLPCC artifacts must be documented, not tracked
Rejected: Copy the whole paper/NLPCC tree | it contains generated PDFs, logs, previews, and portable TeX runtime files
Confidence: medium
Scope-risk: moderate
Tested: git status staged only NLPCC files
Not-tested: LaTeX compilation
```

## Task 2: Record Artifact Inventory And Ignore Policy

**Files:**

- Create: `NLPCC/docs/artifact_inventory.md`
- Create: `NLPCC/docs/llmbot_dependency_boundary.md`
- Create: `NLPCC/docs/migration_notes.md`
- Modify: `.gitignore`

- [ ] **Step 1: Add artifact inventory**

Create `NLPCC/docs/artifact_inventory.md` with this structure:

```markdown
# NLPCC Artifact Inventory

This inventory records local NLPCC artifacts that are intentionally not tracked.

## Excluded Artifact Classes

| Class | Local examples | Reason | Required action |
| --- | --- | --- | --- |
| Compiled manuscripts | `G:\Research\BotDetection\paper\NLPCC\main.pdf`, `reference_doi.pdf`, `main_*_update.pdf` | Generated paper outputs | Rebuild from `NLPCC/paper/main.tex` or inspect local source path |
| TeX build products | `*.log`, `*.aux`, `*.bbl`, `*.blg`, `*.fls`, `*.fdb_latexmk` | Reproducible build output | Do not track; regenerate during local compile |
| Preview/rendered pages | `_preview_*`, `_pdf_preview`, `rendered_pages` | Visual QA cache | Do not track; regenerate from PDFs |
| Portable TeX runtime | `.texenv/` | Large local environment | Recreate locally; do not commit |
| Figure PDF/PNG exports | `figures/Architecture.pdf`, `figures/casestudy.pdf`, `figures/*.png` | Generated from tracked scripts or editable sources | Regenerate or inspect local source path |
| Historical duplicate drafts | `main_round*.pdf`, `main_*_update.*` | Revision snapshots | Keep only in local archive unless promoted by a future plan |

## Active Generated Figure Exports Referenced By LaTeX

| LaTeX reference | Local source path | Tracked source | Regeneration note |
| --- | --- | --- | --- |
| `figures/Architecture.pdf` | `G:\Research\BotDetection\paper\NLPCC\figures\Architecture.pdf` | `NLPCC/paper/figures/Architecture.vsdx`, `NLPCC/paper/figures/gen_architecture_fit.py` | Regenerate through Visio or Python fallback |
| `figures/feature encoder.pdf` | `G:\Research\BotDetection\paper\NLPCC\figures\feature encoder.pdf` | `NLPCC/paper/figures/feature encoder.vsdx`, `NLPCC/paper/figures/feature encoder.svg` | Regenerate from editable figure source |
| `figures/twibot20_seed1_hparam_sensitivity.pdf` | `G:\Research\BotDetection\paper\NLPCC\figures\twibot20_seed1_hparam_sensitivity.pdf` | `NLPCC/paper/figures/gen_twibot20_hparam_sensitivity.py`, `NLPCC/paper/figures/data/twibot20_hparam_sensitivity.csv` | Run the tracked generation script in the LLMbot environment |
| `figures/casestudy.pdf` | `G:\Research\BotDetection\paper\NLPCC\figures\casestudy.pdf` | `NLPCC/paper/figures/casestudy.vsdx`, `NLPCC/paper/figures/gen_casestudy_vsdx_package.py` | Regenerate through Visio or the package builder |
```

- [ ] **Step 2: Add LLMbot boundary note**

Create `NLPCC/docs/llmbot_dependency_boundary.md`:

```markdown
# LLMbot Dependency Boundary

`NLPCC/` consumes evidence from `LLMbot/`, but it does not own executable
pipeline code.

## Rules

- Treat `LLMbot/` as the active code mainline and nested repository boundary.
- Do not copy training pipeline code into `NLPCC/`.
- Reference experiment evidence through paths, manifests, summaries, and
  reproducibility notes.
- Make code changes inside the nested `LLMbot` repository on its own branch.
- Update the parent gitlink only after the nested commit is intentionally
  selected.

## Current Known Boundary Risk

The parent repository records `LLMbot` as a gitlink, while the main checkout has
an independently checked out nested repository. Parent-level cleanup must not
silently update the gitlink pointer as a side effect of paper reorganization.
```

- [ ] **Step 3: Add migration notes**

Create `NLPCC/docs/migration_notes.md`:

```markdown
# NLPCC Migration Notes

## Source

Initial source material came from local ignored path:

`G:\Research\BotDetection\paper\NLPCC`

## Migrated

- LaTeX manuscript entrypoint and section source.
- Bibliography and LNCS style/class files required by the source.
- Writing guidelines, paper plan, evidence matrix, narrative notes, and
  improvement log.
- Editable and regenerable figure/table sources.

## Not Migrated

- Compiled PDFs.
- TeX build products.
- Preview and rendered-page images.
- Portable TeX runtime directories.
- Python bytecode.
- Duplicate revision snapshots and one-off probe files.

## Rationale

The tracked `NLPCC/` tree is intended to be reviewable and reproducible without
turning the parent repository into a build cache or artifact store.
```

- [ ] **Step 4: Update `.gitignore`**

Append an NLPCC block:

```gitignore

# NLPCC generated artifacts and local build state
NLPCC/**/*.pdf
NLPCC/**/*.png
NLPCC/**/*.log
NLPCC/**/*.aux
NLPCC/**/*.bbl
NLPCC/**/*.blg
NLPCC/**/*.fls
NLPCC/**/*.fdb_latexmk
NLPCC/**/.texenv/
NLPCC/**/_preview*/
NLPCC/**/rendered_pages/
NLPCC/**/__pycache__/
NLPCC/**/*.pyc
NLPCC/**/*.xlsx
```

Expected: source files such as `.tex`, `.bib`, `.cls`, `.bst`, `.md`, `.json`, `.csv`, `.py`, `.ps1`, `.vsdx`, and `.svg` remain trackable.

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
git add -- .gitignore NLPCC/docs/artifact_inventory.md NLPCC/docs/llmbot_dependency_boundary.md NLPCC/docs/migration_notes.md
git status --short
```

Expected: only `.gitignore` and the three docs are staged.

Commit message:

```text
Prevent NLPCC artifact drift from re-entering Git

The new NLPCC surface needs a clear line between source and local build state.
This commit records excluded artifact classes and teaches ignore rules about
NLPCC generated outputs.

Constraint: NLPCC LaTeX references generated figure PDFs that are not tracked
Rejected: Track figure PDF exports | they are generated and can be regenerated from tracked sources
Confidence: high
Scope-risk: narrow
Tested: git status staged only ignore and inventory docs
Not-tested: Full LaTeX build
```

## Task 3: Update Workspace Governance And Onboarding

**Files:**

- Modify: `AGENTS.md`
- Modify: `README.md`
- Modify: `docs/README.md`
- Modify: `docs/CONVENTIONS.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `.agents/PLANS.md`
- Modify: `code.md`

- [ ] **Step 1: Update root `AGENTS.md`**

Make these changes:

- Replace "`LLMbot/` is the only active mainline" with "`LLMbot/` is the active code mainline; `NLPCC/` is the active NLPCC paper/task line."
- Add `NLPCC/README.md` and `NLPCC/AGENTS.md` to quick entry points.
- Add `NLPCC/` to default research-wide edit zones.
- Add `NLPCC/docs/artifact_inventory.md` to evidence/artifact guidance.
- Keep `LLMbot/baseline/` and `LLMbot/code/` deprecated.

- [ ] **Step 2: Update root `README.md`**

Replace the stale baseline-centric default pipeline section with:

```markdown
## Active Surfaces

- `LLMbot/` - active code mainline for bot-detection pipeline work.
- `NLPCC/` - active NLPCC paper and task line.
- `docs/` - workspace architecture, protocols, guides, and wiki memory.
- `datasets/` and `results/` - evidence zones; do not hand-edit generated evidence.
- `LMBot/`, `botbr/`, `HyperScan/`, and `SEBot/` - comparison/reference lines.

## Quick Start

For code work, start with `AGENTS.md`, then `LLMbot/AGENTS.md` and
`LLMbot/README.md`.

For NLPCC paper work, start with `NLPCC/AGENTS.md`, `NLPCC/README.md`, and
`NLPCC/docs/artifact_inventory.md`.
```

Keep links to project protocols and wiki memory.

- [ ] **Step 3: Update docs navigation**

Update `docs/README.md` and `docs/CONVENTIONS.md` so both mention:

- current active code mainline: `LLMbot/`;
- current active NLPCC paper/task line: `NLPCC/`;
- artifact rule: large/generated NLPCC files are recorded in `NLPCC/docs/artifact_inventory.md`.

- [ ] **Step 4: Update `docs/ARCHITECTURE.md`**

Add a parent workspace section:

```markdown
## Workspace Surfaces

| Path | Role |
| --- | --- |
| `LLMbot/` | Active code mainline and nested repository boundary |
| `NLPCC/` | Active NLPCC paper/task line with lightweight manuscript source |
| `docs/` | Workspace protocols, architecture, guides, and durable memory |
| `datasets/` | Dataset metadata and ignored raw data zone |
| `results/` | Ignored generated evidence zone |
| `LMBot/`, `botbr/`, `HyperScan/`, `SEBot/` | Comparison/reference lines |
```

Keep the existing LLMbot runtime flow, but make clear it describes only the active code mainline.

- [ ] **Step 5: Update `.agents/PLANS.md`**

Add triggers:

```markdown
- The task changes `NLPCC/` manuscript structure, artifact inventory, evidence mapping, or paper-task boundaries.
- The task changes the boundary between `NLPCC/` paper claims and `LLMbot/` experiment/code evidence.
```

Update current governance:

```markdown
- `LLMbot/` is the active implementation mainline.
- `NLPCC/` is the active NLPCC paper/task line.
```

- [ ] **Step 6: Update `code.md`**

Add a short top section:

```markdown
# Code And Structure Risk Register

Current active code mainline: `LLMbot/`.
Current active NLPCC paper/task line: `NLPCC/`.

The older `LLMbot/baseline/` audit below is historical risk context and does
not redefine the current default code mainline.
```

Do not rewrite the historical risk table unless needed for clarity.

- [ ] **Step 7: Commit Task 3**

Run:

```powershell
git add -- AGENTS.md README.md docs/README.md docs/CONVENTIONS.md docs/ARCHITECTURE.md .agents/PLANS.md code.md
git status --short
```

Expected: only governance/onboarding docs are staged.

Commit message:

```text
Route workspace governance through LLMbot and NLPCC

The parent workspace now has a separate active code mainline and NLPCC paper
line. Updating governance prevents future sessions from routing paper tasks
through stale baseline defaults or treating NLPCC artifacts as ordinary source.

Constraint: Historical wiki evidence should not be rewritten as current routing
Rejected: Rewrite dated wiki records | they preserve historical experiment context
Confidence: medium
Scope-risk: moderate
Tested: git status staged only governance docs
Not-tested: Full governance drift script; validation follows in the next task
```

## Task 4: Validate Layout, Ignore Policy, And Governance Drift

**Files:**

- Modify only if validation fails: files from Tasks 1-3.

- [ ] **Step 1: Verify tracked NLPCC files**

Run:

```powershell
git ls-files NLPCC
```

Expected: output includes source/docs files and excludes `.pdf`, `.png`, `.log`, `.aux`, `.fls`, `.fdb_latexmk`, `.pyc`, `.xlsx`, and `.texenv` files.

- [ ] **Step 2: Verify no prohibited artifact classes are staged/tracked**

Run:

```powershell
$bad = git ls-files NLPCC | Where-Object { $_ -match '\.(pdf|png|log|aux|bbl|blg|fls|fdb_latexmk|pt|pth|pkl|ckpt|safetensors|npy|npz|bin|tar|zip|7z|rar|pyc|xlsx)$' -or $_ -match '(^|/)\\.texenv/' -or $_ -match '(^|/)__pycache__/' }
if ($bad) { $bad; exit 1 } else { 'no prohibited NLPCC tracked artifacts' }
```

Expected: `no prohibited NLPCC tracked artifacts`.

- [ ] **Step 3: Verify LaTeX source references**

Run:

```powershell
$root = 'G:\Research\BotDetection\.worktrees\botdetection-nlpcc-reorg'
$main = Join-Path $root 'NLPCC\paper\main.tex'
$text = Get-Content -LiteralPath $main -Raw
$missing = @()
foreach ($m in [regex]::Matches($text, '\\input\\{([^}]+)\\}')) {
  $path = Join-Path (Join-Path $root 'NLPCC\paper') ($m.Groups[1].Value + '.tex')
  if (-not (Test-Path -LiteralPath $path)) { $missing += $path }
}
if (-not (Test-Path -LiteralPath (Join-Path $root 'NLPCC\paper\references.bib'))) { $missing += 'NLPCC\paper\references.bib' }
if ($missing) { $missing; exit 1 } else { 'latex source references present' }
```

Expected: `latex source references present`.

- [ ] **Step 4: Verify generated figure exports are inventoried**

Run:

```powershell
$root = 'G:\Research\BotDetection\.worktrees\botdetection-nlpcc-reorg'
$sections = Get-ChildItem -LiteralPath (Join-Path $root 'NLPCC\paper\sections') -Filter '*.tex'
$refs = foreach ($file in $sections) {
  $text = Get-Content -LiteralPath $file.FullName -Raw
  foreach ($m in [regex]::Matches($text, '\\includegraphics(?:\\[[^\\]]*\\])?\\{([^}]+)\\}')) {
    $m.Groups[1].Value
  }
}
$inventory = Get-Content -LiteralPath (Join-Path $root 'NLPCC\docs\artifact_inventory.md') -Raw
$missing = $refs | Where-Object { $inventory -notmatch [regex]::Escape($_) }
if ($missing) { $missing; exit 1 } else { 'generated figure references inventoried' }
```

Expected: `generated figure references inventoried`.

- [ ] **Step 5: Verify default route wording**

Run:

```powershell
$files = @('AGENTS.md','README.md','docs/README.md','docs/CONVENTIONS.md','docs/ARCHITECTURE.md','.agents/PLANS.md','code.md')
$bad = Select-String -Path $files -Pattern 'LLMbot/baseline/.*default|baseline/.*default mainline|LLMbot/.*only active mainline|only active implementation mainline' -CaseSensitive:$false
if ($bad) { $bad; exit 1 } else { 'no stale default-route wording in active governance docs' }
```

Expected: `no stale default-route wording in active governance docs`.

- [ ] **Step 6: Verify worktree status**

Run:

```powershell
git status --short --branch
```

Expected: clean branch after all task commits.

- [ ] **Step 7: Commit validation fixes if needed**

If any previous validation step required fixes, commit them with a Lore-format message. If no fixes are needed, do not create an empty commit.

## Task 5: Final Review

**Files:**

- Read-only review of all changed files and validation output.

- [ ] **Step 1: Review full diff**

Run:

```powershell
git diff --stat HEAD~3..HEAD
git diff HEAD~3..HEAD -- . ':!NLPCC/paper/llncs.cls' ':!NLPCC/paper/splncs04.bst'
```

Expected: diff matches the plan; LNCS vendor files are excluded from manual prose review except existence checks.

- [ ] **Step 2: Check branch log**

Run:

```powershell
git log --oneline --decorate -5
```

Expected: includes the design commit and task commits on `codex-botdetection-nlpcc-reorg`.

- [ ] **Step 3: Report residual risks**

The final report must state:

- changed files and created directories;
- that no `LLMbot` gitlink update or nested-repository code edit was made;
- large/generated NLPCC artifacts were inventoried, not tracked;
- validation commands and results;
- remaining risk: LaTeX full compilation was not run unless a later task explicitly runs it.
