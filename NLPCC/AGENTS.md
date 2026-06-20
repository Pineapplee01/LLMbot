# AGENTS.md - NLPCC Paper And Task Line

This file governs `NLPCC/` and every child path under it.

## Scope

`NLPCC/` is the active NLPCC paper and task line for the BotDetection workspace.
It stores lightweight manuscript source, figure and table source, writing
guidelines, migration notes, and artifact inventories.

`NLPCC/` is not the active code mainline. Code behavior changes belong in the
`LLMbot/` nested repository first, then the parent repository may update the
gitlink pointer intentionally.

## Edit Rules

- Keep source and governance files under `NLPCC/`.
- Keep manuscript source under `NLPCC/paper/`.
- Keep paper and task documentation under `NLPCC/docs/`.
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
