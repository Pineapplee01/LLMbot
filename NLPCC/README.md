# NLPCC Paper And Task Line

This directory contains the governed NLPCC paper and task surface for the
BotDetection workspace.

## What Lives Here

- `paper/` - lightweight LaTeX source, section files, bibliography, and
  editable figure and table source.
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
