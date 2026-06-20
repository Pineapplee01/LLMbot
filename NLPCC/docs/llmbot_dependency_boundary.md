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
