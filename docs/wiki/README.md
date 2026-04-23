# LMBot Research Wiki - Persistent Project Memory

**Date**: 2026-04-22  
**Status**: Active project memory layer

---

## Purpose

This wiki is the persistent project memory for the LMBot research workspace. It is the place future sessions should use to recover:
- the current project context
- experiment history and reusable findings
- background notes and decision rationale
- comparison status across method lines

The authority chain for this project is:
1. `docs/protocols/harness-standard.md`
2. `docs/protocols/agent-coding-guideline.md`
3. `docs/protocols/baseline_comparability.md` for formal comparison work
4. `docs/wiki/` as durable memory

---

## Rules

1. Keep `docs/` as the authoritative project doc root.
2. Keep `docs/wiki/` as project memory, not as a second policy source.
3. Preserve historical experiment facts; do not rewrite old experiment files to fit a new narrative.
4. Use append-only updates in `log.md` for context corrections and new state.
5. When wiki notes and verified traces, manifests, or protocol docs disagree, the verified evidence wins.

---

## Use These Entry Files First

- `index.md` for the high-level categorized state
- `query_pack.md` for a compact session bootstrap
- `log.md` for append-only history and clarifications
- `project/` for project-level decision and migration notes
