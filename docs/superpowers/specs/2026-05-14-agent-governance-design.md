# Agent Governance Design

Date: 2026-05-14

## Summary

This design records the accepted governance formula for the BotDetection research workspace:

- Superpowers controls whether design or implementation may start.
- OMX controls how approved plans are orchestrated and verified.
- Project protocols control research boundaries, artifact meaning, active mainline, and claim safety.

The accepted scope is the full governance surface: root governance, `LLMbot/` governance, protocol drift, planning rules, and repo-local skills.

## Decisions

- `LLMbot/` is the only active mainline for future implementation.
- `LLMbot/baseline/` and `LLMbot/code/` are deprecated legacy surfaces scheduled for deletion.
- Deprecated surfaces are default read-only and may be edited only for explicit migration, deletion, archival cleanup, or forensic comparison.
- Superpowers `brainstorming` is a hard gate before creative, behavioral, architectural, method, CLI, manifest, artifact, or research-boundary changes.
- OMX `$ralph` and `$team` are execution orchestration tools after design and plan approval; they must not bypass Superpowers gates or project protocols.
- Repo-local skills split research work into narrow owners: implementation, validation, experiment execution, analysis writing, and review.
- New test files are not created by default. Routine validation uses CLI args, smoke commands, manifest inspection, artifact checks, and governance drift checks unless the user explicitly approves test-code additions.

## Edited Governance Surfaces

- Root `AGENTS.md` defines project-wide authority, active mainline, edit zones, and workflow boundaries.
- `LLMbot/AGENTS.md` defines active-mainline coding rules, file creation rules, validation style, and research boundary gates.
- `docs/protocols/harness-standard.md` and `docs/protocols/agent-coding-guideline.md` remove stale `LLMbot/baseline/` default-mainline guidance.
- `.agents/PLANS.md` removes stale active plans and prevents new default plans from targeting `LLMbot/baseline/core`.
- `.agents/skills/*/SKILL.md` defines role-specific ownership without collapsing implementation, validation, experiment running, analysis, or review.

## Self-Review

- Placeholder scan: no TBD/TODO placeholders are intentionally left in the design.
- Consistency check: the design consistently treats `LLMbot/` as the active mainline and deprecated directories as read-only by default.
- Scope check: the scope is governance only; it does not modify research code or generated evidence.
- Ambiguity check: Superpowers, OMX, and project protocols have distinct responsibilities and conflict precedence.

## Validation

The implementation should be validated by:

- Searching governance documents for stale default-mainline references to `LLMbot/baseline/`.
- Checking `.agents/PLANS.md` for old active plans targeting `LLMbot/baseline/core`.
- Checking each repo-local `SKILL.md` has valid frontmatter with `name` and `description`.
- Checking role skills include explicit forbidden actions that preserve role boundaries.
- Confirming no source files or test files were created.
