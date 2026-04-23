---
type: project
node_id: project:mainline_switch_context_2026-04-23
title: Mainline Switch Context
created_at: 2026-04-23T00:00:00Z
updated_at: 2026-04-23T00:00:00Z
tags: [mainline, switch, baseline, context, lmbot]
---

# Mainline Switch Context

## Purpose

This note is the 2026-04-23 truth card for wiki-level session routing. It records which code line current sessions should start from and which older notes remain historical reference material.

## Current Session Default

- Route current sessions to `LLMbot/baseline/`
- Use `LLMbot/baseline/core/main.py` as the default LMbot baseline entrypoint
- Use `LLMbot/baseline/analysis/` for diagnostics and `LLMbot/baseline/baselines/` for extra comparison baselines
- Use `docs/` as the authoritative doc root and `docs/wiki/` as durable project memory

## Historical Separation

- `LLMbot/code` contains the older rw1/rw4 active-method line
- The validated unified-protocol notes for that line remain historical experiment evidence
- Do not treat `LLMbot/code` as the current wiki-routed mainline after this switch
- The 2026-04-17 migration note remains useful as historical transition context, but current sessions should read this 2026-04-23 card first

## Comparison Guidance

- When formal comparison work uses `LLMbot/baseline/core`, explicitly control split behavior and Macro-F1 evaluation
- Do not project `LLMbot/code` audit conclusions onto `LLMbot/baseline/core`
- Keep reproduced and reported baseline statuses explicitly labeled in future tables and notes

## Use This Note When

Use this note when a future session needs to answer any of the following quickly:

- where a new session should start by default
- whether `LLMbot/code` is current routing or historical reference
- which truth card should anchor `query_pack.md` and `index.md`
- how to reconcile the 2026-04-17 migration note with the 2026-04-23 mainline switch
