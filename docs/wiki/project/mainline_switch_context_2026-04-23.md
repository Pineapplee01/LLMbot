---
type: project
node_id: project:mainline_switch_context_2026-04-23
title: Mainline Switch Context
created_at: 2026-04-23T00:00:00Z
updated_at: 2026-05-23T00:00:00Z
tags: [mainline, switch, active-mainline, llmbot]
---

# Mainline Switch Context

## Purpose

This note is the routing truth card for current sessions. It records which code
line new work should start from and how older LMbot lines should now be treated.

## Current Session Default

- Route current sessions to `LLMbot/`
- Use `LLMbot/main.py` as the default entrypoint
- Use `LLMbot/parser_args.py` as the public CLI contract
- Use `LLMbot/README.md` as the mainline operator guide
- Use `docs/` as the authoritative doc root and `docs/wiki/` as durable memory

## Historical Separation

- `LLMbot/baseline/` is a deprecated legacy baseline surface
- `LLMbot/code/` is a deprecated historical active-method surface
- neither directory is the current wiki-routed mainline
- older migration notes remain useful as historical transition context, but
  current sessions should start from this card plus `LLMbot/AGENTS.md`

## Comparison Guidance

- formal comparison work may still inspect historical surfaces, but only as
  reference lines
- do not route implementation, parser, or architecture work back into
  `LLMbot/baseline/` or `LLMbot/code/`
- keep reproduced and reported baseline statuses explicitly separated

## Use This Note When

Use this note when a session needs to answer:

- where active implementation now lives
- whether `LLMbot/baseline/` or `LLMbot/code/` are current or historical
- which routing card should anchor `query_pack.md` and `index.md`
- how to interpret older baseline-routed notes after the mainline switch
