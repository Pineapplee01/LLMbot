# Query Pack

*Compressed project context for new sessions (target: keep compact)*

Last updated: 2026-04-23T00:00:00Z

---

## Project Direction

**Primary task**: Social bot detection on TwiBot-20  
**Primary metrics**: Accuracy, Macro-F1  
**Auxiliary metrics**: ECE/NLL/Brier/AURC for diagnostics only

**Current routing truth card**: `docs/wiki/project/mainline_switch_context_2026-04-23.md`

The project now has three context layers:

1. **Current session default**
   - Route current sessions to `LLMbot/baseline/`
   - Default entrypoint: `LLMbot/baseline/core/main.py`
   - Use the 2026-04-23 truth card for routing decisions

2. **Historical `LLMbot/code` record**
   - `LLMbot/code` contains the older rw1/rw4 active-method line
   - Keep its audit-backed notes as historical experiment evidence
   - Do not treat it as the current wiki-routed default

3. **Comparison protocol**
   - Docs live in `docs/`
   - Fair-comparison rules live in `docs/protocols/baseline_comparability.md`

Use `docs/` as the authoritative doc root and `docs/wiki/` as durable project memory.

---

## Code-Layer Truth

- `LLMbot/baseline/` is the current wiki-routed mainline for sessions
- `LLMbot/code` rw1/rw4 results are historical validated unified-protocol records from the earlier active-method line
- `LLMbot/baseline/core` is a runnable migrated baseline, not a guaranteed unified-protocol output by default
- `LLMbot/baseline/core` is mostly aligned with `docs/published/lmbot-wsdm2024/src`, but `GNNs.py` and `parser_args.py` already differ from the published snapshot
- `docs/wiki/project/baseline_migration_comparison_context_2026-04-17.md` remains the historical migration note for the earlier layout change

**Formal comparison caution**:
- `baseline/core/main.py` will reset splits unless controlled explicitly
- `baseline/core/trainer.py` uses default binary `f1_score()` calls
- Therefore formal comparisons with the migrated baseline must explicitly enforce canonical split usage and Macro-F1 evaluation discipline

---

## Baseline Anchors

- LMbot RGCN GNN baseline: Macro-F1 `0.8723` reproduced
- LMbot LM co-trained baseline: Macro-F1 `0.8771` reproduced
- LMbot RGT single-pass baseline: Macro-F1 `0.8619` reproduced
- BotBR: about `86.8%` reported or reproduction-in-progress
- HyperScan: `87.2%` reported unless local reproduction is completed and documented

Reference docs:
- `docs/wiki/project/mainline_switch_context_2026-04-23.md`
- `docs/wiki/experiments/lmbot_gnn_lm_baselines.md`
- `docs/wiki/experiments/lmbot_rgt_baseline.md`
- `docs/wiki/project/baseline_migration_comparison_context_2026-04-17.md`

---

## Historical `LLMbot/code` Anchors

- rw1_calibration_only: Macro-F1 `0.7391`, ECE `0.0738`
- rw4_disagreement_local: Macro-F1 `0.7603`, ECE `0.0190`
- kill test B2: repair beats deferral on the disagreement slice
- kill test B3: utility-guided pruning components matter

Important distinction:
- those validated rw1/rw4 numbers are historical notes for the `LLMbot/code` line documented in `docs/wiki/experiments/baseline_audit_correction.md`
- do not project that audit result onto `LLMbot/baseline/core`

---

## Collaboration Preference

- Default to research quality over speed or token economy.
- Confirm high-impact uncertainty with the user before locking assumptions.
- Distinguish verified facts from inference when conclusions depend on the difference.
- Keep research code changes minimal, clear, and easy to validate.
- Prefer using `$prioritize-research-quality` for research-facing tasks in this project.

## Comparison Protocol

All methods in the main comparison table must share:

- canonical TwiBot-20 split files from `datasets/TwiBot-20/`
- identical node ordering
- Accuracy and Macro-F1 as the main metrics
- validation-only tuning discipline

If a baseline is not fully reproduced locally:
- keep it in the comparison story
- mark it clearly as `reported` or `reproduction in progress`

---

## Current Open Work

1. Finalize the comparison matrix across the baseline-routed current work, historical `LLMbot/code` notes, BotBR, and HyperScan handling
2. Run or wrap migrated LMbot baseline experiments under explicit canonical-split and Macro-F1 discipline
3. Keep reproduced and reported baseline statuses clearly separated in future tables
4. Avoid mixing historical `LLMbot/code` audit conclusions and `baseline/core` default behavior


