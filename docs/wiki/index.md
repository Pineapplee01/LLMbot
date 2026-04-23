# Research Wiki Index

*Categorical index of the current project state*

Last updated: 2026-04-23T00:00:00Z

---

## Project Mission

**Primary task**: Social bot detection on TwiBot-20  
**Primary metrics**: Accuracy, Macro-F1  
**Auxiliary metrics**: ECE/NLL/Brier/AURC for diagnostics only

**Current project framing**:
- current wiki-level session routing starts from `LLMbot/baseline/`
- `LLMbot/code` notes remain historical experiment records, not the current default route
- all formal comparisons must follow the unified protocol under `docs/protocols/`

---

## Project Context

### Code Layers

- **Current session default**: `LLMbot/baseline/core/main.py`
- **Current baseline diagnostics**: `LLMbot/baseline/analysis/`
- **Current extra comparison baselines**: `LLMbot/baseline/baselines/`
- **Historical active-method line**: `LLMbot/code/`

### Documentation Layers

- **Authoritative docs root**: `docs/`
- **Persistent memory**: `docs/wiki/`
- **Current routing truth card**: `docs/wiki/project/mainline_switch_context_2026-04-23.md`
- **Historical migration card**: `docs/wiki/project/baseline_migration_comparison_context_2026-04-17.md`
- **Published LMbot snapshot**: `docs/published/lmbot-wsdm2024/`

### Research Phase Vocabulary

Use [project_phase_taxonomy.md](../research/project_phase_taxonomy.md) for current A-G phase naming. These labels are planning vocabulary, not CLI `--stage` names, and the user may override the current phase explicitly.

| Phase | Canonical Name |
| --- | --- |
| A | 阶段 A：semantic encoder → GNN |
| B | 阶段 B：GNN 输出 → post-hoc estimator |
| C | 阶段 C：risk/regime → local semantic enhancement |
| D | 阶段 D：risk/regime → local propagation repair |
| E | 阶段 E：action selection |
| F | 阶段 F：same-head refinement |
| G | 阶段 G：positioning / stress test |

### Baseline Snapshot Status

- `LLMbot/baseline/core` is mostly aligned with `docs/published/lmbot-wsdm2024/src`
- `GNNs.py` and `parser_args.py` differ from the published snapshot
- Therefore the migrated baseline should be treated as a **runnable migrated baseline**, not a completely untouched release mirror

---

## Project Policies

1. **Primary evaluation policy**: main comparison tables use Accuracy and Macro-F1
2. **Auxiliary metrics policy**: ECE/NLL/Brier/AURC are diagnostic only
3. **Comparability policy**: all methods in the main table must share split, node order, and evaluation discipline
4. **Documentation policy**: use `docs/` as the only authoritative document root
5. **Session routing policy**: current sessions start from `LLMbot/baseline/` and the 2026-04-23 truth card
6. **Historical code-line policy**: keep `LLMbot/code` records as historical experiment evidence, not current default routing
7. **Baseline caution policy**: `LLMbot/baseline/core` cannot be treated as unified-protocol output by default; formal comparisons must explicitly enforce canonical split usage and Macro-F1 evaluation

---

## Project Decisions

- **Dataset split**: use `datasets/TwiBot-20/train_idx.pt`, `valid_idx.pt`, and `test_idx.pt` as the canonical comparison split
- **Baseline inclusion**: LMbot, BotBR, HyperScan, and the active method line belong in the comparison story
- **Reported vs reproduced rule**: keep reported baselines in the table with clear annotation when reproduction is incomplete
- **Current truth card**: use docs/wiki/project/mainline_switch_context_2026-04-23.md for current session routing
- **Historical migration note**: keep docs/wiki/project/baseline_migration_comparison_context_2026-04-17.md as the prior layout-change record
- **Collaboration preference note**: use docs/wiki/project/codex-research-collaboration-preferences-2026-04-17.md as the current quality-first research collaboration entry

---

## Experiments

### Baseline Anchors

- [exp:lmbot_gnn_lm_baselines - LMbot GNN + LM Baseline Results](experiments/lmbot_gnn_lm_baselines.md)
  - reproduced
  - LMbot RGCN GNN Macro-F1 `0.8723`
  - LMbot LM co-trained Macro-F1 `0.8771`

- [exp:lmbot_rgt_baseline - LMBot RGT Baseline Reproduction](experiments/lmbot_rgt_baseline.md)
  - reproduced
  - RGT single-pass GNN Macro-F1 `0.8619`

### Historical `LLMbot/code` Active-Method Line

- [exp:baseline_audit_correction - Baseline Comparability Audit Correction](experiments/baseline_audit_correction.md)
  - preserves the historical note that validated rw1/rw4 unified-protocol results referred to `LLMbot/code`

- [exp:failure_regime_diagnostics_v2 - Failure-Regime Diagnostics v2](experiments/failure_regime_diagnostics_v2.md)

- [exp:kill_tests_b2_b3 - Kill Tests B2+B3](experiments/kill_tests_b2_b3.md)

- [exp:confidence_fallback - Confidence Fallback Baseline](experiments/confidence_fallback.md)

### Comparison Planning

- [project:mainline_switch_context_2026-04-23](project/mainline_switch_context_2026-04-23.md)

- [exp:baseline_reproduction_plan - Baseline Reproduction Plan](experiments/baseline_reproduction_plan.md)

- [project:baseline_migration_comparison_context_2026-04-17](project/baseline_migration_comparison_context_2026-04-17.md)

---

## Claims

### Active method claims

- [claim:C1 - Graph Evidence is Uneven and Can Be Harmful](claims/C1_graph_evidence_uneven.md)
- [claim:C2 - Confidence-Based Fallback is a Strong Baseline](claims/C2_confidence_fallback_strong.md)
- [claim:C3 - Dual-Expert Abstention is More Principled Than Hard Rewrite](claims/C3_abstention_principled.md)
- [claim:C4 - Expert Disagreement is a High-Precision Risk Cue](claims/C4_disagreement_structural_signal.md)
- [claim:C5 - Propagation Repair Beats Output-Level Deferral](claims/C5_repair_beats_deferral.md)
- [claim:C6 - Utility-Guided Pruning is Necessary](claims/C6_utility_guided_necessary.md)
- [claim:C7 - Prune-Only Single Pass is Sufficient](claims/C7_prune_only_sufficient.md)
- [claim:C8 - Different Failure Regimes Prefer Different Actions](claims/C8_regime_action_preference.md)

---

## Quick Stats

- **Ideas**: 3 proposed
- **Baseline anchor experiments**: 2 reproduced LMbot baseline notes
- **Historical `LLMbot/code` notes**: rw1/rw4 audit, diagnostics, kill tests, fallback
- **Project notes**: split decision, baseline migration context, mainline switch context
- **Current session routing**: baseline-routed comparison work with historical `LLMbot/code` context preserved
- **Current research phase**: user-directed; use the A-G taxonomy when the user names the active phase

