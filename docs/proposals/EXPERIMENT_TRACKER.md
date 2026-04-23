# Experiment Tracker: LLM-Guided Graph Modification for Social Bot Detection

> Updated 2026-04-09 — Research mainline re-anchored
> Primary task: Social bot detection on X / TwiBot-20
> Primary metrics: Accuracy, Macro-F1
> Innovation: LLM-guided graph modification (not calibration-first)

## Status Legend

- `todo` | `running` | `done` | `blocked` | `drop`

## Research Phase Vocabulary

Use `docs/research/project_phase_taxonomy.md` for current high-level phase naming. These A-G labels are flexible research-planning labels, not code-facing CLI stage names.

| Phase | Canonical Name |
| --- | --- |
| A | 阶段 A：semantic encoder → GNN |
| B | 阶段 B：GNN 输出 → post-hoc estimator |
| C | 阶段 C：risk/regime → local semantic enhancement |
| D | 阶段 D：risk/regime → local propagation repair |
| E | 阶段 E：action selection |
| F | 阶段 F：same-head refinement |
| G | 阶段 G：positioning / stress test |

## Study Goal (Re-Anchored)

**Primary Goal**: Achieve competitive Acc/F1 vs LMbot, BotBR, HyperScan on unified TwiBot-20 protocol
- Hard target: Exceed LMbot & BotBR
- HyperScan (87.2% F1 reported): Include as main reference but not required to exceed in short term

**Mechanism Goal**: Show that LLM-derived semantic signals (embedding, confidence, complementarity) can guide when and how to modify graph structure, improving detection on structurally unreliable nodes

**Target venue**: WWW 2026 / AAAI 2026  
**Current status**: Mechanism validated (kill tests passed), awaiting baseline comparability resolution

---

## Claim Hierarchy

### Tier 1 (Main Contribution) — STATUS: PENDING
- Competitive Acc/F1 on unified protocol vs LMbot/BotBR/HyperScan
- **Blocker**: Need baseline comparability audit + unified protocol reproduction

### Tier 2 (Mechanism Evidence) — STATUS: VALIDATED
- LLM conditional complementarity triggers graph modification
- Modification > no modification on triggered nodes

### Tier 3 (Supporting Evidence) — STATUS: VALIDATED
- ECE/NLL improvements, design choices, mechanism interpretability

---

## Block Status

| Block | Purpose | Status | Priority | Result |
|-------|---------|--------|----------|--------|
| **Claim Gate Group: Comparability-First** | | | | |
| BC | Baseline comparability audit | `todo` | 🔴 CRITICAL | BLOCKS Tier 1 claims |
| BR1 | LMbot reproduction (unified protocol) | `todo` | 🔴 CRITICAL | BLOCKS main table |
| BR2 | BotBR reproduction (unified protocol) | `todo` | 🔴 CRITICAL | BLOCKS main table |
| BR3 | HyperScan handling | `todo` | 🔴 CRITICAL | Reproduce OR reported result + annotation |
| BT | Unified comparison table | `todo` | 🔴 CRITICAL | Acc/F1 main, auxiliary separate |
| **Claim Gate Group: Method Positioning** | | | | |
| B0 | Preflight / reproducibility | `done` | — | ✅ All artifacts verified |
| B1 | Trigger characterization | `done` | — | ✅ Coverage 14.2%, semantic > graph on slice |
| B2 | Modification vs deferral | `done` | — | ✅ **PASS** (+0.16 disagree F1, 15x threshold) |
| B3 | Utility ablation | `done` | — | ✅ **PASS** (+0.006 F1, +0.017-0.043 ECE) |
| B4 | Prune-only sufficiency | `done` | — | ✅ rw2=rw3=0.7620 F1 |
| B6 | Multi-seed confirmation | `todo` | 🟡 MEDIUM | Mechanism validation |
| **Claim Gate Group: Auxiliary Evidence** | | | | |
| B5 | Threshold robustness | `todo` | 🟢 LOW | Optional |
| B7 | Mechanism figures | `todo` | 🟢 LOW | Publication quality |

**Critical Path**: BC → BR1/BR2/BR3 → BT → (B6) → Paper writing

---

## A-G Phase Alignment

| Work Item | A-G Phase Alignment | Notes |
| --- | --- | --- |
| B0/B1 trigger and preflight | 阶段 B：GNN 输出 → post-hoc estimator | Uses existing outputs to characterize risk and triggers. |
| B2 repair vs deferral | 阶段 D：risk/regime → local propagation repair; 阶段 G：positioning / stress test | Compares repair to post-hoc deferral under the same trigger. |
| B3 utility ablation | 阶段 B：GNN 输出 → post-hoc estimator; 阶段 D：risk/regime → local propagation repair | Tests whether utility/risk signals drive repair gains. |
| B4 prune-only sufficiency | 阶段 D：risk/regime → local propagation repair | Tests repair mechanism scope. |
| B5 threshold robustness | 阶段 G：positioning / stress test | Stress-tests sensitivity. |
| B6 multi-seed confirmation | 阶段 G：positioning / stress test | Provides stability evidence. |
| B7 mechanism figures | 阶段 G：positioning / stress test | Paper-facing analysis and positioning. |
| BC/BR1/BR2/BR3/BT | 阶段 G：positioning / stress test | Unified comparison and baseline handling. |

---

## Detailed Run Table

### Mechanism Validation Runs (Tier 2/3 Evidence)

**Note**: These results are from mechanism validation experiments. Main comparison table requires unified protocol baseline reproduction first.

| Run ID | Block | Status | Seed | F1 | ECE | Slice F1 (disagree) | Notes |
|--------|-------|--------|------|----|-----|----------|-------|
| rw1_calibration_only | B0/B1 | `done` | 42 | 0.7391 | 0.0738 | 0.4079 | anchor + trigger source; coverage 29.1% |
| same-trigger-defer (post-hoc) | B2 | `done` | 42 | 0.7444 | 0.0754 | 0.4324 | confidence fallback baseline; tau=0.10 |
| rw4_disagreement_local | B2 | `done` | 42 | **0.7603** | **0.0190** | **0.5881** | ✅ **+0.16 disagree F1 vs deferral** |
| rw2_calib_prune | B2/B3 | `done` | 42 | **0.7620** | **0.0235** | 0.5881 | upper-bound competitor; best overall F1 |
| rw2_calib_prune_no_qgraph_trigger | B3 | `done` | 42 | 0.7559 | 0.0402 | — | ablation: no q_graph trigger (-0.006 F1) |
| rw2_calib_prune_no_soft_gate | B3 | `done` | 42 | 0.7559 | 0.0661 | — | ablation: no soft gating (-0.006 F1, +0.043 ECE) |
| rw2_calib_prune_no_expl | B3 | `done` | 42 | 0.7557 | — | — | ablation: no explanation enrichment |
| rw3_calib_prune_add | B4 | `done` | 42 | 0.7620 | 0.0235 | — | add-budget control (matches rw2) |

### Baseline Reference Runs (Need Unified Protocol Verification)

| Method | Source | Acc | F1 | Notes |
|--------|--------|-----|-----|-------|
| LMbot GNN | exp:lmbot_gnn_lm_baselines | 0.8533 | 0.8732 | Need protocol verification |
| LMbot LM | exp:lmbot_gnn_lm_baselines | 0.8554 | 0.8756 | Need protocol verification |
| BotBR | Paper reported | — | ~0.868 | Need unified protocol reproduction |
| HyperScan | Paper reported (TwiBot-20) | — | 0.872 | Reproduce OR include reported result + annotation |

### Planned Runs

| Run ID | Block | Status | Seed | Purpose |
|--------|-------|--------|------|---------|
| LMbot (unified protocol) | BR1 | `todo` | 1,2,3,4,5 | Baseline reproduction |
| BotBR (unified protocol) | BR2 | `todo` | — | Baseline reproduction |
| HyperScan (unified protocol) | BR3 | `todo` | — | Baseline reproduction OR reported result |
| rw4_disagreement_local | B6 | `todo` | 123 | Multi-seed validation |
| rw4_disagreement_local | B6 | `todo` | 456 | Multi-seed validation |
| rw1_calibration_only | B6 | `todo` | 123 | Multi-seed anchor |
| rw1_calibration_only | B6 | `todo` | 456 | Multi-seed anchor |

---

## Kill Test Results (Mechanism Validation - Tier 2/3)

### B2: Modification vs Deferral ✅ PASS

**Result**: rw4_disagreement_local vs same-trigger-defer (seed 42, 2026-04-08)
- Modification disagreement-slice F1: **0.5881**
- Deferral disagreement-slice F1: **0.4324**
- **Disagree-slice gain: +0.1557** (required: ≥0.01) → **15x the threshold!**
- Overall macro-F1 gain: +0.0159 (0.7444 → 0.7603)
- ECE improvement: -0.0564 (0.0754 → 0.0190)

**Verdict**: ✅ **STRONG PASS** — LLM-guided modification dramatically outperforms output-level deferral

### B3: Utility Ablation ✅ PASS

**Result**: Utility component ablations (seed 42, 2026-04-08)
- rw2_calib_prune (full): **0.7620 F1, 0.0235 ECE**
- no_qgraph_trigger: **0.7559 F1, 0.0402 ECE** (Δ: -0.0060 F1, +0.0166 ECE)
- no_soft_gate: **0.7559 F1, 0.0661 ECE** (Δ: -0.0060 F1, +0.0426 ECE)

**F1 gains**: +0.0060 vs both ablations (required: ≥0.005) ✅  
**ECE improvements**: +0.0166 and +0.0426 (required: ≥0.015) ✅

**Verdict**: ✅ **PASS** — Utility components (q_graph trigger + soft gating) are necessary

### B4: Prune-Only Sufficiency ✅ VALIDATED

**Result**: Prune-only vs prune+add (seed 42, 2026-04-08)
- rw2_calib_prune (prune-only): **0.7620 F1**
- rw3_calib_prune_add (prune+add): **0.7620 F1**

**Verdict**: ✅ **VALIDATED** — Edge addition provides no benefit; prune-only is sufficient

---

## Next Critical Actions (Priority Order)

**Priority Group 1: Comparability-First (BLOCKS Tier 1 CLAIMS)**
1. **BC**: Baseline comparability audit
   - Check LMbot/BotBR/HyperScan protocol compatibility (split, node order, metrics)
   - Reference: `LLMbot/doc/external_baseline_contracts_2026-03-27.md`
2. **BR1**: LMbot reproduction on unified protocol
3. **BR2**: BotBR reproduction on unified protocol
4. **BR3**: HyperScan handling
   - If reproducible on unified protocol → run
   - Else → include paper reported result (87.2% F1) with annotation "reported (HyperScan paper, TwiBot-20)"
5. **BT**: Unified comparison table (Acc/F1 main, auxiliary separate)

**Priority Group 2: Method Positioning**
6. Method re-evaluation on unified protocol (if needed)
7. **B6**: Multi-seed validation (seeds 123, 456)

**Priority Group 3: Auxiliary Evidence (Optional)**
8. **B5**: Threshold robustness sweep
9. **B7**: Publication figures

## Auto Review Loop Gate

Before launching `/auto-review-loop`:
- B2 and B3 kill tests both passed
- B6 multi-seed mean F1 ≥ rw1 + 0.01
- EXPERIMENT_TRACKER.md updated for every finished run
- FINAL_PROPOSAL.md claims match actual results
