# Final Proposal: LLM-Guided Graph Modification for Social Bot Detection

**Date**: 2026-04-09  
**Venue Target**: WWW 2026 / AAAI 2026  
**Status**: Mechanism validated, awaiting baseline comparability resolution  
**Refine Thread**: 019d6b8d-294e-79d3-aac6-c3068deb9fa2

---

## Research Mission

**Primary Task**: Social bot detection on X platform / TwiBot-20 dataset  
**Primary Metrics**: Accuracy, Macro-F1  
**Auxiliary Metrics**: ECE/NLL/Brier/AURC (diagnostic only, not main contribution)

**Success Criterion**: Achieve competitive Acc/F1 vs LMbot, BotBR, HyperScan on unified TwiBot-20 protocol. Hard target: exceed LMbot & BotBR. HyperScan (87.2% F1 reported) included as main reference but not required to exceed in short term.

---

## Problem Anchor

Social bot detection on noisy/heterophilous social graphs (TwiBot-20). Graph propagation hurts on nodes with manipulated/unreliable neighborhoods. LLM-derived semantic signals (embedding, confidence, complementarity) can guide when and how to modify graph structure to improve detection performance.

---

## Innovation Thesis

**LLM-derived semantic signals conditionally modify graph structure to improve social bot detection performance.**

**Theoretical Motivation**: *When do LLMs help with node classification? A comprehensive analysis* shows:
1. LLM and graph have conditional complementarity (each excels in different scenarios)
2. Embedding form offers better efficiency-effectiveness tradeoff for engineering systems

**Core Mechanism**: Expert disagreement between a calibrated semantic expert (LLM-based) and a calibrated graph expert (GNN-based) serves as a structural warning signal. When triggered, LLM semantic signals (confidence, embedding similarity) guide local graph modification (edge pruning) followed by re-propagation.

---

## Method Description

> When a node shows semantic-vs-graph prediction disagreement and low graph reliability, the method uses LLM-derived signals (semantic confidence, embedding conflict, relation consistency) to prune incident edges with negative estimated keep-utility and re-runs graph propagation, improving bot detection on structurally unreliable TwiBot-20 neighborhoods.

**Key Insight**: LLM signals should guide **local graph modification**, not just output-level expert selection after contaminated messages have already propagated.

---

## Research Phase Alignment

Use `docs/research/project_phase_taxonomy.md` for A-G phase naming. These are flexible research-planning labels, not CLI `--stage` names.

| Phase | Canonical Name | Current Role In This Proposal |
| --- | --- | --- |
| A | 阶段 A：semantic encoder → GNN | Semantic-source and encoder evidence feeding or comparing with the GNN path. |
| B | 阶段 B：GNN 输出 → post-hoc estimator | Post-hoc reliability, trigger, and utility estimation from model outputs. |
| C | 阶段 C：risk/regime → local semantic enhancement | Semantic-side enhancement work; claim only when validated artifacts exist. |
| D | 阶段 D：risk/regime → local propagation repair | Main graph repair mechanism and pruning/re-propagation studies. |
| E | 阶段 E：action selection | Selector/router decisions among repair, fallback, enhancement, or no-op. |
| F | 阶段 F：same-head refinement | Planned or evidence-dependent refinement path; not claim-complete by default. |
| G | 阶段 G：positioning / stress test | Baseline comparability, stress testing, robustness, and paper positioning. |

---

## Complexity Intentionally Rejected

| Rejected | Reason |
|----------|--------|
| Oracle trigger based on correctness labels | Not deployable; correctness-based masks are analysis-only |
| Trigger zoo with multiple router scores | One deployable disagreement-aware reliability trigger is sufficient |
| Edge addition in main method | Prune-only gives cleanest causal story; add hurts empirically |
| Learned end-to-end edge scorer | Fixed utilities from existing calibrated signals are enough |
| Iterative rewiring or extra fusion modules | One edit pass + one re-propagation is the paper |
| Explanation-conditioned rewrite | Infrastructure, not a literature-complete module |

---

## Refined Mechanism

```
Algorithm 1: Reliability-Guided Test-Time Graph Surgery (prune-only)

Input:
  graph G=(V,E)
  semantic outputs {pred_sem, prob_sem_cal, q_sem, u_sem}
  raw graph outputs {pred_graph, q_graph}

1. Node trigger
   For each node i:
      trust_i = q_sem(i) * (1 - u_sem(i))
      trigger_i = 1[pred_sem(i) != pred_graph(i)]
                * 1[trust_i * (1 - q_graph(i)) > tau_trig]

2. Counterfactual edge keep-utility
   For each triggered node i and each incident edge e=(j->i) or (i->j):
      conflict_ij = 0.5 * ||prob_sem_cal(i) - prob_sem_cal(j)||_1
      relmatch_ij = train-estimated relation consistency prior for e
      keep_ij = a*trust_j + b*(1-conflict_ij) + c*q_graph(i) + d*relmatch_ij
      risk_ij  = 1 - keep_ij

3. Local surgery
   For each triggered node i:
      prune up to B_in incoming and B_out outgoing edges
      subject to keep_ij < tau_keep
      ranked by largest risk_ij

4. Re-propagation
   Build edited graph G'
   Run frozen graph expert once on G'
   Fuse repaired graph output with semantic expert (same as base system)

Main paper instantiation: disagreement_local with add_budget_in=add_budget_out=0
```

**Minimum Theory**:
- **Proposition**: Pruning an edge with negative estimated keep-utility reduces expected risk for the triggered node under monotone message aggregation
- **Corollary**: Same-trigger deferral cannot remove contaminated messages because it leaves propagation unchanged

---

## Claim Hierarchy

### Tier 1 (Main Contribution)
**Competitive Acc/F1 on unified TwiBot-20 protocol vs LMbot/BotBR/HyperScan**
- **Status**: PENDING baseline reproduction + unified protocol validation
- **Requirement**: All methods must share split/node-order/metric/non-test-selection discipline
- **Target**: Exceed LMbot & BotBR; approach HyperScan (87.2% F1 reported)

### Tier 2 (Mechanism Evidence)
**LLM conditional complementarity triggers graph modification; modification > no modification**
- C1: Graph evidence is uneven and can be harmful (Text F1=0.8115 vs Full F1=0.7945)
- C4: Expert disagreement identifies unreliable propagation (14.2% coverage, semantic > graph on triggered)
- LLM-guided modification improves over baseline on triggered nodes

### Tier 3 (Supporting Evidence)
**Design choices, mechanism interpretability, auxiliary metrics**
- C2: Confidence-based fallback is strong baseline (fallback F1=0.798 vs router F1=0.788)
- C5: Propagation modification beats output-level deferral (+0.16 disagree F1, 15x threshold)
- C6: Utility-guided pruning is necessary (ablations: -0.006 F1, +0.017-0.043 ECE)
- C7: Prune-only single pass is sufficient (rw2=rw3=0.7620 F1)
- ECE improvement: 0.0738 → 0.0190 (supporting evidence only)

---

## Key Claims (Mechanism Validation - Tier 2/3)

| Claim | Test | Status |
|-------|------|--------|
| C4: Disagreement + low q_graph identifies unreliable propagation | Compare triggered vs non-triggered nodes | ✅ VALIDATED |
| C5: Modification beats deferral under same trigger | Same triggered nodes: modify-and-repropagate vs defer-to-semantic | ✅ VALIDATED (+0.16 F1) |
| C6: Utility-based pruning drives the gain | Matched-budget random prune and self-loop controls | ✅ VALIDATED |
| C7: Prune-only single pass is sufficient | Prune-only vs prune+add at matched triggers | ✅ VALIDATED |

---

## Current Status

**Mechanism Validation**: ✅ COMPLETE
- Kill tests B2+B3 passed (2026-04-08)
- Main method rw4_disagreement_local validated on seed 42
- Disagree-slice F1: 0.5881 (vs deferral 0.4324, +0.16)
- ECE: 0.0190 (vs baseline 0.0738, -74%)

**Critical Blocker**: Baseline comparability resolution
- Need unified protocol audit for LMbot/BotBR/HyperScan
- Cannot claim Acc/F1 improvements without unified comparison table
- Current results may use different splits/metrics/protocols

**Next Actions** (Priority Order):
1. Baseline comparability audit (BLOCKS Tier 1 claims)
2. LMbot/BotBR reproduction on unified protocol
3. HyperScan handling (reproduce OR include paper reported result with annotation)
4. Unified comparison table (Acc/F1 main, auxiliary separate)
5. Multi-seed validation (B6)
6. Threshold robustness (B5, optional)

---

## Must-Run Experiments

**Priority Group 1: Comparability-First (BLOCKS ALL MAIN CLAIMS)**
1. Baseline comparability audit
2. LMbot reproduction on unified protocol
3. BotBR reproduction on unified protocol
4. HyperScan handling (reproduce OR reported result + annotation)
5. Unified comparison table

**Priority Group 2: Method Positioning**
6. Method re-evaluation on unified protocol (if needed)
7. Multi-seed validation: rw1+rw4 on seeds 123, 456

**Priority Group 3: Auxiliary Evidence (Optional)**
8. Threshold robustness sweep (B5)
9. Publication figures (B7)

---

## Remaining Risks

| Risk | Mitigation |
|------|-----------|
| Baseline comparability issues | Audit protocol compatibility first; only compare methods on unified split/metric/node-order |
| Cannot exceed LMbot/BotBR on unified protocol | Reframe as targeted improvement on specific slices; analyze where/why method helps |
| HyperScan cannot be reproduced | Include paper reported result (87.2% F1) with annotation "reported (HyperScan paper, TwiBot-20)" |
| Multi-seed instability | Run ≥3 seeds for rw1 and rw4; report mean/std; acceptable if mean gain ≥0.01 F1 |
| Method trails baselines globally | Lead with mechanism evidence (Tier 2/3); position as "when to modify graph" rather than universal SOTA |
| Narrative blurs main vs auxiliary contributions | Maintain claim hierarchy: Tier 1 (Acc/F1 competitiveness) > Tier 2 (mechanism) > Tier 3 (supporting) |
