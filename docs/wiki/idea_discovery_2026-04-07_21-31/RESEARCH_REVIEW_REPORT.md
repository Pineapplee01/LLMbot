# Research Review Report: Reliability-Guided Local Graph Pruning

**Method**: Reliability-Guided Local Graph Pruning for Social Bot Detection  
**Date**: 2026-04-08  
**Reviewer Model**: GPT-5.4 (xhigh reasoning), 3 rounds  
**Thread ID**: 019d6b71-19f8-7ef2-95e7-921f806b0b5e  
**Pipeline Stage**: Phase 4 — External Critical Review

---

## Mock Review Score

- **Score**: 3/10 → projected 6/10 (with mandatory experiments) → 8/10 (with full package)
- **Confidence**: 0.84
- **Current verdict**: Not top-tier ready. Good intuition, overclaimed evidence.

---

## Round 1: Critical Weaknesses

### Logical Gaps / Unjustified Claims

1. **Trigger mismatch**: Claimed trigger is "trusted semantic + disagreement" but actual trigger is `q_graph < τ OR calib_error > τ_cal` — does NOT require semantic trust or conflicting predictions
2. **calib_error undefined at test time**: Calibration error is a population statistic; per-node proxy needs explicit validation
3. **Propagation-level claim unverified**: No same-trigger defer baseline — gains could come from "trust semantic more," not structural intervention
4. **Training vs inference ambiguity**: If pruning is inference-only, cannot claim it prevents contamination during representation learning
5. **Circular edge utility**: Utility estimated from `q_sem, u_sem, q_graph` — may just be "when to ignore graph," not which edges are bad
6. **Below m5 globally**: `0.762 < 0.766` — strong reviewer asks "why keep graph expert at all?"
7. **Slice definition risk**: Without pre-registered slice definition and coverage reporting, looks cherry-picked

---

## Round 2: Pushback Resolution

### Resolved Points
- **m5 tie + slice win IS publishable** — as "targeted robustness" claim, not "new best detector"
  - Requires: slice defined without test labels, coverage reported, same-trigger defer loses, global gap statistically insignificant
- **Prune > same-trigger defer validates propagation-level claim** (inference-time version)
  - Stronger contamination story needs: neighbor spillover evidence OR training-time pruning variant
- **Second dataset**: TwiBot-22 preferred over Cresci-2015; synthetic corruption on TwiBot-20 also acceptable

### Venue Projection (with 4 key results)
| Venue | Assessment |
|-------|-----------|
| WWW | Best fit — social network integrity framing |
| AAAI | Viable — broad empirical AI |
| EMNLP | Weak fit — novelty is graph, not NLP |
| ICLR/NeurIPS/ICML | Not without broader generality + stronger theory |

---

## Round 3: Claims Matrix + Paper Outline + Experiment Priority

### Claims Matrix

| Outcome | Evidence Pattern | Allowed Claims | Venue | Next Step |
|---------|-----------------|----------------|-------|-----------|
| **Optimistic** | prune > defer ≥3 F1 on triggered nodes, global within 0.5 F1 of m5, 5 seeds stable, triggered nodes show higher heterophily | "Expert disagreement is a structural warning signal." "Test-time pruning outperforms prediction-level deferral on unreliable nodes." | WWW best, AAAI viable, TMLR safe | Write as targeted robustness paper; add TwiBot-22 before submission |
| **Neutral** | prune > defer by 1-2 F1 or only some seeds, global 0.5-1.5 F1 below m5, heterophily signal weak | "Reliability-guided pruning can help some high-risk nodes on TwiBot-20." | TMLR, ECML-PKDD, workshop | Drop strong principle framing; recast as empirical study |
| **Pessimistic** | prune ≈ defer or worse, global clearly below m5, controls close gap | Very narrow claim only | Workshop or internal note | Pivot to semantic-only selective prediction or diagnostic paper |

**Decision rule**: Optimistic → submit. Neutral → pivot framing. Pessimistic → stop.

---

## Minimal Paper Outline (Optimistic Scenario)

**Title**: When Disagreement Signals Bad Neighborhoods: Reliability-Guided Test-Time Graph Pruning for Social Bot Detection

**Abstract** (3 sentences):
Social bot detection graphs are often structurally unreliable because manipulated edges create misleading local neighborhoods. We propose a test-time intervention that uses semantic-graph disagreement to identify high-risk nodes and prune incident edges with negative estimated utility before re-running message passing, rather than merely deferring the final prediction. On TwiBot-20 [and TwiBot-22], this intervention improves F1 on triggered high-risk nodes over same-trigger semantic deferral while matching the best semantic anchor globally and improving calibration.

**Section Structure**:
1. **Introduction** — disagreement is not only a routing signal; it can indicate local structural corruption
2. **Problem Setup** — test-time recovery on nodes whose neighborhoods are unreliable
3. **Method** — trigger high-risk nodes from expert disagreement; prune edges with negative estimated utility before re-propagation
4. **Why Pruning Can Beat Deferral** — output-level deferral changes only label decision; pruning changes messages received; proposition + corollary
5. **Experimental Protocol** — same-trigger head-to-head: pruning vs output-level deferral, with semantic-only and graph-only anchors
6. **Main Results** — pruning yields consistent gains on triggered nodes, preserves global performance
7. **Mechanism Analysis** — triggered nodes are structurally worse; improvement from removing harmful edges, not generic sparsification
8. **Related Work** — prior selective prediction / L2D act at prediction level; this intervenes on message passing
9. **Conclusion** — targeted test-time graph surgery is viable for bot detection on unreliable social graphs

**Figure Plan** (4 figures):
1. Method schematic: semantic + graph experts → risk trigger → local edge pruning → re-propagation
2. Main head-to-head bar plot: semantic-only, graph-only, late fusion, same-trigger defer, same-trigger prune (global + triggered slice)
3. Trigger validation / threshold sweep: triggered-slice F1, global F1, coverage vs threshold; heterophily comparison
4. Mechanism figure: histogram of utility for pruned vs retained edges, OR neighborhood visualization

**Appendix**: theorem proof, hyperparameters, full ablation tables, controls (random/degree/self-loop), per-seed results, TwiBot-22 or corruption study, failure cases, ethics note

---

## Priority Experiment Order

### Kill Tests — Run First (Day 1)
1. **Same-trigger prune vs same-trigger defer** — if this fails, propagation-level claim collapses
2. **Global comparison to m5 + triggered-slice coverage** — if slice is tiny or threshold-sensitive, story breaks
3. **Self-loop, random-prune, degree-matched controls** — if these match, method is just sparsification

### Run Next If Kill Tests Pass (Days 2-5)
4. **Triggered-node characterization** — higher heterophily / lower graph conditional accuracy
5. **3-seed replication** → **5-seed with significance tests**
6. **Threshold sweep and risk-coverage analysis**
7. **Synthetic edge-corruption study** (cheaper than second dataset, supports causal story)
8. **TwiBot-22** (needed for WWW/AAAI without looking benchmark-fragile)

### Defer to Revision
- 10 seeds instead of 5
- Train-time pruning variant
- Neighbor spillover beyond 1-hop
- Explanation module and edge-addition variants
- Large ablation grids over utility function design

### Do NOT Spend GPU On This Week
- More edge-addition ideas
- Explanation modules
- Architecture changes before kill tests pass
- Broad formalism not tied to actual experiments

---

## Practical 1-Week Plan

| Day | Task |
|-----|------|
| 1 | Same-trigger defer baseline, global m5 check, slice coverage, self-loop/random controls |
| 2 | 3-seed replication |
| 3 | Heterophily + local unreliability analysis, threshold sweep |
| 4-5 | Full 5 seeds on surviving configs |
| 6-7 | Synthetic corruption study OR TwiBot-22 (whichever finishes cleanly) |

---

## Minimum Theory Required

**One proposition + one corollary** (not tautological):

Setup: linearized 1-layer message passing, binary classification, calibrated semantic posterior, additive signed message terms per edge.

- **Proposition**: Pruning edges with negative estimated utility reduces expected conditional risk / increases expected margin for triggered nodes
- **Corollary**: Output-level deferral cannot remove contaminated neighbor influence; there exist transductive settings where deferral leaves joint risk unchanged while pruning strictly lowers it

**Avoid**: "If utility is defined as improvement after pruning, then pruning negative-utility edges helps" — this proves nothing.

---

## Strongest 1-Week Pivot (if kill tests fail)

**Reliability-Guided Test-Time Graph Surgery** (novelty ~7/10):
1. Learn node risk score `r_i` from `q_graph, q_sem, u_sem`, disagreement margin, local structural features
2. Trigger only on high-risk nodes
3. Estimate counterfactual edge influence (leave-one-edge-out or gradient proxy)
4. Prune edges whose removal moves graph posterior toward semantic posterior
5. Re-run 1-2 propagation steps

Why stronger: directly operationalizes propagation risk; cleaner head-to-head vs deferral; drops low-yield distractions.

---

*Generated by: idea-discovery pipeline → research-review phase*  
*Models: Claude Opus 4.6 + GPT-5.4 (xhigh), 3 rounds*  
*Thread ID: 019d6b71-19f8-7ef2-95e7-921f806b0b5e*
