# External Review — FRMI (GPT-5.4 xhigh via Codex MCP)

**Date**: 2026-04-15
**Reviewer**: GPT-5.4 (xhigh reasoning) via Codex MCP
**Thread**: 019d8fb9-7815-7cc2-a82b-638452564385
**Rounds**: 3

---

## Round 1: Initial Review

**Score**: 4/10 (Weak Reject)
**Confidence**: 4/5

### Summary
The paper studies social bot detection on TwiBot-20 and proposes a failure-aware intervention framework (FRMI) with composite hard-node score, 3-action policy, and trust-weighted attenuation. The motivation is sensible but the paper overstates what is established.

### Strengths
- Problem is real and well motivated
- Sharper mechanism story than generic "LLM + GNN fusion"
- Local repair vs deferral result is the strongest part
- Calibration improvement is large and practically interesting
- Authors aware of important caveats (narrow semantic ceiling)

### Weaknesses
- 3-action policy collapses under scrutiny (Enhance covers only 3%)
- Soft attenuation reads as soft pruning without stronger evidence
- Trigger score is heuristic, only modestly differentiated from simpler alternatives
- Single dataset, lacks multi-seed validation
- 12.1% both_wrong untouched
- No credible end-to-end competitive result against strong baselines

### What Would Move Toward Accept
1. Clean unified main table with reproduced baselines (3+ seeds)
2. Prove 3-action > 2-action
3. Prove soft attenuation > hard prune under matched conditions
4. Multi-seed results
5. Narrow the claim: "selective local correction on recoverable regimes"

---

## Round 2: Pushback & Guidance

### Key Decisions
- 3-action framing defensible as framework/taxonomy claim if ablation is clean
- Soft attenuation needs: beats hard prune + binarizing hurts + w_ji not bimodal + correlates with edge utility
- One more dataset helps ~+1 score point
- d_i: use validation-trained logistic regression, show composite > each component

### Minimum Viable Paper (4/10 → 6/10)
1. 3-seed unified main table on TwiBot-20
2. Controller necessity ablation (3-action vs 2-action)
3. TwiBot-22 transfer (or matched-budget soft vs hard if TwiBot-22 unavailable)

### Narrower Thesis
"Graph failures in social bot detection are concentrated in a small set of recoverable regimes. A regime-conditioned local correction policy that mostly abstains, invokes semantic fallback for graph-missing cases, and repairs propagation-corrupted neighborhoods improves targeted failure slices and calibration while leaving the irreducible both-wrong region outside its claim scope."

---

## Round 3: Submission Plan

### Recommended Title
`Selective Local Correction for Recoverable Failure Regimes in Social Bot Detection`

### Tables Required
| Table | Content | Licenses |
|-------|---------|----------|
| Table 1 | Trigger quality (AUROC/AUPRC/top-k for d_i vs entropy vs disagreement) | Composite trigger claim |
| Table 2 | Main results (baselines + FRMI-2A + FRMI-3A, 3 seeds) | Competitive performance claim |
| Table 3 | Controller ablation (3A vs 2A vs w/o Enhance vs w/o Repair, slice metrics) | 3-action necessity claim |
| Table 4 | Repair ablation (hard prune vs soft attenuation vs binarized, matched budget) | Beyond-pruning claim |
| Table 5 | Transfer (TwiBot-22 if available) | Generalizability claim |

### Figures Required
| Figure | Content |
|--------|---------|
| Fig 1 | Method overview (d_i → policy → 3 actions, "both_wrong outside scope" box) |
| Fig 2 | Trigger quality curves (precision-at-budget) |
| Fig 3 | w_ji distribution histogram + correlation with edge utility |
| Fig 4 | Action allocation by regime (stacked bars) |
| Fig 5 | Slice outcomes (graph_missing, disagreement, prop_corruption, both_wrong) |
| Fig 6 | Qualitative case study (1 success + 1 failure neighborhood) |

### Results-to-Claims Matrix
| Experiment | If positive | If negative |
|-----------|------------|-------------|
| E1: Main results | Selective correction is competitive | Slice-level utility only |
| E2: Controller ablation | Different regimes need different actions | Repair is the only useful action |
| E3: Trigger comparison | Composite d_i > uncertainty-only | Entropy dominates triggering |
| E4: Soft vs hard repair | Continuous attenuation > soft pruning | Hard pruning is sufficient |
| E5: TwiBot-22 transfer | Regime pattern transfers | TwiBot-20 case study only |

### Contingency Matrix
| Outcome | Reframe to | Venue impact |
|---------|-----------|-------------|
| Soft ≈ Hard | "Selective local repair" (drop attenuation novelty) | Small downgrade |
| 3-action ≈ 2-action | "Selective repair with optional semantic rescue" | Bigger downgrade, likely short/workshop |
| TwiBot-22 fails | "TwiBot-20 case study" | Main-track odds drop |
| All 3 fail | Diagnostic paper on recoverable vs irreducible errors | Workshop/short only |

### Venue Recommendations (as of 2026-04-15)
1. **ASONAM 2026** — abstract Apr 19, paper Apr 26, conf Aug 24-27 (BEST NEAR-TERM FIT)
2. **CIKM 2026** — abstract May 16, paper May 23, conf Nov 7-11 (short/applied track)
3. **ICML 2026 workshops** — submission ~Apr 24 (diagnostic/failure-analysis version)
4. **NeurIPS 2026 Evaluations & Datasets** — abstract May 4, paper May 6
5. **WSDM 2027** — CFP not yet posted (next top-fit main-track target)

### Abstract (Draft)
Social bot detectors that combine text and graph signals do not fail uniformly: on TwiBot-style social graphs, errors concentrate in a small set of recoverable regimes such as graph-missing users, semantic-graph disagreement cases, and locally corrupted neighborhoods. We propose a selective local correction framework that leaves most nodes untouched and invokes either semantic rescue or local graph repair only when regime-specific evidence suggests that graph information is unreliable but still recoverable. On a unified evaluation protocol, the framework primarily improves targeted failure slices and calibration relative to no-intervention and deferral baselines, with most of the benefit coming from propagation repair rather than the narrow semantic branch. We explicitly do not claim to solve the irreducible both-wrong region, and we limit our contribution to selective correction on recoverable regimes rather than universal performance gains for all nodes.

---

## Priority Experiments (Top 3)
1. **3-seed unified main table on TwiBot-20** — mandatory credibility
2. **Controller necessity ablation** (FRMI-3A vs FRMI-2A) — addresses biggest novelty risk
3. **TwiBot-22 transfer** (or matched-budget soft vs hard if unavailable) — external validity
