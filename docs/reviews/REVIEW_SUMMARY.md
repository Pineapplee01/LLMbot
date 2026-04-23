# Review Summary

**Last updated**: 2026-04-15
**Method**: FRMI (Failure-Regime-Conditioned Minimal Intervention)
**Recommended title**: Selective Local Correction for Recoverable Failure Regimes in Social Bot Detection

---

## Score Trajectory

| Date | Stage | Score | Verdict |
|------|-------|-------|---------|
| Mar 25 | Academic code review | 6.4/10 | Prototype, not submission-clean |
| Apr 1 | Proof-chain validation (rw0-rw4) | — | Mechanism validated |
| Apr 8 | rw4 mechanism review (GPT-5.4) | 7/10 | Workshop viable |
| Apr 15 | FRMI full review (GPT-5.4 xhigh) | **4/10** | Weak reject — needs 3 experiments |
| Apr 15 | Projected with experiments | 6/10 | Borderline accept |

Score dropped from 7→4 because the Apr 15 review honestly assessed the full FRMI claim (3-action policy + composite d_i + beyond-pruning repair) rather than just the validated rw4 mechanism.

---

## E1/E4 Experiment Results (2026-04-15)

**E1: Trigger Quality** — C1 (composite d_i) KILLED

| Trigger | AUROC | AUPRC | Top-5% | Top-10% |
|---------|-------|-------|--------|---------|
| gnn_entropy | 0.719 | 0.464 | 57.6% | 55.1% |
| entropy+disagree | **0.733** | **0.493** | 55.9% | **61.0%** |
| d_i_learned | 0.692 | 0.453 | 45.8% | 51.7% |

Composite d_i does NOT beat entropy. Use entropy+disagree as trigger instead.

**E4: Single-Seed Pilot** (seed 42, proxy repair)

| System | F1 | gm_F1 | dis_F1 | pc_F1 |
|--------|-----|-------|--------|-------|
| Semantic (S) | 0.7477 | 0.8161 | 0.5763 | 0.6614 |
| Graph (G0) | 0.7236 | 0.6620 | 0.3943 | 0.6875 |
| FRMI-3A (pilot) | 0.7472 | 0.8161 | 0.5530 | 0.6692 |

Marginal regression (proxy repair uses LM, not actual attenuation).

---

## Positioning

**From**: "Adaptive failure-aware social bot detection" (RESEARCH_BRIEF.md)
**To**: "Selective local correction for recoverable failure regimes" (GPT-5.4 recommendation)

**Paper thesis**: Graph failures in social bot detection concentrate in a small set of recoverable regimes. A selective local correction policy that mostly abstains, invokes semantic rescue for graph-missing cases, and repairs propagation-corrupted neighborhoods improves targeted failure slices and calibration while leaving the irreducible both-wrong region outside its claim scope.

---

## Contingency Matrix

| Outcome | Reframe to | Venue impact |
|---------|-----------|-------------|
| Soft attenuation does not beat hard prune | "Selective local repair" (drop attenuation novelty) | Small downgrade |
| 3-action does not beat 2-action | "Selective repair with optional semantic rescue" | Bigger downgrade, likely short/workshop |
| TwiBot-22 transfer fails | "TwiBot-20 case study" | Main-track odds drop |
| All 3 fail | Diagnostic paper on recoverable vs irreducible errors | Workshop/short only |

---

## Venue Targets (as of 2026-04-15)

1. **ASONAM 2026** — abstract Apr 19, paper Apr 26 (nearest)
2. **CIKM 2026** — abstract May 16, paper May 23 (short/applied)
3. **ICML 2026 workshops** — ~Apr 24
4. **WSDM 2027** — CFP not yet posted (best future main-track)

---

## Reviewer Attacks to Prepare For

- "This is just learning-to-defer with two experts" → counter: operates at propagation level, not prediction level
- "NCwR already does graph abstention" → counter: NCwR is output-level; we intervene inside message passing
- "Why not just use semantic expert always?" → counter: repair recovers graph expert on triggered nodes
- "The 3-action policy is effectively 2-action" → counter: Enhance targets graph-missing (30% rescue rate), ablation shows removal hurts
- "Soft attenuation is just soft pruning" → counter: show w_ji distribution is not bimodal + beats hard prune at matched budget

---

## Minimum Theory Required

- **Proposition**: Pruning edges with negative estimated keep-utility reduces expected risk for triggered nodes under monotone message aggregation
- **Corollary**: Output-level deferral cannot remove contaminated messages; propagation-level repair strictly dominates deferral when contamination is local

---

## Review Thread IDs

| Review | Thread ID |
|--------|-----------|
| Apr 8 rw4 mechanism | 019d6b75-0066-78a3-a774-ed2cbde2bc3b |
| Apr 15 FRMI external | 019d8fb9-7815-7cc2-a82b-638452564385 |
