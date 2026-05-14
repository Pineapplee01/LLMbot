# Round 8 Review (Codex GPT-5.2 xhigh, NIGHTMARE difficulty) — READY

**Thread ID**: `019e1147-4db9-7bc2-845c-fde19c29cbdd`
**Date**: 2026-05-09
**Trigger**: verify v7 surgical fixes (C5 compound gate + u_edge Block-S4) restore READY after v6 regression flagged in Round 7.

---

## Scores

| Dimension | R6 (READY) | R7 (REVISE) | R8 | vs R6 | vs R7 |
|---|---|---|---|---|---|
| 1. Problem Fidelity | 9.6 | 9.6 | **9.6** | 0 | 0 |
| 2. Method Specificity | 9.5 | 9.4 | **9.5** | 0 | +0.1 |
| 3. Contribution Quality | 9.1 | 8.7 | **9.1** | 0 | +0.4 |
| 4. Frontier Leverage | 8.9 | 8.6 | **8.9** | 0 | +0.3 |
| 5. Feasibility | 8.7 | 8.5 | **8.7** | 0 | +0.2 |
| 6. Validation Focus | 9.4 | 9.3 | **9.4** | 0 | +0.1 |
| 7. Venue Readiness | 9.1 | 8.6 | **9.1** | 0 | +0.5 |
| **OVERALL (weighted)** | **9.2** | **9.0** | **9.2** | 0 | **+0.2** |

**Verdict**: **READY** (nightmare difficulty).
**Drift Warning**: **NONE**.
**Action items**: **NONE** (no IMPORTANT-or-higher remaining).

The v7 surgical patch precisely recovered the 0.2 lost in Round 7. Every dimension returned to its Round-6 level or better; no secondary regression.

## Strict Improvement Confirmation (vs v6)

Reviewer verdict: "**Strict improvement with no regressions to v5-locked properties.**"

v7 delta:
- C5 strengthened back to centrality gate (restoring sprawl-containment that v6 compromised).
- S4 reduces residual criticism surface without widening scope in a way that threatens the anchor (pruning/discipline, not new mechanism).
- Feasibility impact: minor extra evaluation variants; not a conceptual or scope regression.

## Direct Answers to Round 8 Audit (1-3)

- **(1) Compound C5 restores v5 sprawl-containment?** — **Yes**. `(M-ro − M-base) ≥ 0.4·(M-rr − M-base)` on both datasets ensures rewrite+gate accounts for a **substantial fraction** of total gain, not merely nonzero effect. Prevents Stage-8 from being hidden mechanism while still allowing synergy. `β = 0.4` defensible: pre-registered as compromise between reviewer minimum (0.25) and v5's 0.5; 0.5pp absolute floor prevents trivial "40% of tiny" passes.
- **(2) Block-S4 preempts heuristic stacking moved to u_edge?** — **Yes, to venue bar**. Key is not "one ablation exists" but that S4 imposes **same pruning discipline** as composite: components failing ≥ 1 seed-std improvement are REMOVED from final method, not retained as optional knobs. Converts u_edge from "heuristic buffet" into pre-registered necessity-filtered commodity.
- **(3) Residual regressions vs v6?** — **None**. Strict improvement. All v5-locked properties preserved.

## MINOR Modernization Opportunities (non-blocking, easy to fold in)

1. Report C5 with paired-bootstrap CI on the **ratio** `(M-ro − M-base) / (M-rr − M-base)` in addition to the absolute/threshold checks — makes mechanism-centrality presentation airtight.
2. For Block-S4, publish the **post-pruning final u_edge** (which components survived) as the actual method; keep full u_edge only as pre-registered candidate set.

Both are presentational; no new experiments required. Folded into v7 artifact documentation.

## Final Round-by-Round Summary

| Round | Difficulty | Overall | Verdict |
|---|---|---|---|
| 1 | standard | 6.5 | REVISE |
| 2 | standard | 8.1 | REVISE |
| 3 | standard | 8.8 | REVISE |
| 4 | standard | 9.2 | READY (spec-level) |
| 5 | **nightmare** | 8.6 | REVISE (rescope cost) |
| 6 | **nightmare** | 9.2 | READY |
| 7 | **nightmare** | 9.0 | REVISE (external critique; v6 C5 regressed v5) |
| 8 | **nightmare** | **9.2** | **READY (strict improvement; no regressions)** |

Eight rounds total. Three times at READY under nightmare difficulty (R6, R8). The proposal has survived both an external reviewer's P0-P2 attack surface and the self-audit patch cycle it triggered.

---

<details>
<summary>Raw reviewer response (full verbatim)</summary>

Scores 9.6/9.5/9.1/8.9/8.7/9.4/9.1 = 9.2. Verdict READY. Drift NONE. Simplifications NONE (v7 changes are corrective and further reduce hidden-DOF / heuristic-stacking risk). Modernizations (MINOR, non-blocking): ratio CI on C5; post-pruning final u_edge publication. Action items NONE. Captured verbatim above.

</details>
