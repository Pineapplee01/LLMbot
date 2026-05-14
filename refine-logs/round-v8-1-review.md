# Round v8-1 Review (Codex GPT-5.2 xhigh) — RETHINK

**Thread ID**: `019e1507-3c62-7651-98d7-4a9921fb8c62` (fresh v8 thread)
**Date**: 2026-05-09
**Cycle**: v8 (user rescope: LLM main stage restored; 7-stage pipeline hard-locked)

---

## Scores

| Dimension | Score |
|---|---|
| Problem Fidelity | 8.0 |
| Method Specificity | 6.5 |
| Contribution Quality | 6.0 |
| Frontier Leverage | 7.0 |
| Feasibility | 7.0 |
| Validation Focus | 6.5 |
| Venue Readiness | 6.5 |
| **OVERALL (weighted)** | **6.675 / 10** |

**Verdict**: **RETHINK**.
**Drift Warning**: **EXPLAIN** — "drifting from one dominant contribution into multi-heuristic pipeline tuning. Stage-1, Stage-3 gate, Stage-4 rollback criteria, and Stage-6 MLP each introduce independent degrees of freedom."

## Core Issue (honest diagnosis)

The reviewer's top critique is the same criticism I resolved in v7 rounds 1-3 (sprawl → single primitive), now re-introduced because the user restored Stage-6 LLM to the main pipeline and Stage-3 retrieval became iterative. The reviewer counts **3 parallel novelties**: (1) retrieval loop + gate + stop, (2) rewrite + accept/rollback constraints, (3) LLM embedding + new MLP refiner.

The skill rule "one dominant contribution" and the user rule "seven stages mandatory" need reconciliation. The path is **not** to delete stages — it is to designate exactly ONE stage as the intellectual center, and freeze the others to the smallest pre-committed implementations that the anchor requires.

## Focus Critiques (A–E), Key Findings

- **(A) Iterative retrieval — GraphRAG adjacent**. Novelty hinges on "confirmed nodes expand frontier under strict budget with task-specific stop rule". Missing: formal statement of failure mode unreachable by single-pass + "best-possible non-iterative" budget-matched baseline (larger k_struct once / 2-hop once / mixed PPR seeds).
- **(B) Hard gate brittle**. Cosine threshold + JSD-difference constraint may encode dataset artifacts (embedding anisotropy, label shift, calibration drift). Must replace with calibrated scorer (still frozen encoders) or temperature-scaled similarity; document precision/recall tradeoffs.
- **(C) Stage-6 LLM at risk of winning by extra features**. Stage-4 already expressive; Stage-6 adds modality + trained MLP. Need non-LLM text encoder with **same serialization** baseline (not just "RoBERTa sentence embedding" — same RoBERTa with structured pooling over serialized evidence fields) to isolate LLM's semantic-abstraction value.
- **(D) 7 stages dilute single-contribution story**. Three coupled novelties read as parallel. Must **demote two** to "minimal plumbing" and make only one the true intellectual center.
- **(E) Refiner MLP is smuggled contribution risk**. The only newly trained module is the MLP; reviewers will suspect gains come from "better head + more features". Mitigation: start Stage-6 as linear probe; upgrade to MLP only if ablation proves need.

## Simplification Opportunities (reviewer)

1. Make Stage-1 minimal: keep exactly one disagreement metric + conformal machinery (drop either `s_lbl` or one of `s_tg`/`s_npi`).
2. Reduce Stage-4 accept criteria from 4 to 2: (i) externally calibrated error-proxy improvement, (ii) bounded L∞ change. Remove set_size/margin constraints unless proven to matter.
3. Stage-6 readout as linear probe first; upgrade to MLP only if ablation justifies.

## Modernization Opportunities (reviewer)

1. Replace cosine-threshold gating with calibrated semantic entailment / temperature-scaled similarity (controls anisotropy).
2. Add budget-matched non-iterative baseline (single-pass larger k / multi-seed PPR) to fairly isolate iteration's value.
3. Structured prompt schema + deterministic serialization with versioning; ablate fields.

## Dimension Critiques (< 7)

All 6 dimensions below 7 trace to the same root cause: three parallel novelties. Fix the sprawl and the others follow.

---

<details>
<summary>Raw reviewer response (full verbatim, truncated at harness limit)</summary>

Weighted 6.675/10. RETHINK. Stage-by-stage minimal-sufficiency audit across Stages 0-6. Focus critiques (A)–(E) answered in detail. Three simplifications + three modernizations + drift-explanation captured above. Reviewer offers to rewrite contribution statement + claims into "single-dominant contribution" framing with locked 7-stage skeleton intact and tightest de-risking ablation table. Key quotations:
- "Stage-3/4 loop + Stage-6 LLM+MLP reads like two contributions unless you sharply demote one to plumbing."
- "Stage-6 'wins' because it's extra features + trained MLP, not because 'LLM understands evidence'."
- "Make only one the true intellectual center (with the others treated as necessary adapters)."

</details>
