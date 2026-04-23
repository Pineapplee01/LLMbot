# Novelty Check Report: Selective Abstention Guided Graph Updates

**Method**: Selective Abstention Guided Graph Updates for Social Bot Detection  
**Date**: 2026-04-08  
**Reviewer Model**: GPT-5.4 (xhigh reasoning)  
**Pipeline Stage**: Phase 3 — Deep Novelty Verification

---

## Proposed Method

A dual-expert system (GNN graph expert + LLM semantic expert) where the graph expert
**selectively abstains** when calibration confidence is low (`q_graph < τ` OR `calib_error > τ_cal`),
and abstained nodes use semantic logits instead. Applied to social bot detection on
noisy/heterophilous graphs (TwiBot-20).

---

## Core Claims — Novelty Assessment

| # | Claim | Novelty | Closest Prior Work | Notes |
|---|-------|---------|-------------------|-------|
| 1 | Dual-expert architecture with selective abstention | **LOW** | Mozannar & Sontag 2020 (L2D), Verma et al. 2023 | Essentially 2-expert learning-to-defer |
| 2 | Calibration-driven abstention via `q_graph` + calib error | **MEDIUM** | Calibrated Selective Classification (ICLR), GATS (NeurIPS'22) | Calibration angle helps; "use reliability to reject" not new in spirit |
| 3 | Learned abstention via risk-coverage objective | **LOW** | SelectiveNet (ICML'19), NCwR (TMLR 2025) | NCwR already does graph node classification with reject option |
| 4 | Application to social bot detection | **LOW** | TopGateGNN (IJCNLP'25), BotLGT (2025) | "Apply X to Y" without bot-specific insight |
| 5 | No topology modification; routing instead of editing | **LOW** | NCwR, SelectiveNet | Design choice, not a contribution |

---

## Closest Prior Work

| Paper | Year | Venue | Overlap | Key Difference |
|-------|------|-------|---------|----------------|
| [NCwR — Node Classification with Reject Option](https://openreview.net/forum?id=4xXJDO8Bvu) | 2025 | TMLR | Learned abstention for GNN node classification with coverage objectives | Does not use dual-expert or calibration disagreement as trigger |
| [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html) | 2019 | ICML | Training-time learned abstention with coverage constraint | General, not graph-specific |
| [Learning to Defer (Mozannar & Sontag)](https://proceedings.mlr.press/v119/mozannar20b.html) | 2020 | ICML | 2-expert deferral framework | No graph structure; no calibration disagreement trigger |
| [Calibrated L2D](https://proceedings.mlr.press/v162/verma22c.html) | 2022 | ICML | Calibrated learning-to-defer | No graph-specific propagation intervention |
| [TopGateGNN](https://aclanthology.org/2025.ijcnlp-long.14/) | 2025 | IJCNLP | Selective message gating for bot detection | Hierarchical gating, NOT expert abstention |
| [UnGSL / Uncertainty-aware GSL](https://openreview.net/forum?id=AB22PNdfwP) | 2025 | WWW | Uncertainty for graph structure refinement | Structure refinement, not abstention/routing |
| [CF-GNN / Conformal Prediction for GNNs](https://proceedings.mlr.press/v202/h-zargarbashi23a.html) | 2023 | ICML | Prediction sets / coverage guarantees | Not learned abstention; no expert switching |

---

## Overall Novelty Assessment

- **Score**: 4/10
- **Recommendation**: **PROCEED WITH CAUTION**
- **Key differentiator**: The exact combo "graph expert defers to semantic expert using calibration disagreement" is **plausibly new**, but sits at the intersection of established areas
- **Risk**: Reviewers will cite NCwR, SelectiveNet, and L2D as prior art; "just learning-to-defer with two experts" attack is strong

---

## Minimum Viable Delta for Top Venue

Based on GPT-5.4 deep analysis, the following is needed to reach ICLR/NeurIPS/ICML:

### Sharpest Defensible Claim
> **"Expert disagreement is a structural warning signal, not just a routing event."**
> When semantic expert is trusted and graph expert disagrees, **locally prune propagation**
> around that node — not just defer the final label.

This is the cleanest gap vs. SelectiveNet/NCwR/L2D because they operate at **prediction level**;
this contribution is a **propagation-level intervention**.

### Mandatory Experiments (7)

1. **Repair vs defer-only** on same triggered nodes — does graph repair matter beyond abstaining?
2. **3-5 seed baseline matrix** vs `m5`, `m9`, `rw0`, `rw1` — is effect real, not seed noise?
3. **Frozen-slice evaluation** on original `disagreement`, `graph_missing`, `low_degree` slices
4. **Graph-expert recovery** — does rewrite improve graph expert itself, or just hide failure via routing?
5. **Minimal-mechanism ablation** — prune-only vs prune+add+explanation
6. **Self-loop and random-placebo controls** — rule out trivial structural regularization
7. **Sparsity/budget curve + corruption stress test** — is this truly selective and noise-robust?

### Required Theory
A **local propagation-risk bound**:
- Under linearized message-passing, define edge utility as expected contribution to focal node's margin
- Show that if semantic correctness > graph correctness (calibrated), pruning edges with negative
  estimated utility improves expected post-propagation margin (up to calibration error terms)
- One proposition + one corollary is sufficient

### Positioning Strategy
- **Do NOT** lead with "abstention" or "defer"
- **Lead with**: "reliability-guided local graph repair"
- Graph-specific insight: bad graph decision contaminates representation learning for node AND neighborhood — output-level defer does not fix that
- Bot-specific insight: text/profile semantics are stable exactly when interaction edges are least trustworthy → semantics should supervise **when to stop trusting propagation**
- Use L2D as a baseline family, not as framing
- Title direction: `When Experts Disagree, Repair the Graph` or `Reliability-Guided Local Graph Pruning for Noisy Bot Graphs`

---

## Alternative Pivot (Higher Novelty Ceiling)

**Reliability-Gated Propagation** — novelty ceiling ~7/10, implementable in ~1 week:

```
a_i = MLP(q_sem, u_sem, q_graph, struct_i)
h_i' = a_i * W_sem * z_i + (1 - a_i) * AGG_j(g_ij * m_ji)
```

Deferral happens **inside message passing**, not at final classifier.
More novel, easier to analyze, cleaner than heuristic hard rewiring.

---

## Reviewer Attack Predictions

- "This is just learning-to-defer with two experts."
- "Learned abstention on graphs already exists (NCwR)."
- "Calibration-based rejection already exists."
- "Why not just ensemble or always use the semantic/LLM expert?"
- "How is `calib_error` defined per test node without labels?"
- "No topology modification is not novelty — it's a design choice."
- "Bot detection is an application domain, not a method contribution."

---

## Recommended Next Steps

1. **If staying with current idea**: Pivot framing to "propagation-level intervention" + add mandatory experiments 1, 3, 6 first (highest kill risk)
2. **If pivoting**: Implement reliability-gated propagation (inside message passing) — higher novelty, cleaner story
3. **Either way**: Add local propagation-risk bound (theory) before submission
4. **Run `/research-refine`** with this novelty check as input to sharpen the proposal

---

*Generated by: idea-discovery pipeline → novelty-check phase*  
*Models: Claude Opus 4.6 + GPT-5.4 (xhigh)*
