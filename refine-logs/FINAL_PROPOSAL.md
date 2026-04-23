# Final Proposal: Embedding-Dominant Failure Analysis with Post-Hoc Semantic Override for Social Bot Detection

**Date**: 2026-04-19
**Venue Target**: CIKM 2026 short/applied track (May 23) / ASONAM 2026 (Apr 26 stretch)
**Status**: Method refined, ready for implementation
**Supersedes**: FRMI proposal (2026-04-15) — FRMI on RGT system is abandoned

---

## Problem Anchor

Social bot detection on TwiBot-20 (11,826 nodes). Prior work pursues stronger graph architectures or fusion methods, but the relative contribution of embedding quality vs graph structure modification is not systematically characterized. Errors concentrate in a small set of recoverable failure regimes (prop_corruption, 1.84x enrichment), not uniformly.

**Scope boundary**: Post-hoc correction on a frozen co-trained checkpoint. No retraining. No graph modification as a primary contribution.

---

## Innovation Thesis

**Embedding quality dominates graph structure modification in social bot detection. On TwiBot-20, iterative LM-GNN co-training provides +0.72pp F1 gain while graph rewriting provides ±0.3pp (within noise). A lightweight post-hoc semantic override policy, triggered by entropy+disagree on the top-15% risky nodes, improves targeted failure slices by +8.3pp accuracy without retraining or graph modification.**

**Dominant contribution**: Systematic 2×2 embedding×graph analysis + post-hoc semantic override policy
**Supporting contribution**: Failure regime characterization (prop_corruption 1.84x enrichment, class-conditional disagreement frontier)
**Explicitly rejected**: Graph modification as primary contribution; composite d_i trigger; FRMI 3-action policy; end-to-end retraining

---

## Method

### System Overview

```
Co-trained checkpoint (frozen):
  LM: RoBERTa → iterative co-training → finetuned embeddings
  GNN: RGCN → trained on finetuned embeddings → predictions + entropy

Post-hoc override (no retraining):
  Trigger: entropy+disagree score → top-15% risky nodes → set C
  Action: p_final(i) = α * p_sem(i) + (1-α) * p_gnn(i)  for i ∈ C
          p_final(i) = p_gnn(i)                           for i ∉ C
  Alpha: selected on valid_cal split (fixed per experiment, not per-seed)
```

### Key Design Decisions

1. **Trigger**: entropy+disagree (AUROC=0.820), not composite d_i (AUROC=0.692, killed)
2. **Action**: Semantic override with interpolation alpha, not edge attenuation (graph rewrite within noise)
3. **Alpha selection**: Fixed on valid_cal, same alpha for all seeds — fixes the instability problem
4. **Coverage**: Top-15% by entropy+disagree score (matches Phase 4 protocol)
5. **No retraining**: Post-hoc only — clean separation from base system

### Why Not Graph Repair?

Phase 3 shows graph rewrite gains (+0.003 F1) are within noise (±0.011 std). Phase 5 shows graph effect is ±0.003 while embedding effect is +0.0072. Repair helps on prop_corruption (+0.027 acc) and graph_missing (+0.047 acc) in Phase 4, but these are small slices. The dominant action is semantic override.

---

## Complexity Intentionally Rejected

| Rejected | Reason |
|----------|--------|
| Composite d_i trigger | AUROC 0.692 < entropy 0.719 (E1 killed) |
| Graph edge attenuation | Phase 3/5: gains within noise; embedding dominates |
| 3-action policy (FRMI) | Pilot F1=0.7472 < Semantic 0.7477 on wrong base system |
| End-to-end retraining | Contaminates post-hoc claim; not needed |
| Per-seed alpha tuning | Creates instability; fix alpha on valid_cal |
| TwiBot-22 transfer | Nice-to-have; not required for CIKM short |
| both_wrong intervention | 12.1% irreducible; explicitly out of scope |

---

## Claim Hierarchy

### Tier 1 (Main — must prove)
**C1**: Embedding quality (co-training iterations) dominates graph structure modification on TwiBot-20. M3−M1 = +0.0072 F1 (embedding effect) vs M2−M1 = +0.0021 F1 (graph effect, within noise).

**C2**: Post-hoc semantic override on top-15% risky nodes improves targeted failure slice accuracy by ≥5pp without retraining, while maintaining competitive global F1.

### Tier 2 (Mechanism — should prove)
**C3**: entropy+disagree trigger (AUROC=0.820) identifies failure-prone nodes better than entropy alone (0.819) or disagreement alone (0.576).

**C4**: Failure regimes are concentrated: prop_corruption nodes have 1.84x error enrichment; sparse_evidence has no enrichment.

**C5**: Class-conditional disagreement frontier: when LM and GNN disagree, each expert captures one class with near-perfect accuracy (confirmed pre/post-calibration, test+valid).

### Tier 3 (Supporting — already proven)
**C6**: C&S post-processing (+1.18% acc) is a strong frozen baseline but does not address regime-specific failure patterns.
**C7**: Graph rewrite (aux_50) provides marginal and inconsistent gains (±0.003 F1, within noise).

---

## Kill Conditions

1. **Alpha instability persists after fixing to valid_cal**: If alpha selected on valid_cal still varies >0.15 across seeds, the override policy is not stable. Fall back to pure diagnostic paper.

2. **Override provides < 3pp target improvement**: If semantic override at fixed alpha improves target acc by < 3pp (vs Phase 4's +8.3pp), the intervention claim is too weak. Fall back to diagnostic + embedding analysis only.

3. **Global F1 drops > 0.5pp**: If override hurts global F1 by more than 0.5pp, the method is not safe to deploy. Narrow claim to "improves targeted slices at cost of global performance."

---

## Remaining Risks

| Risk | Mitigation |
|------|-----------|
| Alpha instability | Fix alpha on valid_cal; report std across seeds |
| Override hurts global F1 | Report both global and targeted metrics; frame as tradeoff |
| Embedding effect too small for novelty | Frame as systematic characterization, not new method |
| Single dataset | Explicitly scope to TwiBot-20; add TwiBot-22 if time permits |
| CIKM short track competitive | Target ASONAM 2026 as fallback (Apr 26) |
