---
type: claim
node_id: claim:C3
title: Dual-Expert Abstention is More Principled Than Hard Rewrite
status: partial
confidence: medium
evidence_type: conceptual
created_at: 2026-04-07T21:31:00Z
updated_at: 2026-04-08T00:00:00Z
tags: [abstention, rewrite, principled-design]
---

# Claim: Dual-Expert Abstention is More Principled Than Hard Rewrite

## Statement

Selective abstention at the prediction level is more principled than hard topology rewrite because it is reversible, interpretable, and avoids permanent topology damage.

## Evidence Status

⚠️ **CONCEPTUALLY SOUND, EMPIRICALLY UNVALIDATED**

## Supporting Evidence

### Conceptual Advantages

1. **Reversibility**:
   - Abstention: No topology modification, fully reversible
   - Hard rewrite: Permanent edge changes, cannot undo

2. **Interpretability**:
   - Abstention: "Graph unreliable, don't use it"
   - Hard rewrite: "Rewrite makes graph better" (hard to verify)

3. **Risk profile**:
   - Abstention: No topology risk, worst case = text-only performance
   - Hard rewrite: Bad rewrite can permanently damage graph structure

### Required Validation

- [ ] Implement abstention mechanism
- [ ] Compare: abstention vs hard rewrite on same trigger nodes
- [ ] Measure: F1 gain, ECE, abstention rate, locality preservation
- [ ] Kill test: Must beat same-trigger output-level deferral
- [ ] Theory: Need local propagation-risk bound (one proposition + corollary)

## Tested By

*Not yet tested — proposed in idea:001*

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Implications

- If validated, suggests test-time decision > preprocessing topology surgery
- Supports reliability-aware routing narrative
- Challenges graph structure learning approaches that modify topology
