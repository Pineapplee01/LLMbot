---
type: idea
node_id: idea:001
title: Selective Abstention Guided Graph Updates
stage: proposed
outcome: pending
novelty_score: 4
feasibility_score: 9
narrative_strength: 8
overall_score: 86
origin_skill: idea-creator
created_at: 2026-04-07T21:31:00Z
updated_at: 2026-04-08T00:00:00Z
tags: [abstention, reliability-aware, dual-expert, test-time]
target_gaps: [G1, G5]
---

# Selective Abstention Guided Graph Updates

## One-Line Thesis

Graph expert selectively abstains when unreliable, semantic expert takes over decision.

## Core Mechanism

- Graph expert computes abstention mask: `abstain = (q_graph < τ_abstain) OR (calibration_error > τ_cal)`
- Abstained nodes use semantic logits: `logits_final = (1-abstain)*logits_graph + abstain*logits_sem`
- Optional: Train with risk-coverage objective to learn optimal abstention policy

## Why Better Than Current Rewrite

| Dimension | Current Hard Rewrite | Abstention Approach |
|-----------|---------------------|---------------------|
| Intervention Point | Preprocessing topology | Inference-time decision |
| Reversibility | Irreversible | Fully reversible |
| Interpretability | "Rewrite makes graph better" | "Graph unreliable, don't use" |
| Risk | Bad rewrite permanently damages | No topology risk |

## Expected Outcomes

- **F1 gain**: +0.5-1.5% (modest but stable)
- **ECE improvement**: -0.01-0.03
- **Abstention rate**: 10-25% (threshold-dependent)
- **Best slices**: graph-missing (+2-3%), disagreement (+1-2%)

## Implementation Timeline

3-5 days

## Novelty Risks

⚠️ **Must differentiate from**:
- CF-GNN (ICLR'24): Test-time conformal coverage vs training-time learned abstention
- Selective Classification on Graphs (ICML'23): Generic selective prediction vs calibration-driven graph-semantic switching

✅ **Key differentiation**:
- Abstention is dual-expert architecture specific, not generic selective prediction
- Decision based on calibration disagreement, not single-model uncertainty
- Goal is reliability-aware modality switching, not coverage guarantees

## Related Papers

- paper:mozannar2020_learning_to_defer
- paper:geifman2019_selectivenet
- paper:feng2025_ncwr

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Failure Notes

*Not yet tested*
