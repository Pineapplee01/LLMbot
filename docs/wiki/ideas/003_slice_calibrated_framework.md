---
type: idea
node_id: idea:003
title: Slice-Calibrated Risk Coverage Detection Framework
stage: proposed
outcome: pending
novelty_score: 3
feasibility_score: 10
narrative_strength: 9
overall_score: 83
origin_skill: idea-creator
created_at: 2026-04-07T21:31:00Z
updated_at: 2026-04-08T00:00:00Z
tags: [diagnostics, slice-analysis, risk-coverage, calibration]
target_gaps: [G1, G5]
---

# Slice-Calibrated Risk Coverage Detection Framework

## One-Line Thesis

Analysis-driven reliability diagnostic framework that explains where graph methods fail, why they fail, and how to intervene.

## Core Mechanism

- Build slice detectors: based on degree, graph missingness, disagreement gap, calibration gap
- Generate risk-coverage curves per slice
- Design slice-specific correction strategies: reweight, abstain, fallback, hybrid

## Why Better Than Current Rewrite

| Dimension | Current Hard Rewrite | Slice Framework |
|-----------|---------------------|-----------------|
| Claim | Method innovation | Analysis + method |
| Dependency | F1 gain | Diagnostic insights |
| Risk | Modest gain hard to defend | Value even without gain |
| Contribution | Single method | General diagnostic framework |

## Expected Outcomes

- **F1 gain**: +0.3-1.0% (depends on correction strategy)
- **Main value**: Diagnostic insights and paper narrative
- **Deliverables**:
  - Complete slice diagnostic report
  - Risk-coverage curves
  - Slice-specific best strategies
  - Analysis directly usable for paper

## Implementation Timeline

2-4 days (fastest path to insights)

## Novelty Risks

⚠️ **Must differentiate from**:
- Graph Conformal Prediction (ICLR'24): Distribution-free coverage vs calibration-driven diagnostics
- Slice Discovery: Generic slice discovery vs task-specific reliability slices

✅ **Key differentiation**:
- Framework is calibration-driven, not distribution-free statistical guarantees
- Slice definitions are task-specific (bot detection failure modes)
- Goal is reliability diagnostics, not generic slice discovery

## Related Papers

- paper:huang2024_conformal_gnn
- paper:eyuboglu2022_domino

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Failure Notes

*Not yet tested*
