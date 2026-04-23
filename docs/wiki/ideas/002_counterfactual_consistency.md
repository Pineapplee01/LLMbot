---
type: idea
node_id: idea:002
title: Counterfactual Neighborhood Consistency Regularization
stage: proposed
outcome: pending
novelty_score: 4
feasibility_score: 7
narrative_strength: 8
overall_score: 80
origin_skill: idea-creator
created_at: 2026-04-07T21:31:00Z
updated_at: 2026-04-08T00:00:00Z
tags: [counterfactual, robustness, consistency, adversarial]
target_gaps: [G1]
---

# Counterfactual Neighborhood Consistency Regularization

## One-Line Thesis

Train graph expert to be robust to neighborhood corruption, rather than preprocessing to fix the graph.

## Core Mechanism

- Generate counterfactual neighborhoods for disagreement nodes: mask suspicious neighbors or perturb edge weights
- Train consistency loss: `L_consist = ||pred_original - pred_counterfactual||²`
- Only allow prediction changes when `q_graph` is genuinely high, otherwise enforce consistency

## Why Better Than Current Rewrite

| Dimension | Current Hard Rewrite | Counterfactual Approach |
|-----------|---------------------|------------------------|
| Assumption | Rewritten graph is better | Model should be robust to bad neighborhoods |
| Validation | Hard to verify rewrite quality | Directly test robustness |
| Generalization | Depends on rewrite heuristics | Learn general robustness |
| Story | Graph repair | Trustworthy AI robustness |

## Expected Outcomes

- **F1 gain**: +1-2% (stronger on weak slices)
- **Disagreement F1**: +2-3%
- **Robustness**: F1 drop under perturbation reduced by 30-50%
- **Best slices**: disagreement, fragile false positives/negatives

## Implementation Timeline

4-6 days

## Novelty Risks

⚠️ **Must differentiate from**:
- GRAND (NeurIPS'20): Global consistency vs selective counterfactual
- Adversarial Training on Graphs: Generic adversarial training vs calibration-guided selective regularization

✅ **Key differentiation**:
- Regularization is selective, only applied to calibration-identified risk nodes
- Counterfactual generation based on semantic-graph disagreement, not random/adversarial perturbation
- Goal is neighborhood corruption robustness, not generic adversarial robustness

## Related Papers

- paper:feng2020_grand
- paper:zugner2019_adversarial_attacks

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Failure Notes

*Not yet tested*
