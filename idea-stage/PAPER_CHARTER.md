# Paper Charter: FRMI — Failure-Regime-Conditioned Minimal Intervention

**Created**: 2026-04-22
**Purpose**: Prevent claim drift. This file is the single source of truth for what this paper is and is not allowed to claim.

---

## Allowed Claims

1. Residual errors in GNN-based social bot detection concentrate in interpretable failure regimes (sparse-evidence, propagation-corruption)
2. Matched local post-hoc correction can beat unconditional repair under fixed intervention budget
3. Gain-based action selection (B1) is the main method contribution — deciding WHICH action to apply WHERE matters more than any single operator
4. The two-layer probe (calibrated risk + regime evidence) identifies failure-prone nodes better than uncertainty-only baselines

## Forbidden Claims

1. Novel semantic adapter / contrastive residual adapter as headline contribution
2. Graph rewrite / graph structure learning as contribution
3. Causal intervention / treatment effect estimation (we do NOT do causal identification)
4. New SOTA detector (we are a post-hoc correction layer, not a new detector)
5. "First" or "confirmed novel" without hedge ("to our knowledge...")
6. Any claim about A2 being a mechanism innovation (it is application-specific recombination)

## Required Experimental Proof

### Gate 1: Semantic Arm Survival (S0)
- Best semantic operator > no-op on TwiBot-20 AND TwiBot-22, 3 seeds, same sparse-screened set
- If A2 is to be named in the paper: A2 must stably beat A4 (direct logit blend)
- Comparison set: A4, A2, EGNN-style focal editor, all-node semantic fusion

### Gate 2: Decision Layer (S2)
- B1 > heuristic rule AND binary router
- Same operator, same screened set, same intervention budget

### Gate 3: System (S3)
- Three-action > repair-only, binary policy, unconditional operator
- Matched action > swapped action (paired bootstrap, p < 0.05)
- Easy-node preservation: bottom-50% safest nodes show no degradation
- Budget-gain curve: intervention % vs gain

### Cross-cutting
- 2 datasets (TwiBot-20 + TwiBot-22)
- 3 seeds per condition
- Matched budget across all comparisons

**Any gate fails → immediate claim downgrade. No full FRMI methods paper.**

## Required Baselines

| Baseline | What it covers | Priority |
|----------|---------------|----------|
| EGNN-style frozen GNN+MLP editor | Editable GNN line | MUST |
| Direct logit blending (A4) | Simplest semantic correction | MUST |
| All-node semantic fusion | LGB-style global fusion | MUST |
| GLANCE-style binary router | Selective LLM routing | MUST |
| Node-MoE / heterophily-adaptive | Per-node adaptive treatment | SHOULD |
| BotBR / RABot | Reliability-enhanced graph learning | SHOULD |

## Terminology

| Internal (code) | Paper name |
|-----------------|-----------|
| ContrastiveResidualAdapter | sparse semantic patch operator |
| EgoEdgeReweight | local propagation repair operator |
| PolicyDispatcher + B1 | gain-based action selection layer |
| FocalSemanticResidual (Ridge) | DEPRECATED — do not use in paper |

## Fallback Plan

If Gate 1 fails (semantic arm dead):
→ Paper becomes B1 + repair/no-op (two-action) OR probe/diagnostic paper (C1)

If Gate 2 fails (B1 ≈ heuristic):
→ Paper becomes diagnostic/probe contribution with simple heuristic policy

If Gate 3 fails (system ≤ repair-only):
→ Paper becomes pure diagnostic: "When Does Graph Structure Hurt Social Bot Detection?"
