# Experiment Plan: FRMI Full Paper — Failure-Regime-Conditioned Minimal Intervention (v2)

## Context

**Problem**: Social bot detection errors are not uniform — they concentrate in structurally distinct failure regimes. Existing work pursues global fusion (LGB), global reliability (BotBR), global graph learning (BotGSL), or LLM routing (GLANCE). Our safe space: **regime diagnosis + matched local correction**.

**Method Thesis**: A two-layer diagnostic (calibrated risk + regime evidence) identifies failure-prone nodes by type; a three-action policy (no-op / focal semantic residual / ego-subgraph edge reweight) applies the matched correction locally, touching <20% of nodes while recovering hard-subgroup accuracy.

**Pivot from**: Previous CIKM short paper plan (embedding-dominant analysis + post-hoc override). Now targeting full paper with stronger causal evidence.

**Compute**: Remote server (172.31.106.108) + local RTX 4060 for analysis.

**Key protocol constraint**: TwiBot-22 official train/val/test split is frozen. OOF is used ONLY internally to generate probe training labels, never as benchmark protocol.

---

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Blocks |
|-------|----------------|-----------------------------|--------|
| P1: Errors concentrate in pre-registered regime types; two-layer probe captures them better than uncertainty-only | Core novelty — typed failure, not just "hard nodes" | Regime enrichment on ≥2 pre-registered subgroups + probe utility@15% and HCW capture significantly > uncertainty-only | A, C |
| P2: Three-action regime-conditioned policy > global/binary/unconditional alternatives | Main result — justifies the full method | 8-condition comparison; Three-Action > Always-Semantic, Always-Repair, Binary Policy, Uncertainty-only on macro-F1 + hard-subgroup ΔF1 | D, B |
| S1: Action-regime matching is genuine (causal) | Prevents "just pick best single action" dismissal | Regime-conditional matched-node table: within same screened set and budget, matched > swapped action with significant net loss | D |
| S2: Backbone-portable (directional consistency) | Generality | On BotRGCN: P1/P2/S1 hold directionally (same sign, not requiring same absolute gap) | A |

---

## Pre-Registered Subgroup Protocol

5 subgroups, defined on validation set, frozen before test evaluation:
1. **Degree buckets**: bottom-20% degree (sparse-evidence proxy)
2. **Adjusted neighborhood informativeness**: bottom-20% (propagation-corruption proxy, per homophily metrics literature)
3. **Semantic-structural conflict**: top-20% by |p_LM - p_GNN| divergence
4. **Camouflage-heavy**: nodes where ≥50% 1-hop neighbors have opposite label
5. **High-confidence-wrong (HCW)**: nodes with MSP > 0.9 but incorrect prediction (OOF-defined)

These subgroups are computed once on validation, thresholds frozen, then applied to test without modification.

---

## Paper Storyline

**Main paper must prove**: P1 (regime structure + probe), P2 (policy comparison), S1 (action swap)
**Appendix can support**: TwiBot-20 cross-dataset, BotRGCN replication, C&S orthogonality, hyperparameter sensitivity, conformal coverage
**Experiments cut**: Composite d_i trigger (killed), RGT as primary, LLM-based correction, end-to-end retraining, C&S as main contribution

---

## Experiment Blocks

### Block A0: Protocol Freeze (Week 1, Day 1-2)

- **Purpose**: Freeze all evaluation rules before any experiment touches test data
- **Protocol decisions**:
  1. TwiBot-22 official train/val/test split — frozen, no modification
  2. OOF is internal only: used to generate probe failure labels within train split (k-fold on train, predict held-out train portion). Never enters main table.
  3. All subgroup thresholds (degree buckets, neighborhood informativeness, etc.) computed on validation, frozen, applied to test without modification
  4. All probe/policy hyperparameters selected on validation only
  5. Test is forward-only: no threshold tuning, no alpha selection, no regime rule adjustment
- **Output**: `PROTOCOL_FREEZE.md` documenting all frozen decisions
- **Compute**: 0h (documentation only)
- **Priority**: MUST-RUN (blocks everything else)

### Block A1: Backbone Freeze (Week 1-2)

- **Claim tested**: Foundation for all subsequent blocks
- **Why**: Need frozen GATv2 on TwiBot-22 + internal OOF failure labels for probe training
- **Dataset**: TwiBot-22 official split (primary), TwiBot-20 (appendix confirmatory)
- **Implementation**:
  1. Add `GATv2Bot` class to [GNNs.py](LLMbot/baseline/core/GNNs.py) — 2-layer GATv2Conv, same `forward(x, edge_index, edge_type)` interface
  2. Set up TwiBot-22 data pipeline (adapt existing TwiBot-20 loader in [dataloader.py](LLMbot/baseline/core/dataloader.py))
  3. Train GATv2 on official train split (3 seeds), evaluate on val/test
  4. Generate internal OOF failure labels: k-fold within train split only (for probe training in Block C)
  5. Train BotRGCN on TwiBot-22 for confirmatory baseline (Week 6)
- **Reuse**: `GNN_Trainer` class, `build_GNN_model()`, `load_gnn_checkpoint()`
- **Success criterion**: GATv2 trains stably; produces reasonable F1 on TwiBot-22 (no rigid "within 2pp" — different backbones have different ceilings per official benchmark)
- **Failure interpretation**: If GATv2 diverges or produces degenerate predictions → BotRGCN becomes primary; reweight operator implemented as scalar edge mask instead of attention rewrite
- **Compute**: ~2h remote
- **Priority**: MUST-RUN

### Block C: Failure Risk + Regime Evidence (Week 2-3)

- **Claim tested**: P1 (regime structure + probe quality)
- **Why**: Must prove errors are regime-TYPED, not just uncertain. Two-layer design: risk score + regime evidence scores. Regime labels remain validation-defined rules, NOT learned.
- **Two-layer probe design**:
  - Layer 1 — **Calibrated risk score**: probability that base GATv2 misclassifies this node (trained on internal OOF failure labels)
  - Layer 2 — **Regime evidence scores** (NOT learned regime labels): sparse-evidence score (degree + edge-type coverage), propagation-corruption score (neighborhood informativeness + h^(0) vs h^(L) divergence + neighbor prediction inconsistency)
  - Final regime assignment: validation-frozen rules on evidence scores (e.g., sparse-evidence if degree < threshold AND edge-type coverage < threshold)
- **Probe feature vector** (low-dimensional, no raw embeddings):
  - Temperature-scaled entropy, MSP (from base GATv2 predictions)
  - In-degree, out-degree, edge-type distribution skew
  - Local neighborhood informativeness (adjusted homophily per NeurIPS 2023 metrics paper)
  - h^(0) vs h^(L) cosine divergence (intermediate GATv2 activations)
  - Neighbor prediction inconsistency (fraction of 1-hop neighbors with different predicted label)
- **3 comparison variants**: (1) uncertainty-only (entropy threshold), (2) rule-based regime trigger, (3) L2-logistic probe on full feature vector
- **Training target**: Internal OOF failure event (binary). Regime evidence scores are computed features, not learned targets.
- **Primary metrics**: utility@15% (precision of intervention at 15% budget), HCW capture rate (what fraction of high-confidence-wrong nodes are in top-k), risk-capture curve dominance over uncertainty-only
- **Secondary metrics**: AUROC, AUPRC, ECE, Brier
- **Reuse**: [diagnose_gnn.py](LLMbot/baseline/analysis/diagnose_gnn.py), [frmi_experiment.py](LLMbot/baseline/analysis/frmi_experiment.py), [edge_scorer.py](LLMbot/baseline/analysis/edge_scorer.py)
- **New code**: Feature extraction consolidation (~100 lines), probe training (sklearn LogisticRegression), risk-capture curve plotting
- **Success criterion**: Probe utility@15% and HCW capture significantly > uncertainty-only; enrichment visible in ≥2 pre-registered subgroups
- **Failure interpretation**: If probe ≈ uncertainty-only on utility@budget → expand features (add GATv2 attention entropy); if still no gain → downgrade probe to simple trigger, do not claim estimator novelty
- **Table target**: Table 2 (probe comparison), Figure 2 (risk-capture curves), Figure 3 (regime enrichment heatmap)
- **Compute**: <30min (CPU-bound after one forward pass)
- **Priority**: MUST-RUN

### Block D: Local Operator Policy (Week 3-4) — Core Contribution

- **Claim tested**: P2 (policy comparison) + S1 (action-regime matching)
- **Why**: This IS the paper — regime-conditioned correction beats global/binary/unconditional alternatives

#### Operator Formal Definitions (backbone-agnostic)

**Operator 1 — Focal Semantic Residual** (for Sparse-Evidence regime):
- Input: frozen LM embedding e_i for focal node i
- Mechanism: r_i = σ(W_proj · e_i) (small learned projection, frozen after val selection)
- Application: h_i^(L) ← h_i^(L) + α · r_i (only at focal node, no neighbor propagation)
- On GATv2: adds to final-layer hidden state before classifier
- On BotRGCN: identical (backbone-agnostic — operates on final hidden state)
- Does NOT output independent label; does NOT modify other nodes

**Operator 2 — Ego-Subgraph Edge Reweight/Mask** (for Propagation-Corruption regime):
- Input: focal node i's 1-hop ego-subgraph edges
- Mechanism: compute per-edge harm score s_ij = f(disagreement_j, informativeness_j); mask or downweight edges with s_ij > threshold
- On GATv2: implemented as attention rewrite (multiply attention weights by (1 - s_ij))
- On BotRGCN: implemented as scalar edge mask (multiply message by (1 - s_ij))
- Only modifies edges incident to focal node; other nodes' aggregation unchanged

#### 8 Comparison Conditions

| ID | Condition | Description |
|----|-----------|-------------|
| D1 | No-op | Base GATv2 unchanged |
| D2 | Always Semantic | Apply focal semantic residual to ALL nodes |
| D3 | Always Repair | Apply ego-subgraph reweight to ALL nodes |
| D4 | Budget-Matched Random | Randomly assign semantic/repair to 15% of nodes (same budget as policy) |
| D5 | Uncertainty-only Trigger | Entropy threshold → apply single best action to flagged nodes |
| D6 | Binary Policy | Probe → intervene-or-not (single action, no regime typing). Corresponds to GLANCE-style binary router. |
| D7 | **Three-Action Policy** | Probe → regime evidence → matched action (no-op / semantic / repair) |
| D8 | **Regime-Conditional Action Swap** | Same screened set, same budget, same regime assignment — but SWAP actions (semantic↔repair within each regime) |

#### Key Causal Test (S1): Regime-Conditional Matched-Node Table

Not a global "swap and see overall F1 drop." Instead:
- Fix the screened node set (same nodes selected by probe in D7)
- Fix the intervention budget (same 15%)
- Within each regime: compare matched action vs swapped action on the SAME nodes
- Report per-regime: gain/broke/net for matched vs swapped
- Statistical test: paired bootstrap over nodes within each regime

- **Reuse**: [frmi_experiment.py](LLMbot/baseline/analysis/frmi_experiment.py), [action_outcome_table.py](LLMbot/baseline/analysis/action_outcome_table.py)
- **New code**: Focal semantic residual operator (~80 lines), ego-subgraph reweight operator (~120 lines), policy dispatcher (~60 lines), action swap variant (trivial — swap assignment in dispatcher)
- **Metrics**: Overall macro-F1, bot-F1, hard-subgroup ΔF1 (per pre-registered subgroup), easy-node preservation (bottom-50% safest), budget curve (intervention % vs gain)
- **Success criterion**: D7 > D2, D3, D5, D6 on macro-F1 AND hard-subgroup ΔF1; D7 > D8 with significant net loss in matched-node table
- **Failure interpretation**: If D7 ≈ D6 (binary policy) → three-action claim degrades; if D7 ≈ D2 → regime typing adds nothing
- **Table target**: Table 1 (main 8-condition comparison), Table 3 (regime-conditional matched-node action table)
- **Compute**: ~3h remote (8 conditions × 3 seeds)
- **Priority**: MUST-RUN

### Block B: Global Family vs Local Family (Week 5)

- **Claim tested**: P2 (FRMI is complementary to, not a subset of, global methods)
- **Why**: Must explicitly position against occupied claim-space (LGB, BotBR, BotGSL, GLANCE). Placed AFTER Block D because main table should first prove "local policy works" before comparing with global family.

#### Two-Tier Baseline Design

**Tier 1 — Occupied-Claim Baselines** (faithful reproduction where feasible):
- LGB (if code available): full LM+GNN semantic compensation pipeline
- BotBR (if code available): reliability-enhanced graph learning
- BotGSL (if code available): graph structure learning for bot detection
- If any cannot be faithfully reproduced → report published numbers + note in caption

**Tier 2 — Mechanism Proxies** (on our frozen GATv2, clearly labeled as proxies):
- All-Node Semantic Residual: apply Operator 1 to ALL nodes (not just sparse-evidence)
- Global GNNGuard-style Repair: apply edge reweighting to ALL edges (reuse [baseline_gnnguard.py](LLMbot/baseline/baselines/baseline_gnnguard.py))
- Global kNN/Mutual-kNN Graph Augmentation: add edges based on feature similarity

**Caption protocol**: Tier 2 entries are labeled as "mechanism proxy for [family]" in table caption. Never claim to be faithful reproduction of a specific paper.

- **Metrics**: Overall macro-F1, bot-F1, hard-subgroup ΔF1, easy-node drop, intervention budget (% nodes modified)
- **Success criterion**: FRMI (D7) competitive on overall metrics, superior on hard-subgroup ΔF1, with lower intervention budget than global methods (which touch 100% of nodes)
- **Failure interpretation**: If global methods show no gain at all → FRMI expectations also lower; if global methods dominate everywhere → FRMI story pivots to "efficiency at same gain"
- **Table target**: Table 4 (global vs local family comparison)
- **Compute**: ~4h remote
- **Priority**: MUST-RUN

### Block E: Appendix Experiments (Week 6)

- **C&S orthogonality**: Apply C&S on top of D7 output. Report in appendix only. Reuse [baseline_cs.py](LLMbot/baseline/baselines/baseline_cs.py).
- **BotRGCN confirmatory**: Repeat Blocks C+D on BotRGCN backbone. Success = P1/P2/S1 hold directionally (same sign), not same absolute gap.
- **TwiBot-20 cross-dataset**: Run full pipeline on TwiBot-20 (existing data). Success = regime enrichment and action differentiation directionally consistent.
- **Hyperparameter sensitivity**: Vary intervention budget (10%, 15%, 20%, 25%), projection dim, reweight temperature.
- **Compute**: ~4h remote
- **Priority**: NICE-TO-HAVE (but BotRGCN confirmatory is SHOULD-RUN for S2)

---

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0 (Wk1) | Protocol + backbone frozen | A0 + A1 | GATv2 trains stably on TwiBot-22 | 2h | Pipeline issues |
| M1 (Wk2-3) | Probe validated | Block C | utility@15% + HCW capture > uncertainty-only; ≥2 regimes enriched | 0.5h | Weak signal |
| M2 (Wk3-4) | Policy works | Block D (8 conditions) | D7 > D2,D3,D5,D6; D7 > D8 in matched-node table | 3h | Operators too weak |
| M3 (Wk5) | Global positioning | Block B | FRMI competitive overall, superior on hard subgroup at lower budget | 4h | Global methods unexpectedly strong |
| M4 (Wk6) | Portability + appendix | Block E | BotRGCN directionally consistent | 4h | Dataset-specific |

**Total must-run**: ~10 GPU-hours remote (M0-M3)
**Total should-run**: ~4 GPU-hours (M4)

**Statistical protocol**: Seed-level mean±std reported in all tables. Core comparisons (D7 vs D2/D3/D5/D6, matched vs swapped) use paired bootstrap over nodes on same test set (1000 resamples, 95% CI).

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| GATv2 diverges on TwiBot-22 | BotRGCN becomes primary; reweight operator uses scalar edge mask |
| Probe ≈ uncertainty-only on utility@budget | Add GATv2 attention entropy + per-head disagreement; try 1-layer MLP (still lightweight) |
| Three-action ≈ binary policy (D7 ≈ D6) | Pivot to "regime-aware budgeting" — same gain with fewer interventions and better subgroup targeting |
| Action swap non-significant | Increase power via paired bootstrap over all test nodes; if still non-significant → weaken S1 to "probe-triggered > unconditional" |
| Reviewer: "just calibration + post-hoc fix" | Block B explicitly separates global calibration family; emphasize: FRMI is local, node-specific, action-diverse, backbone-agnostic |
| Reviewer: "strawman global baselines" | Two-tier design: faithful reproductions where possible + clearly-labeled mechanism proxies |
| TwiBot-22 pipeline issues | Well-documented benchmark with community loaders; budget 2 days |
| OOF leakage concern | OOF is strictly within train split; main results use official split; document clearly in paper |

---

## Verification

1. Block A0: `PROTOCOL_FREEZE.md` written and reviewed before any experiment
2. Block A1: `python -c "from core.GNNs import GATv2Bot"` compiles; training converges; val F1 reasonable
3. Block C: Risk-capture curve of probe dominates uncertainty-only; enrichment in ≥2 subgroups
4. Block D: 8 conditions × 3 seeds complete; paired bootstrap CIs for D7 vs D6, D7 vs D8
5. Block B: All tier-2 proxies produce valid scores; tier-1 baselines either reproduced or published numbers cited
6. Block E: BotRGCN shows P1/P2/S1 directionally consistent
