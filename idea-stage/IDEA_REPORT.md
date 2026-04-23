# Idea Discovery Report: FRMI for Social Bot Detection (v2)

**Direction**: Failure-Regime-Conditioned Minimal Intervention for Social Bot Detection
**Date**: 2026-04-22
**Pipeline**: research-lit → idea-creator → novelty-check → research-review → research-refine-pipeline
**Model**: Claude Opus 4.6 (1M context)
**Effort**: beast

---

## Executive Summary

The defensible paper identity is **regime-typed residual error recovery under minimal intervention**, not any specific operator. B1 (gain-based action selection) is the headline method contribution. A2 (contrastive residual adapter) is a candidate supporting operator pending S0 tribunal — it is NOT the paper identity and scores only 3/10 on novelty as a standalone mechanism.

Current claim stability: 4/10. D7 underperforms D3 in R012 Seed 1. The semantic arm's existence is unproven. Next step is S0 operator tribunal (prove semantic arm exists), NOT system-level R012b deployment.

The narrow novelty space: to our knowledge, no prior work explicitly formulates interpretable failure regimes and learns matched local corrective actions in a frozen-backbone post-hoc bot-detection setting. But this is combination novelty in a narrow gap between LGB (sparse semantic rescue), GLANCE (selective LLM routing), EGNN/GRE (editable GNN correction), Node-MoE (per-node adaptive treatment), SimCalib/SCAR/WATS (post-hoc graph calibration), and BotBR/RABot (reliability-enhanced graph learning). The paper can only survive if strong empirical proof demonstrates that multi-action matched correction outperforms unconditional repair and binary routing under fixed budget.

---

## Phase 1: Literature Landscape (2026-04-22)

### Delta from 2026-04-07 Survey

| Finding | 2026-04-07 | 2026-04-22 |
|---------|-----------|-----------|
| Core novelty (regime-typed intervention) | Unchecked | Combination novelty in narrow gap — to our knowledge, no prior work explicitly formulates interpretable failure regimes + matched local correction in frozen-backbone post-hoc bot-detection setting |
| Semantic operator alternatives | None identified | TTReFT, UAdapterGNN, TEA-GLM |
| Edge repair mechanisms | GNNGuard only | RABot, Shapley Sparsification, LASER |
| Bot detection SOTA | LMbot F1≈0.87 | HW-GNN F1=0.9151, RABot F1=0.8840 |
| Diagnostic framing precedents | None | "Why Does Your GNN Fail?" (2509.10337) |

### Key New Papers

| Paper | Venue | Relevance | Key Insight |
|-------|-------|-----------|-------------|
| TTReFT (2601.21615) | arXiv 2026 | VERY HIGH | Low-rank interventions on high-uncertainty nodes of frozen GNN |
| RABot (2602.21749) | arXiv 2026 | VERY HIGH | RL-driven edge filtering for bot detection, TwiBot-20 F1=88.40% |
| UAdapterGNN (2511.18859) | arXiv 2025 | HIGH | Gaussian probabilistic adapters on frozen GNN backbone |
| Shapley Sparsification (KDD WS 2025) | KDD 2025 | HIGH | Post-hoc Shapley-based edge pruning, 80% edges removed <2% loss |
| L2D Post-Processing (2407.12710) | arXiv 2024 | HIGH | Principled multi-objective deferral — drop-in replacement for action rule |
| HINet (2510.21457) | arXiv 2025 | HIGH | Treatment effect estimation on graphs with domain adversarial training |
| BotHP (2506.00989) | KDD 2025 | HIGH | Dual-encoder + prototype clustering, TwiBot-20 acc=87.73% |
| BotUmc (2503.03775) | arXiv 2025 | HIGH | Dempster-Shafer uncertainty + causal graph intervention |
| NodePro (2509.12094) | arXiv 2025 | MEDIUM | Instance-level GNN profiling — diagnostic only, no correction |
| Node-wise Filtering MoE (2406.03464) | arXiv 2024 | MEDIUM | Per-node adaptive spectral filters — implicit routing, not typed |
| RobustCRF (2411.05399) | arXiv 2024 | MEDIUM | Post-hoc CRF-based GNN robustness enhancement |
| HW-GNN (2511.22493) | arXiv 2025 | MEDIUM | Spectral bot detection, TwiBot-20 F1=91.51% |
| GNFBC (2603.03662) | arXiv 2026 | MEDIUM | Post-hoc negative feedback correction for heterophily — uniform, not typed |
| Grimm (2412.08555) | arXiv 2024 | LOW | Plug-and-play perturbation rectifier (adversarial, not failure-regime) |

### TwiBot SOTA Landscape (2026-04-22)

| Method | TwiBot-20 Acc | TwiBot-20 F1 | TwiBot-22 Acc | TwiBot-22 F1 |
|--------|--------------|-------------|--------------|-------------|
| HW-GNN | — | 91.51% | — | 61.95% |
| RABot | 87.92% | 88.40% | — | — |
| RMNP | 87.04% | 88.88% | — | — |
| BotHP | 87.73% | — | — | — |
| BotUmc | 87.37% | — | 72.71% | 58.52% |
| BotSCL | 87.26% | — | 82.39% | — |
| BotDGT | — | — | 79.33% | 58.15% |
| LGB | — | — | 80.42% | — |
| Our RGCN | — | 87.23% | — | — |

### Novelty Assessment (G1 Gate)

**PASS**: No published work does regime-typed (not binary) GNN intervention with matched per-regime corrections. Comprehensive check (13 papers):

| Component | Exists? | Closest Work | Gap vs Ours |
|-----------|---------|-------------|-------------|
| Binary regime (homophily/heterophily) | Yes | Platonov et al. NeurIPS 2023 | Binary, not typed |
| Per-node adaptive filtering | Yes | Node-wise Filtering MoE (2406.03464) | Implicit routing, not explicit typed regimes |
| Typed risk regimes (3+ types) | Yes (regression) | PAGER (2023) | Not for graphs |
| Node-level profiling | Yes | NodePro (2509.12094) | Diagnostic only, no correction |
| Quadrant analysis (when graph helps) | Yes | Diagnostic Study (2512.12947) | No per-quadrant correction |
| Post-hoc GNN correction | Yes | GNFBC (2603.03662) | Uniform correction, not typed |
| **Typed regimes + matched correction** | **NO** | **— our novelty claim —** | |
| Editable GNN (frozen + targeted correction) | Yes | EGNN (2305.15529), GRE (NeurIPS 2024) | Frozen GNN + MLP editor for misclassified nodes — same setting, different mechanism |
| Post-hoc graph calibration + prototypes | Yes | SimCalib, SCAR, WATS | Node-wise correction with class centroids/structural signatures — covers calibration slot |
| Selective LLM/semantic usage | Yes | GLANCE, E-LLaGNN, LOGIN | Binary "query LLM or not" — we extend to 3 actions |
| Sparse semantic rescue (bot detection) | Yes | LGB (2406.08762) | Isolated nodes need semantic compensation — we add regime typing |
| Reliability-enhanced graph learning | Yes | BotBR, RABot (2602.21749) | Edge filtering/reliability — we do post-hoc, they do in-training |

Closest threat: EGNN/GRE (frozen GNN + targeted correction on misclassified nodes). Our differentiation: (a) explicit interpretable regime types vs implicit error detection, (b) multi-action matched correction vs single correction type, (c) gain-based action selection vs uniform correction. But this is a narrow gap, not an open field.

---

## Phase 2: Ranked Ideas

### Category A — Fix the Semantic Operator

#### A1: TTReFT-Style Low-Rank Residual Injection
- **Core**: Replace Ridge regression with low-rank intervention matrices (rank r=4-8) trained on OOF failure labels. Instead of `LM_emb @ W ≈ h_gnn` (reproduces errors), learn `h_corrected = h_gnn + ΔW * LM_emb` where ΔW is trained to minimize failure prediction loss.
- **Why better**: TTReFT shows this works on frozen GNN backbones. Low-rank keeps it lightweight. Trained on failure labels, not on reproducing GNN states.
- **Implementation**: Replace `Ridge(alpha=1.0).fit(lm_train, h_gnn_train)` with `nn.Linear(lm_dim, rank) → nn.Linear(rank, hidden_dim)` trained on OOF failure cross-entropy.
- **Risk**: May overfit to OOF failure patterns. Needs careful regularization.
- **Estimated effort**: 1-2 days code, 0.5h GPU per seed.

#### A2: Contrastive Residual Adapter (RECOMMENDED)
- **Core**: Train a small adapter (2-layer MLP, 768→64→128) with contrastive loss: push LM projection toward correct-class prototype centroid, pull away from wrong-class centroid. The adapter learns to CORRECT, not reproduce.
- **Why better**: Contrastive alignment (TEA-GLM insight) preserves semantic structure. Class prototypes are computed from training nodes. The adapter output is added as residual to h_gnn only at focal nodes.
- **Loss**: `L = -log(sim(adapter(LM_i), proto_correct) / (sim(adapter(LM_i), proto_correct) + sim(adapter(LM_i), proto_wrong)))` on OOF failure nodes.
- **Implementation**: New `ContrastiveResidualAdapter` class replacing `FocalSemanticResidual`. Train on OOF failures within train split. Apply at test time to probe-selected nodes.
- **Risk**: Prototype quality depends on class balance. Need to handle class imbalance.
- **Estimated effort**: 2-3 days code, 1h GPU per seed.

#### A3: Uncertainty-Gated LM Injection
- **Core**: UAdapterGNN-inspired. Learn a Gaussian adapter: `μ, σ = adapter(LM_emb)`. Gate: `g_i = sigmoid(1/σ_i - threshold)`. Inject: `h_corrected = h_gnn + g_i * μ_i`. High uncertainty (large σ) → gate closes → no injection.
- **Why better**: Built-in uncertainty quantification. Self-regulating — won't inject when uncertain about the correction.
- **Risk**: Gaussian reparameterization adds training complexity. May be overkill for our setting.
- **Estimated effort**: 3-4 days code, 1h GPU per seed.

#### A4: Direct Logit Blending (Simplest Baseline)
- **Core**: Skip hidden state injection entirely. Blend at output level: `p_final = α * softmax(W_lm @ LM_emb) + (1-α) * p_gnn` for focal nodes. Train W_lm on OOF failures.
- **Why better**: Simplest possible semantic operator. No hidden state alignment needed. Already partially validated in Phase 4 (semantic_a0.3 gave +8.3pp target acc).
- **Risk**: Less principled than hidden-state injection. May not capture GNN-specific failure patterns.
- **Estimated effort**: 0.5 days code, 0.5h GPU per seed.

### Category B — Strengthen the Policy

#### B1: Utility-Aware Action Selection
- **Core**: For each probe-selected node, estimate expected gain per action: `utility(node_i, action_j) = probe_features @ W_j`. Select action with highest estimated utility. Train W_j on OOF action outcomes.
- **Why better**: Replaces blunt `sparse > prop` rule with learned utility estimation. Can capture non-linear regime boundaries.
- **Upgrade path**: L2D Post-Processing framework (2407.12710) provides a principled multi-objective deferral that could be a drop-in replacement.
- **Risk**: Requires OOF action outcome labels (need to run both operators on OOF nodes and measure which helped).
- **Estimated effort**: 1-2 days code, 1h GPU for OOF action outcomes.

#### B2: Learned Regime Classifier
- **Core**: Replace `sparse_score > prop_score` with small MLP (8→16→3) on probe features. Output: P(no-op), P(semantic), P(repair). Train on OOF action outcomes.
- **Why better**: Can learn non-linear regime boundaries. Probe features already extracted.
- **Risk**: Small training set (only OOF failure nodes). May overfit.
- **Estimated effort**: 0.5 days code, minimal GPU.

### Category C — Reframe if D7 Can't Be Saved

#### C1: Probe-as-Contribution Paper
- **Core**: The two-layer probe (risk + regime evidence) IS the novelty. Show it captures failure-prone nodes better than uncertainty-only. Operators are secondary — show they help but don't claim they're the main contribution.
- **Paper title**: "Typed Failure Regimes in Graph-Based Social Bot Detection: Diagnosis and Targeted Correction"
- **Venue**: CIKM 2026 short/applied, ASONAM 2026
- **Risk**: Weaker contribution if operators don't work. But probe results (Block C) are already strong.

#### C2: Two-Action Paper (Drop Semantic Arm)
- **Core**: D3 (always-repair) already works. Drop the semantic arm entirely. Paper becomes: probe → repair-or-not. Simpler, cleaner, and D3 > D1 is already shown.
- **Paper title**: "When to Repair: Probe-Guided Local Edge Correction for Social Bot Detection"
- **Venue**: Same as C1.
- **Risk**: Loses the "regime-typed" novelty. Becomes binary intervention.

#### C3: Pure Diagnostic Paper
- **Core**: No intervention claim. Paper is: (1) failure regimes exist and are typed, (2) embedding quality >> graph modification, (3) class-conditional disagreement frontier. All evidence already collected.
- **Paper title**: "When Does Graph Structure Hurt Social Bot Detection? A Failure Regime Analysis"
- **Venue**: Workshop or CIKM short.
- **Risk**: Lowest novelty. But all evidence is already in hand.

---

## Phase 2.5: Idea Ranking (Pre-Pilot)

| Rank | Idea | Novelty | Feasibility | Narrative | Total | Category |
|------|------|---------|-------------|-----------|-------|----------|
| 1 | B1: Gain-Based Action Selection (HEADLINE) | 5/10 | 6/10 | 9/10 | **20/30** | Decision layer |
| 2 | A4: Direct Logit Blending (FIRST GATE BASELINE) | 4/10 | 10/10 | 7/10 | **21/30** | Semantic op |
| 3 | A2: Contrastive Residual Adapter (PENDING TRIBUNAL) | 3/10 | 7/10 | 6/10 | **16/30** | Semantic op |
| 4 | A1: TTReFT Low-Rank Residual | 3/10 | 8/10 | 6/10 | **17/30** | Semantic op |
| 5 | A3: Uncertainty-Gated Injection | 4/10 | 5/10 | 6/10 | **15/30** | Semantic op |
| 6 | B2: Learned Regime Classifier | 3/10 | 9/10 | 5/10 | **17/30** | Policy |
| 7 | C1: Probe-as-Contribution (FALLBACK) | 6/10 | 10/10 | 7/10 | **23/30** | Reframe |
| 8 | C2: Two-Action Paper | 4/10 | 10/10 | 6/10 | **20/30** | Reframe |
| 9 | C3: Pure Diagnostic | 3/10 | 10/10 | 5/10 | **18/30** | Reframe |

Note: B1 is the headline contribution. A2/A4 are operator candidates under S0 tribunal. C1 is the fallback if any gate fails.

---

## Phase 2.5: Pilot Assessment

**Pilot status**: Cannot run locally — data and checkpoints are on remote GPU server (172.31.106.108). Pilots deferred to `/run-experiment` phase.

**Paper-based pilot assessment** (using existing Phase 4 evidence):
- A4 (Direct Logit Blending) is already partially validated: Phase 4 showed semantic_a0.3 gave +8.3pp target accuracy on frozen RGCN. This is the strongest existing evidence.
- A2 (Contrastive Residual Adapter) is theoretically stronger than A4 but unvalidated.
- A1 (TTReFT Low-Rank) is the most principled approach but requires implementation.

**Decision gate G2**: Deferred to remote pilot. For now, proceed with A2 as recommended direction based on theoretical analysis + A4 as validated fallback.

---

## Phase 3: Novelty Verification

### A2: Contrastive Residual Adapter — Novelty Score: 3/10

**Closest prior work (expanded threat set):**
1. ConGraT (2305.14321): Contrastive graph-text alignment — directly covers the contrastive mechanism
2. TEA-GLM (2408.14512): Contrastive alignment between GNN and LLM spaces
3. SimCalib, SCAR: Class prototypes / centroids in node-level post-hoc correction / calibration
4. WATS: Node-specific post-hoc calibration
5. EGNN (2305.15529): Frozen GNN + stitched MLP + targeted correction on misclassified nodes
6. GRE (NeurIPS 2024): Editable GNN at conference level
7. TTReFT (2601.21615): Uncertainty-guided node selection + low-rank representation intervention

**Assessment**: A2 is application-specific recombination, not mechanism innovation. Each component (contrastive alignment, class prototypes, frozen-backbone correction, node selection) has clear precedent. The combination applied to bot detection failure nodes is a "clever integration" that top venues will likely reject as insufficient novelty. A2 can only survive as a supporting operator, never as headline.

### A1: TTReFT Low-Rank Residual — Novelty Score: 5/10

**Closest prior work:**
1. TTReFT itself (2601.21615): Almost identical mechanism. Our differentiation is (a) using LM embeddings as the source signal and (b) regime-typed node selection.
2. LoRA/adapter literature: Low-rank residual injection is well-established.

**Risk**: Too close to TTReFT. Would need strong differentiation in the regime-typing aspect.

### B1: Gain-Based Action Selection — Novelty Score: 5/10

**Closest prior work:**
1. GLANCE (2510.10849): Advantage-based utility view — binary "query LLM or not" router. We extend to 3 actions.
2. L2D (2407.12710): General multi-objective defer/post-processing framework.

**Assessment**: Extending binary routing to three-action matched correction is a reasonable increment, not headline-level independent innovation. B1 is only valuable if empirical proof shows multi-action matched correction significantly outperforms binary routing and unconditional operator. Without that proof, B1 is just a routing small upgrade.

### Overall System Novelty: Combination novelty, medium — requires strong empirical proof

The system-level novelty (two-layer probe + explicit regime typing + gain-based multi-action policy) occupies a narrow gap between LGB, GLANCE, EGNN/GRE, Node-MoE, SimCalib/SCAR/WATS, and BotBR/RABot. Individual components have clear precedents. The combination is defensible only with strong empirical evidence that matched correction outperforms all simpler alternatives under fixed budget.

**G3 Gate: CONDITIONAL** — depends on S0-S3 experimental results. No gate has been passed yet.

---

## Phase 4: Critical Review (Self-Review, No Codex MCP Available)

### Reviewer Perspective (Simulated NeurIPS/ICML Level)

**Strengths:**
1. Well-motivated problem: failure regimes in GNN-based bot detection are empirically validated (Phase 1: 1.84x enrichment)
2. Clean experimental design: frozen backbone, post-hoc correction, pre-registered subgroups
3. Comprehensive comparison: 8 conditions including causal swap test (D8)
4. Regime typing is genuinely novel — no prior work does this

**Weaknesses:**
1. **W1 (Critical)**: D7 underperforms D3 in Seed 1. The three-action policy must beat always-repair to justify the complexity. If the semantic arm doesn't help, the paper degrades to a two-action paper (C2).
2. **W2 (Major)**: Single dataset (TwiBot-22). Need at least directional consistency on TwiBot-20 or another dataset.
3. **W3 (Major)**: The semantic operator (Ridge regression) is fundamentally flawed. Replacing it with A2 (contrastive adapter) is necessary but adds implementation risk.
4. **W4 (Minor)**: Probe training on OOF failure labels is a conservative approximation (not true k-fold OOF). Should acknowledge this.

**Verdict**: 6/10 projected (with A2 fix). 4/10 without fixing the semantic operator.

**Minimum viable improvements:**
1. Replace Ridge regression with contrastive adapter (A2) — MUST
2. Show D7 > D3 on at least 2 of 3 seeds — MUST
3. Add utility-aware action selection (B1) if D7 ≈ D6 — SHOULD
4. TwiBot-20 directional consistency — SHOULD

**G4 Gate: CONDITIONAL PASS** — 6/10 projected if A2 is implemented and D7 > D3.

---

## Phase 4.5: Corrected Experiment Sequence (Post-Review)

### Experiment Resequencing

The original plan ("deploy R012b with A2") is NOT approved. The correct sequence isolates each component for independent validation:

| Stage | Purpose | What to run | Kill Gate |
|-------|---------|-------------|-----------|
| S0 | Semantic arm existence | A4 vs A2 vs EGNN-style vs all-node-fusion, same sparse-screened set, 2 datasets, 3 seeds | Best semantic op > no-op stably; A2 > A4 to survive as named operator |
| S1 | Repair stability | Local propagation repair on TwiBot-20 + TwiBot-22, 3 seeds | Repair > no-op stably |
| S2 | Decision layer | B1 vs heuristic rule vs binary router, same operator, same budget | B1 > heuristic AND binary router |
| S3 | Full system | Three-action FRMI, 8 conditions, 3 seeds | D7 > D3, binary, unconditional; matched > swapped significantly |

Any gate fails → immediate claim downgrade. No full FRMI methods paper.

### Operator Naming (Paper vs Code)

| Internal (code) | Paper name |
|-----------------|-----------|
| ContrastiveResidualAdapter | sparse semantic patch operator |
| EgoEdgeReweight | local propagation repair operator |
| PolicyDispatcher + B1 | gain-based action selection / regime-conditioned minimal intervention layer |

### Required Baselines (must survive review)

1. EGNN-style frozen GNN+MLP editor
2. Direct logit blending (A4)
3. All-node semantic fusion
4. GLANCE-style binary router
5. Node-MoE / heterophily-adaptive
6. BotBR / RABot reliability baselines

---

## Next Steps

1. S0 operator tribunal — prove semantic arm exists (or kill it)
2. S1 repair stability — prove local propagation repair works
3. S2 decision layer — prove B1 > heuristic/binary
4. S3 full FRMI — only if S0-S2 pass
5. If any gate fails → pivot to C1 (probe-as-contribution) or diagnostic paper

See `idea-stage/PAPER_CHARTER.md` for allowed/forbidden claims and required proof.
