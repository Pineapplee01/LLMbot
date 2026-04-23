# Idea Discovery Report v2 — Brief-Aligned

**Direction**: Adaptive failure-aware social bot detection (from RESEARCH_BRIEF.md)
**Date**: 2026-04-15
**Pipeline**: research-lit → idea-creator → novelty-check (inline) → critical review
**Supersedes**: v1 (2026-04-11, UGBP-focused) and docs/wiki/idea/FINAL_IDEA_DISCOVERY_REPORT.md (2026-04-07)

---

## Executive Summary

**Recommended idea**: **Failure-Regime-Conditioned Minimal Intervention (FRMI)** — a unified framework satisfying all 3 innovation claims from RESEARCH_BRIEF.md:

1. **Hard-node score d_i** = composite of prop_corruption_score (1.84x enrichment) + expert disagreement + structural rarity (degree bucket), going beyond LOGIN's uncertainty-only trigger
2. **3-action policy** via lightweight MLP head: No-op (dominant, ~82%) / Semantic Enhance (precision tool for graph-missing, ~3%) / Propagation Repair (directional reweighting on triggered nodes, ~15%)
3. **Beyond-pruning repair** via trust-weighted directional message attenuation: edges receive continuous weights w_ij ∈ [0,1] based on semantic conflict + relation consistency, strictly generalizing binary pruning

**Key insight**: The 3-action policy's value is in selective NON-intervention. No-op dominates because 82.4% of nodes are both_correct. Semantic Enhance targets the narrow 2.7% LM-only-correct slice (30% rescue rate on graph-missing). Propagation Repair targets the 14.2% disagreement slice where rw4 already shows +0.16 F1.

**Differentiation from LOGIN**: d_i uses structural + semantic signals (not uncertainty alone); policy is 3-way (not binary hard switch); repair is directional reweighting (not pruning-only).
**Differentiation from GLANCE**: Includes propagation-level repair (GLANCE preserves graph); d_i includes structural rarity (GLANCE uses homophily only); policy includes No-op as explicit action.

---

## Literature Landscape (Updated 2026-04-15)

### Thread 1: Hard-Node Detection Probes
| Paper | Venue | Key Contribution | Gap vs Our Work |
|-------|-------|-----------------|-----------------|
| G-DeltaUQ | ICLR 2024 | Epistemic uncertainty under distribution shift | Node-level only, no structural rarity |
| EviNet | arXiv 2025 | Evidential reasoning for OOD detection on graphs | No semantic-structural composite score |
| NCwR | arXiv 2024 | Node classification with reject option | Abstention only, no repair action |
| GLC-GNN | arXiv 2024 | Global+local confidence for fraud detection | No 3-action policy, no bot detection |
| C-EDL | arXiv 2025 | Post-hoc conflict-aware evidential uncertainty | No graph structure modification |

### Thread 2: Multi-Action Intervention Policies
| Paper | Venue | Key Contribution | Gap vs Our Work |
|-------|-------|-----------------|-----------------|
| Mowst | ICLR 2024 | Weak/strong expert routing via confidence gating | Binary (2-action), not 3-action |
| GLANCE | ICLR 2026 | Node-aware selective LLM invocation | No graph repair, binary invoke/skip |
| LOGIN | arXiv 2024 | LLM-as-consultant, uncertainty trigger | Uncertainty-only, hard switch |
| MoE-NP | arXiv 2024 | Mixture of node predictors | No propagation repair |
| Per-Sample Adaptive Routing | arXiv 2025 | Multimodal multitask routing | Not graph-specific |

### Thread 3: Beyond-Pruning Local Repair
| Paper | Venue | Key Contribution | Gap vs Our Work |
|-------|-------|-----------------|-----------------|
| Sparse Bayesian MP | arXiv 2026 | Signed adjacency posteriors | No semantic guidance |
| Torque-Driven Rewiring | arXiv 2025 | Physics-inspired edge reweighting | No failure-aware triggering |
| Directed Homophily-Aware GNN | arXiv 2025 | Direction-aware heterophily | No selective intervention |
| Self-Corrective Propagation | arXiv 2026 | Adversarial synthesis + correction | No semantic signals |
| LLM4RGNN | arXiv 2024 | LLM-guided edge purification | Binary prune, no soft reweighting |

### Confirmed Literature Gaps
1. **No 3-action intervention policy in graph learning**: All existing work is binary (route/skip, prune/keep, invoke/abstain)
2. **No composite hard-node score combining structural + semantic signals**: LOGIN uses uncertainty alone; GLANCE uses homophily alone
3. **No semantic-guided directional edge reweighting**: Sparse Bayesian MP is structural-only; LLM4RGNN is binary prune
4. **No failure-regime-conditioned intervention for bot detection**: Existing bot detectors apply uniform processing

---

## Ranked Ideas (All satisfy 3 innovation claims)

### 🏆 Idea 1: Failure-Regime-Conditioned Minimal Intervention (FRMI) — RECOMMENDED

**One-sentence**: A post-hoc framework that computes a composite hard-node score from structural rarity + semantic conflict + calibrated entropy, routes each node to one of three actions (No-op / Semantic Enhance / Propagation Repair) via a lightweight policy head, and performs directional trust-weighted message attenuation as the repair operator.

**Hard-node score d_i**:
```
d_i = α * prop_corruption_score(i) + β * disagreement(i) + γ * structural_rarity(i)
where:
  prop_corruption_score = 0.4*pred_homophily + 0.3*neighbor_entropy + 0.3*own_entropy  (validated: 1.84x enrichment)
  disagreement = 1[pred_sem ≠ pred_graph]  (validated: high precision at top-5%)
  structural_rarity = quantile_bucket(degree_i)  (graph_missing=1.0, low_degree=0.7, medium=0.3, high=0.0)
  α, β, γ learned on validation set (or fixed: 0.5, 0.3, 0.2)
```

**3-action policy**:
- Input: [d_i, q_sem_i, q_graph_i, degree_bucket_i, disagreement_i]
- Architecture: 2-layer MLP (5→16→3) with softmax → π_noop, π_enhance, π_repair
- Training: supervised on validation set using oracle action labels (No-op if both_correct, Enhance if lm_only_correct AND graph_missing, Repair if disagreement AND prop_corruption)
- Expected distribution: ~82% No-op, ~3% Enhance, ~15% Repair

**Propagation Repair (beyond pruning)**:
- For each triggered node i and incident edge (j→i):
  - Compute trust weight: w_ji = σ(a*trust_j + b*(1-conflict_ji) + c*relmatch_ji)
  - w_ji ∈ [0,1] — continuous, not binary prune
  - Re-propagate: h_i' = Σ_j w_ji * h_j (trust-weighted aggregation)
- This strictly generalizes pruning: w=0 is prune, w=1 is keep, w∈(0,1) is attenuation
- Run frozen graph expert once on reweighted graph

**Semantic Enhance**:
- For graph-missing nodes (degree=0) where d_i is high:
  - Replace graph expert prediction with semantic expert prediction
  - Inject LM embedding as node feature for re-propagation of neighbors
- Precision tool: targets ~3% of nodes where LM rescue rate is 30%

**Novelty**: 7/10
- Novel: 3-action policy with composite d_i in graph learning (no prior work)
- Novel: directional trust-weighted attenuation guided by semantic signals
- Incremental: builds on validated rw4 mechanism

**Risk**: LOW-MEDIUM
- Low risk on d_i (prop_corruption already validated at 1.84x)
- Low risk on repair (soft generalization of validated pruning)
- Medium risk on policy head (may collapse to 2 actions in practice)

**GPU-hours**: 3-5h on RTX 4060
**Expected gain**: +0.5-1.5pp F1 over current rw4 (0.765→0.775)

**Differentiation**:
- vs LOGIN: d_i uses structural+semantic (not uncertainty alone); 3-action (not binary); soft reweighting (not hard prune)
- vs GLANCE: includes propagation repair (GLANCE preserves graph); d_i includes structural rarity; explicit No-op action

---

### Idea 2: Regime-Adaptive Soft Surgery (RASS) — BACKUP

**One-sentence**: Partition nodes into failure regimes (graph_missing / prop_corruption / weak_support / normal) using prior-only signals, then apply regime-specific repair operators: semantic injection for graph_missing, trust-weighted attenuation for prop_corruption, and no-op for normal.

**Hard-node score d_i**:
```
regime_i = argmax(softmax(MLP([degree_i, pred_homophily_i, neighbor_entropy_i, own_entropy_i, disagreement_i])))
d_i = 1 - π_normal(regime_i)  (difficulty = probability of NOT being in normal regime)
```

**3-action policy**: Regime-to-action mapping (structured, not learned end-to-end):
- graph_missing regime → Semantic Enhance
- prop_corruption regime → Propagation Repair
- weak_support regime → Propagation Repair (lighter budget)
- normal regime → No-op

**Propagation Repair**: Same trust-weighted attenuation as FRMI, but with regime-specific budgets:
- prop_corruption: aggressive attenuation (lower w threshold)
- weak_support: conservative attenuation (higher w threshold)

**Novelty**: 6/10 — Regime-conditioned intervention is novel; regime definitions are heuristic
**Risk**: LOW — Builds directly on validated regime diagnostics
**GPU-hours**: 2-3h
**Expected gain**: +0.3-1.0pp F1 over rw4

---

### Idea 3: Uncertainty-Gated Belief Propagation + Policy (UGBP-P) — HIGH NOVELTY

**One-sentence**: Extend UGBP (from v1 report) with a 3-action policy head and directional trust weights, making it satisfy all 3 innovation claims.

**Hard-node score d_i**:
```
d_i = entropy(b_i) * (1 - agreement_i) + structural_rarity_i
where b_i = belief after 1 round of uncertainty-gated BP
```

**3-action policy**: Based on belief convergence:
- If b_i converges quickly (< 0.1 change) → No-op
- If b_i diverges AND degree=0 → Semantic Enhance
- If b_i diverges AND degree>0 → Propagation Repair (trust-weighted BP messages)

**Propagation Repair**: Trust-weighted BP messages:
- m_ji = w_ji * ((1-u_j)*b_j + u_j*uniform) where w_ji = trust weight from semantic conflict
- Strictly generalizes both pruning (w=0) and standard BP (w=1)

**Novelty**: 8/10 — Uncertainty-gated BP with 3-action policy is novel
**Risk**: MEDIUM — BP convergence on social graphs is uncertain
**GPU-hours**: 4-6h
**Expected gain**: +1.0-2.0pp F1 over rw4 (if BP converges)

---

### Idea 4: Evidential Difficulty Probe + Adaptive Repair (EDP-AR) — ALTERNATIVE

**One-sentence**: Train a lightweight evidential probe (Dirichlet output) on frozen GNN features to estimate per-node difficulty, then use the evidence mass to route between 3 actions and perform trust-weighted repair.

**Hard-node score d_i**:
```
α_i = EvidentialProbe(h_gnn_i)  (Dirichlet parameters)
d_i = K / sum(α_i)  (inverse total evidence = vacuity)
Combined: d_i_final = d_i * (1 + disagreement_i) * structural_rarity_i
```

**3-action policy**: Thresholded on d_i_final:
- d_i_final < τ_low → No-op
- d_i_final > τ_high AND degree=0 → Semantic Enhance
- d_i_final > τ_high AND degree>0 → Propagation Repair

**Novelty**: 6/10 — Evidential probes exist; combining with 3-action policy is novel
**Risk**: MEDIUM — Evidential probe may not calibrate well on small dataset
**GPU-hours**: 3-4h

---

### Idea 5: Semantic-Structural Conflict Router (SSCR) — SIMPLER VARIANT

**One-sentence**: Use the cosine distance between LM embedding and GNN embedding as the primary hard-node signal, route via learned 3-way head, repair via embedding-guided edge reweighting.

**Hard-node score d_i**:
```
d_i = 1 - cos(z_sem_i, z_graph_i)  (semantic-structural conflict)
```

**3-action policy**: MLP on [d_i, q_sem, q_graph, degree] → 3-way softmax
**Propagation Repair**: Reweight edges by neighbor embedding alignment: w_ji = cos(z_sem_j, z_graph_i)

**Novelty**: 5/10 — Embedding conflict is straightforward
**Risk**: LOW
**GPU-hours**: 2-3h

---

### Idea 6: Counterfactual Edge Utility with Soft Decisions (CEU-S) — EXTENSION OF RW4

**One-sentence**: Extend the validated rw4 keep-utility scoring to output continuous edge weights instead of binary prune decisions, and add a 3-action policy head on top.

**Hard-node score d_i**: Same as rw4 trigger (disagreement + low q_graph), extended with prop_corruption_score
**3-action policy**: Rule-based on trigger + degree:
- Not triggered → No-op
- Triggered AND degree=0 → Semantic Enhance
- Triggered AND degree>0 → Propagation Repair

**Propagation Repair**: w_ji = σ(keep_utility_ji) instead of 1[keep_utility > τ]

**Novelty**: 4/10 — Minimal extension of validated mechanism
**Risk**: LOW — Closest to what already works
**GPU-hours**: 1-2h
**Expected gain**: +0.1-0.5pp F1 over rw4

---

## Eliminated Ideas

| Idea | Reason Eliminated |
|------|-------------------|
| Pure UGBP (v1 Idea 1) | Doesn't satisfy Claim 2 (no 3-action policy) without extension |
| CC-PPR (v1 Idea 2) | Strong baseline, not a method paper; no 3-action policy |
| Global Graph Rewrite | Explicitly excluded by RESEARCH_BRIEF.md |
| Homophily-Only Routing | Too close to GLANCE; excluded by brief |
| Structural Entropy Core | Too close to SEBot; excluded by brief |
| Multi-View Graph Reliability Router | Too heavy for RTX 4060; overfitting risk |
| Learned Rewrite Trigger from Expected Utility | Circular supervision; shaky training signal |
| Explanation as Reliability Feature | Templated explanations too weak |

---

## Novelty Assessment (Inline)

### FRMI (Idea 1) vs Key Competitors

| Aspect | LOGIN | GLANCE | Mowst | FRMI (Ours) |
|--------|-------|--------|-------|-------------|
| Hard-node score | Uncertainty only | Homophily + uncertainty | Confidence dispersion | Composite: prop_corruption + disagreement + structural rarity |
| Actions | 2 (consult/skip) | 2 (invoke/skip) | 2 (weak/strong) | 3 (No-op / Enhance / Repair) |
| Graph repair | Prune only | None | None | Trust-weighted directional attenuation |
| Semantic guidance | LLM features | LLM invocation | None | LLM confidence + embedding conflict |
| Bot detection | No | No | No | Yes (TwiBot-20) |

**Novelty verdict**: The combination of (1) composite d_i, (2) 3-action policy, and (3) semantic-guided directional attenuation is novel. No single paper covers all three. Closest: LOGIN covers (partial 1) + (partial 3); GLANCE covers (partial 1) + (partial 2).

---

## Answers to RESEARCH_BRIEF.md Questions

### Q1: How should d_i be defined?

**Recommended (FRMI)**: Weighted composite
```
d_i = 0.5 * prop_corruption_score(i) + 0.3 * disagreement(i) + 0.2 * structural_rarity(i)
```
- prop_corruption_score: already validated at 1.84x enrichment (prior-only, deployable)
- disagreement: high precision at top-5% (50.5% precision)
- structural_rarity: degree-based bucket (graph_missing=1.0, low=0.7, medium=0.3, high=0.0)

**Alternative A**: Learned MLP on [prop_corruption, disagreement, degree, q_sem, q_graph] → scalar d_i
**Alternative B**: Evidential probe vacuity * (1 + disagreement) * structural_rarity

**Recommendation**: Start with fixed weights (0.5/0.3/0.2), validate on val set, then optionally learn weights.

### Q2: Best v1 intervention policy?

**Recommended**: Hybrid (rule-based structure + learned thresholds)
- Structure: degree=0 AND high d_i → Enhance; degree>0 AND high d_i → Repair; else → No-op
- Thresholds: τ_enhance and τ_repair learned on validation set
- Why not fully learned: 3% Enhance coverage is too small for stable MLP training
- Why not fully rule-based: optimal thresholds depend on calibration quality

### Q3: Minimum viable implementation of each branch?

**Semantic Enhance**:
- Operation: Replace graph expert logits with semantic expert logits for triggered nodes
- Injection: For neighbors of enhanced nodes, add LM embedding as auxiliary feature
- Scope: ~3% of nodes (graph_missing with high d_i)

**Propagation Repair**:
- Operation: For each incident edge (j→i) of triggered node i:
  - w_ji = σ(a*trust_j + b*(1-conflict_ji) + c*relmatch_ji)  (reuse rw4 utility components)
  - Re-propagate: h_i' = Σ_j w_ji * W * h_j  (trust-weighted aggregation)
- Scope: ~15% of nodes (disagreement + high prop_corruption)
- One pass, frozen graph expert weights

### Q4: Differentiation from LOGIN and GLANCE?

**vs LOGIN**:
- d_i: composite (structural + semantic) vs uncertainty-only
- Policy: 3-action vs binary hard switch
- Repair: directional attenuation vs pruning-only
- Mandatory ablation: d_i vs uncertainty-only trigger (shows composite is better)

**vs GLANCE**:
- Includes propagation repair (GLANCE preserves graph structure)
- d_i includes structural rarity (GLANCE uses homophily primarily)
- Explicit No-op action (GLANCE's skip is implicit)
- Mandatory ablation: with vs without repair (shows repair adds value beyond routing)

### Q5: Claim-to-experiment matrix?

| Claim | Experiment | Kill Criterion |
|-------|-----------|---------------|
| C1: d_i identifies hard nodes | d_i AUROC for GNN errors > 0.82 (beat entropy-only 0.819) | AUROC < 0.82 |
| C2: 3-action > 2-action | FRMI F1 > binary-policy F1 on disagree slice | No improvement |
| C3: Repair > No-repair | FRMI F1 > enhance-only F1 on prop_corruption slice | No improvement |
| C4: Soft attenuation > hard prune | Attenuation ECE ≤ prune ECE AND F1 ≥ prune F1 | ECE worse |
| C5: Competitive with baselines | Global F1 ≥ LM-only (0.8756) | F1 < 0.87 |

**Ablation table**:
| System | d_i | Policy | Repair |
|--------|-----|--------|--------|
| Base GNN | - | - | - |
| LM only | - | - | - |
| rw4 (current) | disagreement+q_graph | binary | prune |
| FRMI-no-probe | uniform | 3-action | attenuation |
| FRMI-no-policy | composite d_i | binary | attenuation |
| FRMI-prune-only | composite d_i | 3-action | prune |
| **FRMI (full)** | **composite d_i** | **3-action** | **attenuation** |

---

## Next Steps

1. [ ] Implement FRMI in `LLMbot/` — extend DualRouterTrainer with:
   - `compute_difficulty_score()` for composite d_i
   - `policy_head` MLP (5→16→3)
   - `trust_weighted_propagation()` for soft attenuation
2. [ ] Run single-seed pilot (seed 42) on canonical split
3. [ ] Compare against rw4, LM-only, and ablation variants
4. [ ] If pilot passes kill criteria → multi-seed (42, 123, 456)
5. [ ] External review via `/research-review` (when Codex MCP auth is restored)
6. [ ] Method refinement via `/research-refine-pipeline`

---

---

## External Review (GPT-5.4 xhigh, 3 rounds)

**Score**: 4/10 (Weak Reject) → potential 6/10 with minimum experiment package
**Thread**: 019d8fb9-7815-7cc2-a82b-638452564385
**Full review**: [docs/reviews/external_review_2026-04-15.md](docs/reviews/external_review_2026-04-15.md)

### Key Verdicts
1. **3-action policy**: Defensible as framework/taxonomy claim IF ablation shows removing Enhance hurts graph_missing slice
2. **Soft attenuation**: Needs w_ji histogram (not bimodal) + beats hard prune at matched budget
3. **Composite d_i**: Must show AUROC > entropy-only and homophily-only via validation-trained logistic regression
4. **Single dataset**: +1 score point if TwiBot-22 transfer works
5. **both_wrong**: Not fatal if framed honestly as "outside claim scope"

### Recommended Reframing
**From**: "Adaptive failure-aware social bot detection"
**To**: "Selective local correction for recoverable failure regimes in social bot detection"

### Minimum Experiment Package (4/10 → 6/10)
1. 3-seed unified main table on TwiBot-20
2. Controller necessity ablation (3A vs 2A, slice metrics)
3. TwiBot-22 transfer OR matched-budget soft vs hard repair

### Venue Reality Check
- WWW 2026 / AAAI 2026: deadlines passed
- **ASONAM 2026**: abstract Apr 19, paper Apr 26 (NEAREST)
- **CIKM 2026**: abstract May 16, paper May 23 (short/applied)
- **ICML 2026 workshops**: ~Apr 24
- **WSDM 2027**: CFP not yet posted (best future main-track target)

---

**Report generated by**: Claude Opus 4.6 via ARIS idea-discovery pipeline
**External review by**: GPT-5.4 xhigh via Codex MCP (3 rounds)
**Status**: COMPLETE — Ideas generated, ranked, reviewed, and submission plan provided
