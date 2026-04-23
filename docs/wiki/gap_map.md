# Field Gap Map

*Stable IDs for known gaps in the research landscape*

Last updated: 2026-04-15T00:00:00Z

---

## Active Gaps

### G1: Hard Rewrite Assumes One-Size-Fits-All Intervention
**Status**: `partially_addressed`  
**Severity**: High  
**Evidence**: Router underperforms both single experts (F1=0.761 vs 0.766/0.767)

**Problem**: Current calibration-guided graph rewrite assumes the same intervention works for all suspicious nodes. However, heterophily ≠ noise, and different node types (low-degree, graph-missing, disagreement) need different strategies.

**Implication**: Need regime-based routing or selective abstention instead of uniform topology surgery.

**Addressed by**: 
- rw4_disagreement_local: targeted repair only on triggered nodes (14.2% coverage)
- claim:C5: repair beats deferral on triggered nodes (+0.16 F1)
- claim:C8: v2 diagnostics show graph_missing nodes have 30% LM rescue rate vs 18% elsewhere; prop_corruption nodes show 23.8% vs 15.2% — different regimes, different action preferences

---

### G2: Node-Level Calibration ≠ Edge-Level Trust
**Status**: `unresolved`  
**Severity**: Medium  
**Evidence**: GATS (NeurIPS'22) explicitly shows this mismatch

**Problem**: `q_graph` is a node-level signal but used for edge pruning decisions. This creates a semantic mismatch between what is measured (node reliability) and what is modified (edge structure).

**Implication**: Need explicit edge reliability modeling or pairwise calibration features.

**Addressed by**: None yet (potential future work)

---

### G3: Candidate Recall Still Depends on Embedding Similarity
**Status**: `unresolved`  
**Severity**: Low  
**Evidence**: GNN-Ret distinguishes recall from decision stages

**Problem**: Final decision uses calibration, but recall stage still uses similarity. This creates a bottleneck where good anchors might be missed if they're not similar in embedding space.

**Implication**: Could use calibration signals to train retrieval or use prototype anchors.

**Addressed by**: None yet (potential future work)

---

### G4: Explanation Enrichment Quality is Infrastructure-Level
**Status**: `partially_addressed`  
**Severity**: Low  
**Evidence**: Current implementation is templated + hashed

**Problem**: Current explanation cache is templated + hashed, not LLM-generated. This limits the quality of explanation-based enrichment.

**Implication**: Should describe as first implementation, not complete module. Genuine LLM rationales needed for full Harnessing Explanations (ICLR'24) alignment.

**Addressed by**: Current implementation (partial)

---

### G5: No Explicit "When NOT to Use Graph" Mechanism
**Status**: `partially_addressed`  
**Severity**: High  
**Evidence**: Text-only branch outperforms full model (F1=0.8115 vs 0.7945)

**Problem**: Current method edits graph but doesn't decide whether to use it at all. Trustworthy AI literature emphasizes selective usage over forced fusion.

**Implication**: Need abstention mechanism, risk-coverage tradeoff, or slice-specific policies.

**Addressed by**: 
- claim:C4: disagreement trigger identifies 14.2% of nodes where graph is unreliable
- rw4_disagreement_local: repairs graph on triggered nodes instead of blindly using it
- Confidence fallback (exp:confidence_fallback): simple tau-based switching validated
- exp:failure_regime_diagnostics_v2: matched-budget comparison shows entropy+disagreement combo (AUROC 0.820) is best risk scoring; disagreement alone is high-precision (top-5% prec 49.8%) but low-coverage (AUROC 0.576)

---

## Resolved Gaps

*No resolved gaps yet*

---

## Gap ID Assignment Rules

- Gap IDs are stable and never reused: G1, G2, G3, ...
- Status: `unresolved` | `partially_addressed` | `resolved`
- Each gap links to papers that address it and ideas that target it
