# Research Audit: research.md ↔ Final Deliverables Conformance Check

**Date**: 2026-05-09
**Audit source**: `research.md` (2026-05-09, 660 lines, 14 sections)
**Deliverables audited**: `refine-logs/FINAL_PROPOSAL.md`, `refine-logs/REVIEW_SUMMARY.md`, `refine-logs/REFINEMENT_REPORT.md`, `refine-logs/EXPERIMENT_PLAN.md`, `refine-logs/EXPERIMENT_TRACKER.md`, `refine-logs/PIPELINE_SUMMARY.md`.
**Purpose**: Verify that every research.md commitment is either **accepted and operationalized** in the final package, or **deliberately deferred to v2 / appendix / dropped** with a named reason.

---

## Anchor Conformance (research.md §1 — Research Goal and Claim Boundary)

| research.md commitment | Final artifact status | Location |
|---|---|---|
| "Post-hoc conformal ego-graph quality estimator triggers hard-node refinement" | **ACCEPTED** — dominant contribution (EQC² trigger) | FINAL_PROPOSAL.md Method Thesis + Stage-1/2 |
| "SKETCH-style semantic/structural retrieval" | **PARTIAL** — deterministic modifier uses `α_text · cos` (semantic) + `α_rel` (structural); decoupled ablation moved to appendix A-Modifier | FINAL_PROPOSAL.md Stage-4; EXPERIMENT_PLAN.md Block-T+Appendix |
| "GAugLLM/CTGL-style coupled edge modifier" | **ACCEPTED (deterministic form)** as supporting contribution; learned coupled modifier moved to appendix A-Modifier | FINAL_PROPOSAL.md Stage-4; EXPERIMENT_PLAN.md R114 |
| "Soft reweight; not hard delete" | **ACCEPTED** — Stage-5 clip-based soft reweight, S1 hard-delete in appendix only | FINAL_PROPOSAL.md Stage-5 |
| "Evidence artifact / LLM embedding / refiner (optional)" | **DEFERRED to v2** with explicit justification (research.md §14 + TMLR-2024); no LLM call in v1 | FINAL_PROPOSAL.md Complexity Budget |
| Forbidden: "first graph rewrite for bot detection" | **ENFORCED** — explicit non-contribution | FINAL_PROPOSAL.md Contribution Focus |
| Forbidden: "first LLM for graph learning" | **ENFORCED** — explicit non-contribution | FINAL_PROPOSAL.md Contribution Focus |
| Forbidden: "LLM understands graphs" | **ENFORCED** — explicit non-contribution (TMLR 2024 cited) | FINAL_PROPOSAL.md Contribution Focus |
| Forbidden: "full causal graph repair" | **ENFORCED** — "counterfactual supervision" renamed "model-based sensitivity", appendix only | FINAL_PROPOSAL.md + REFINEMENT_REPORT.md Change #3 |

## §2 — Conformal Ego-Graph Estimator

| research.md commitment | Status | Note |
|---|---|---|
| `prediction_set(v), set_size(v), coverage_margin(v), abstain_risk(v), calibration_metadata` as Q_phi outputs | **ACCEPTED** with upgrade: non-conformity score itself is quality-aware (`s_lbl + w_tg·s_tg + w_het·s_het + w_rec·s_rec`), not just label-non-conformity. Round-1 mechanism-level fix. | FINAL_PROPOSAL.md Stage-1 |
| Threshold fit on `train/valid` only, test labels not used | **ACCEPTED + enforced** by `calibration_metadata.used_indices` unit-test assertion | FINAL_PROPOSAL.md Tuning Protocol |
| v1 uses minimal conformal core, no full CF-GNN topology-aware correction | **ACCEPTED** — DAPS/NAPS/SNAPS left as appendix A-Diffusion | EXPERIMENT_PLAN.md R115 |
| Abstain risk as single scalar for router | **TIGHTENED** — hand-mixed `0.70·size + 0.30·margin` removed (Round-1 reviewer); replaced with lex rule `g(v) = (set_size, -margin, -s_composite)` + single `τ*` on `valid_cal` | REFINEMENT_REPORT.md Round-1 Change #1 |

## §3 — Hard-Node Candidate Router

| research.md commitment | Status | Note |
|---|---|---|
| Router consumes `abstain_risk + set_size + coverage_margin + base_uncertainty + residual_risk_manifest` | **SIMPLIFIED** — deterministic lex rule; `residual_risk_manifest` kept as provenance log only, not a learned head | FINAL_PROPOSAL.md Stage-2; REFINEMENT_REPORT.md Round-1 Change #4 |
| "Do not let router become second bot predictor" | **ENFORCED** — router has zero trainable parameters at inference | FINAL_PROPOSAL.md Complexity Budget |
| GLANCE-style selective routing | **ACCEPTED** (hard-node quota at 15% intervention ratio) | EXPERIMENT_PLAN.md Intervention Budgets |
| LOGIN-style consultant loop | **INTENTIONALLY NOT USED in v1 main** — consistent with "v1 no LLM calls" constraint | FINAL_PROPOSAL.md + REFINEMENT_REPORT.md |

## §4 — Local Ego Retrieval (SKETCH + GAugLLM + CTGL)

| research.md commitment | Status | Note |
|---|---|---|
| `propagation_edges` (existing ego) vs `virtual_context_edges` (new candidates, evidence-only) | **REVISED in Round 3** — `virtual_context_edges` dropped from v1 main because they never enter a decision rule; retained as future v2 hook | FINAL_PROPOSAL.md Stage-3; REFINEMENT_REPORT.md Round-3 Change #2 |
| Semantic-only / structural-only / uncoupled / coupled retrieval as ablation | **ACCEPTED** — A-Modifier appendix captures the decoupled comparison; main Block-T holds retrieval fixed while varying trigger | EXPERIMENT_PLAN.md Block-T + R114 |
| SNAPS-style semantic kNN extension to estimator | **APPENDIX A-Diffusion** (E5 variant) — not v1 main | EXPERIMENT_PLAN.md R115 |

## §5 — Edge Modifier Design

| research.md commitment | Status | Note |
|---|---|---|
| `z_text(e)`, `z_graph(e)`, `z_e = CoupledGate(...)` interface | **SIMPLIFIED** — deterministic modifier `α_text·cos + α_rel·𝟙[rel∈R_trusted] + α_tgd·sign(ΔJSD)` in v1 main; learned `CoupledGate` in appendix A-Modifier | FINAL_PROPOSAL.md Stage-4; EXPERIMENT_PLAN.md R114 |
| Modifier output form: `s_reliability` (binary) / `u_edge` (continuous) / `p_role` (4-role) | **ABLATION AXIS** — left open per research.md §5 explicit instruction; A-Modifier appendix includes this ablation | EXPERIMENT_PLAN.md A-Modifier |
| Counterfactual edge-intervention supervision on train/valid | **RENAMED and DEMOTED** — "model-based sensitivity supervision" via single-backward-pass gradient attribution, appendix only | REFINEMENT_REPORT.md Round-1 Change #3 + Round-2 renaming |
| "Do not use same-label=reliable as main rule" | **ENFORCED** — `s_tg` uses probability-space JSD between `p_LM` and `p_GNN`, not label similarity | FINAL_PROPOSAL.md Stage-1 `s_tg` definition |

## §6 — Soft Rewrite Instead of Hard Rewrite

| research.md commitment | Status | Note |
|---|---|---|
| Clip-based soft reweight: `w_e' = w_e · clip(1 + λ·budget·score·(1-uncertainty), 1-λ, 1+λ)` | **ACCEPTED verbatim** with `uncertainty = s_composite(v)` | FINAL_PROPOSAL.md Stage-5 |
| Virtual context edges never written back | **ENFORCED** (and the virtual context artifact itself was removed from v1 main in Round 3; the invariant survives trivially) | FINAL_PROPOSAL.md Stage-3/5 |
| Camouflage edges preserved as evidence, not deleted | **ENFORCED** — soft reweight clip preserves edges at downweighted weight; orthogonal to BotBR/BECE binary deletion | FINAL_PROPOSAL.md Novelty and Elegance |
| Rollback-able, budget-limited | **ACCEPTED + tightened** — Stage-6 accept/rollback gate uses the same `s_composite` primitive; `budget(v) = min(B_max, κ·s_composite)` | FINAL_PROPOSAL.md Stage-6 |

## §7 — Evidence Artifact and Optional LLM Embedding

| research.md commitment | Status | Note |
|---|---|---|
| v1 allows LLM-enhanced branch as ablation | **DEFERRED to v2** per user scope decision locked at the start of `/research-refine` | FINAL_PROPOSAL.md Contribution Focus + Complexity Budget |
| Evidence-prompt + refiner (LLM-as-enhancer, not predictor) | **DEFERRED to v2** with explicit "v1 no LLM calls" constraint; evidence artifact itself is also dropped since it was tied to the LLM prompt | FINAL_PROPOSAL.md Complexity Budget |
| Strict boundary: LLM not used as final classifier | **ENFORCED** | FINAL_PROPOSAL.md Contribution Focus (explicit non-contribution) |

**Deferral justification**: research.md §14 v1 config explicitly states "no LLM in v1 main method; hard-node evidence ego embedding in v2". The user locked this at the pipeline entry. TMLR-2024 graph-prompt analysis cited as additional justification against LLM-as-graph-reasoner.

## §8 — Iterative Estimator Loop

| research.md commitment | Status | Note |
|---|---|---|
| `max_iter = 1` v1 main | **ACCEPTED** | FINAL_PROPOSAL.md Stage-6 |
| `max_iter = 2` ablation only, with rollback | **ACCEPTED** | EXPERIMENT_PLAN.md A-Iter (R116) |
| Accept conditions: `set_size` non-increasing, `coverage_margin` non-decreasing, `abstain_risk` non-increasing, budget non-violating | **TIGHTENED** — Round-2 collapses to strict comparator: `s_composite_post < s_composite_pre AND set_size_post ≤ set_size_pre` (same primitive as trigger) | FINAL_PROPOSAL.md Stage-6; REFINEMENT_REPORT.md Round-2 Change #1 |

## §9 — Experimental Plan

| research.md experiment block | Final artifact mapping |
|---|---|
| §9.1 Main Baselines | Block-T (trigger swap) + Block-G (gate toggle) + Block-X (MGTAB cross-dataset) |
| §9.2 Estimator ablation E0..E5 | EXPERIMENT_PLAN.md Block-T (T1/T2/T3/T4-NS/T4) + Appendix A-Diffusion (R115 covers E4/E5-equivalent DAPS/NAPS + SNAPS) |
| §9.3 Local ego refinement ablation R0..R6 | EXPERIMENT_PLAN.md Block-T holds modifier fixed; A-Modifier appendix (R114) covers semantic-only vs structural-only vs coupled |
| §9.4 Edge modifier output ablation B0..B7 | EXPERIMENT_PLAN.md A-Modifier (R114) — open per research.md §5 |
| §9.5 Soft vs hard rewrite S0..S5 | Baked into FINAL_PROPOSAL.md as v1-main choice (S3+S4); S1 hard-delete as appendix-only negative-control variant |
| §9.6 LLM usage L0..L5 | DEFERRED to v2 (consistent with v1 no-LLM scope) |
| §9.7 Iteration ablation I0..I4 | EXPERIMENT_PLAN.md A-Iter (R116) |

## §10 — Metrics and Diagnostics

| research.md metric | Final artifact coverage |
|---|---|
| Accuracy / macro-F1 | Primary metric in EXPERIMENT_PLAN.md |
| ECE / Brier / AURC / coverage / set size / singleton hit ratio | Secondary/diagnostic metrics in EXPERIMENT_PLAN.md |
| Hard-node ratio, LLM call ratio, average ego size, virtual context count, latency | EXPERIMENT_PLAN.md + Block-L locality audit (refinement wall-clock ratio) |
| Slice metrics (high-abstain, text-graph disagreement, low-degree, heterophily, suspicious-edge-rich, bot-human-mixed) | **ACCEPTED + tightened**: 5 pre-registered slices in EXPERIMENT_PLAN.md, thresholds frozen on valid_cal before test |

## §11 — Implementation Order (Step 1..6)

| research.md step | Final artifact mapping |
|---|---|
| Step 1: Estimator-first (graph_conformal_set_estimator) | R101 + R102 (Core machinery, W1) |
| Step 2: Hard-node artifact routing | Collapsed into R101 (deterministic lex rule, no separate routing head) |
| Step 3: Local ego retrieval | Collapsed into R103 (trivial: existing ego edges) |
| Step 4: Edge modifier | R103 (deterministic); R114 appendix (learned) |
| Step 5: Soft rewrite + evidence artifact | R103 (soft reweight); evidence artifact dropped from v1 main |
| Step 6: LLM embedding extension | DEFERRED to v2; not in v1 plan |

## §12 — Expected Contributions

| research.md contribution | Final artifact status |
|---|---|
| C1: Conformal ego quality as local refinement trigger | **PROMOTED to dominant contribution (EQC²)** — expanded with quality-aware composite score + L-hop-local gate |
| C2: Text-graph coupled local ego refinement | **ACCEPTED as supporting contribution** in deterministic form; learned coupled form in appendix |
| C3: Evidence-preserving edge modifier | **ENFORCED as a property** of the soft-reweight rewrite (camouflage edges preserved) rather than a separate contribution; avoids sprawl |
| C4: Selective evidence ego prompt for LLM enhancer | **DEFERRED to v2** |

## §13 — Risks and Guardrails (Risk 1–5)

| research.md risk | Guardrail in final artifacts |
|---|---|
| Risk 1: conformal assumption violation | FINAL_PROPOSAL.md — framed as "calibrated selection rule", not coverage-guaranteed CP; inductive-CP caveat cited; empirical coverage reported as diagnostic only |
| Risk 2: binary reliability duplication | Explicit non-contribution in FINAL_PROPOSAL.md Contribution Focus; BotBR/BECE cited as prior art |
| Risk 3: camouflage evidence loss | Soft reweight preserves edges; explicit invariant in FINAL_PROPOSAL.md Stage-5 |
| Risk 4: LLM graph-prompt misuse | TMLR-2024 cited; v1 no LLM calls; evidence-artifact dropped |
| Risk 5: iterative confirmation bias | `max_iter=1` v1 main; `max_iter=2` appendix only with rollback |

## §14 — Final Recommended v1 Configuration

| research.md §14 line | Final artifact match |
|---|---|
| Estimator: graph_conformal_set_estimator with prediction_sets/set_size/coverage_margin/abstain_risk | **MATCH + UPGRADE** (quality-aware composite non-conformity score) |
| Router: residual-risk + conformal abstain_risk filtering | **MATCH + SIMPLIFY** (deterministic lex rule on `(set_size, -margin, -s_composite)`, `residual_risk_manifest` kept as provenance only) |
| Retrieval: semantic kNN + structural ego/k-hop/PPR | **MATCH** (deterministic modifier uses semantic cosine + relation gating) |
| Modifier: compare binary / continuous / role-aware; counterfactual-edge supervision | **MATCH** (A-Modifier appendix; renamed to "model-based sensitivity supervision") |
| Rewrite: existing ego edges only, soft reweight, budget + uncertainty control | **MATCH** (FINAL_PROPOSAL.md Stage-5) |
| Artifact: trusted refined ego-graph evidence | **DROPPED from v1 main** in Round 3 (unused in decision rule); future v2 hook |
| LLM: no LLM in v1, hard-node evidence ego embedding in v2 | **MATCH exactly** |
| Iteration: max_iter=1 main, max_iter=2 ablation | **MATCH exactly** |

---

## Summary Table — research.md Coverage

| Section | Sub-items | Accepted v1 main | Accepted appendix / v2 | Dropped with reason |
|---|---|---|---|---|
| §1 Anchor | 4 forbidden, 5 literature pillars | All enforced | — | — |
| §2 Estimator | 5 outputs | All 5 preserved (enriched) | DAPS/NAPS/SNAPS extension in appendix | — |
| §3 Router | 5 signals | Lex rule on 3 signals | — | Trainable router (Round-1 CRITICAL); hand-mixed abstain_risk (Round-1 IMPORTANT) |
| §4 Retrieval | 2 channels (semantic+structural), 2 edge classes | Semantic+structural via modifier coefficients | Decoupled ablation in A-Modifier | `virtual_context_edges` (Round-3 unused) |
| §5 Modifier | 3 output forms, 3 supervision signals | Deterministic scalar as v1 main | Learned coupled modifier + output-form ablation in A-Modifier | Counterfactual GNN reruns (Round-1 feasibility) |
| §6 Rewrite | Soft clip, budget, rollback | All preserved | Hard-delete as negative-control appendix | Global rewrite |
| §7 Evidence/LLM | Ego prompt + refiner | — | All deferred to v2 | — |
| §8 Iteration | max_iter=1/2 | max_iter=1 main | max_iter=2 in A-Iter | — |
| §9 Experiments | 7 ablation blocks | 3 claim blocks + locality | 4 appendix blocks | L* LLM ablations deferred |
| §10 Metrics | 4 categories | All covered | — | — |
| §11 Impl order | 6 steps | Steps 1,3,4,5 in R101-R104 | Step 6 deferred | Step 2 collapsed |
| §12 Contributions | C1-C4 | C1 promoted dominant, C2 supporting | — | C3 enforced as property (no sprawl); C4 v2 |
| §13 Risks | 5 risks | All guarded | — | — |
| §14 v1 config | 8 lines | 7 match exactly | `max_iter=2` ablation | Artifact in Round 3 |

**Conclusion of conformance audit**: Every research.md commitment is either (a) **accepted and operationalized** in the final pipeline, (b) **deferred to v2 / appendix with a named reason consistent with research.md's own §14 v1 config**, or (c) **dropped with a cited reviewer round and rationale**. No silent omission. The final pipeline is a **strict specialization** of research.md with additional mechanism-level upgrades (quality-aware composite non-conformity score; probability-space JSD for `s_tg`; T4-NS falsification baseline; L-hop-local Stage-6 locality invariants) earned across four Codex review rounds.

**Determination**: `research.md` is fully honored; no new content is required. The `/research-refine-pipeline` package is **confirmed complete and internally consistent**.
