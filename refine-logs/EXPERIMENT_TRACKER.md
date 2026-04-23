# Experiment Tracker: FRMI Full Paper v2

**Created**: 2026-04-20
**Method**: Failure-Regime-Conditioned Minimal Intervention (two-layer probe + three-action policy)
**Primary dataset**: TwiBot-22 (official split)
**Supersedes**: Embedding-dominant tracker (2026-04-19)

## Status

| Run ID | Milestone | Block | Purpose | System / Variant | Priority | Status | Notes |
|--------|-----------|-------|---------|------------------|----------|--------|-------|
| R001 | M0 | A0 | Protocol freeze | — | MUST | DONE | PROTOCOL_FREEZE.md written |
| R002 | M0 | A1 | GATv2 implementation | GATv2Bot in GNNs.py | MUST | DONE | 2-layer GATv2Conv + model_building dispatch |
| R003 | M0 | A1 | TwiBot-22 pipeline | dataloader adaptation | MUST | DONE | Pipeline is dataset-agnostic; just needs data files |
| R004 | M0 | A1 | GATv2 training | 3 seeds, official split | MUST | DONE | GATv2 F1=0.70 → RGCN fallback (F1≈0.87) |
| R005 | M0 | A1 | OOF failure labels | k-fold within train only | MUST | TODO | For probe training |
| R006 | M1 | C | Feature extraction | probe features (7 signals) | MUST | DONE | frmi_v2_probe.py implemented |
| R007 | M1 | C | Probe training | 3 variants comparison | MUST | DONE | frmi_v2_probe.py: LogisticRegression |
| R008 | M1 | C | Risk-capture curves | utility@budget, HCW capture | MUST | DONE | Metrics in frmi_v2_probe.py |
| R009 | M2 | D | Semantic residual op | focal node LM residual | MUST | DONE | frmi_v2_operators.py |
| R010 | M2 | D | Edge reweight op | ego-subgraph mask/reweight | MUST | DONE | frmi_v2_operators.py |
| R011 | M2 | D | Policy dispatcher | probe → regime → action | MUST | DONE | frmi_v2_operators.py |
| R012 | M2 | D | 8-condition comparison | D1-D8, 3 seeds each | MUST | RUNNING | Seed 1: D3 best (+0.0018), D7 underperforms (semantic op needs training) |
| R012b | M2 | D | D7 with ContrastiveResidualAdapter | Replace Ridge with contrastive adapter | MUST | TODO | Idea discovery A2 recommendation |
| R013 | M2 | D | Action swap table | regime-conditional matched-node | MUST | TODO | Paired bootstrap |
| R014 | M3 | B | Tier-2 mechanism proxies | all-node semantic, GNNGuard, kNN | MUST | TODO | ~4h remote |
| R015 | M3 | B | Tier-1 faithful baselines | LGB/BotBR/BotGSL (if code avail) | MUST | TODO | Or cite published |
| R016 | M4 | E | BotRGCN confirmatory | Blocks C+D on BotRGCN | SHOULD | TODO | Directional consistency |
| R017 | M4 | E | TwiBot-20 cross-dataset | Full pipeline on TwiBot-20 | NICE | TODO | Existing data |
| R018 | M4 | E | C&S orthogonality | C&S on top of D7 | NICE | TODO | Appendix only |

## Decision Log

| Date | Decision | Reason |
|------|----------|--------|
| 2026-04-15 | C1 KILLED: composite d_i | AUROC 0.692 < entropy 0.719 |
| 2026-04-17 | Graph rewrite not viable | aux_50 gain +0.003 within noise ±0.011 |
| 2026-04-19 | FRMI on RGT abandoned | Built on wrong base (RGT F1≈0.76 vs baselines≈0.87) |
| 2026-04-19 | Embedding-dominant + override direction | Phase 5 strongest result |
| 2026-04-20 | **Full paper pivot: FRMI v2** | Regime-conditioned minimal intervention with GATv2+TwiBot-22 |
| 2026-04-20 | OOF internal only | Avoid protocol leakage; main results use official split |
| 2026-04-20 | Two-layer probe design | Risk score + regime evidence; regime labels stay rule-frozen |
| 2026-04-20 | 8 conditions (added Binary Policy) | Must beat GLANCE-style binary router |
| 2026-04-20 | Paired bootstrap for stats | Node-level intervention needs node-level power |
| 2026-04-20 | **TwiBot-20 first** | Validate full pipeline on existing data before TwiBot-22 |
| 2026-04-22 | **Ridge semantic op identified as root cause** | D7 underperforms D3; Ridge reproduces GNN errors instead of correcting |
| 2026-04-22 | **ContrastiveResidualAdapter recommended** | Idea discovery A2: contrastive adapter trained on OOF failures with class prototypes |
| 2026-04-22 | R012b added | Re-run D7 with new semantic operator before proceeding |
| 2026-04-22 | **Expert review corrected novelty** | A2 downgraded 7→3/10. B1 promoted to headline. Claim stability 4/10. |
| 2026-04-22 | **Experiment resequenced: S0→S1→S2→S3** | S0 operator tribunal before any system-level run. Three-layered kill gates. |
| 2026-04-22 | **R012b demoted** | CONDITIONAL on S0 pass. Not default next step. |
| 2026-04-22 | **PAPER_CHARTER.md created** | Allowed/forbidden claims, required proof, baselines, terminology frozen. |

## Prior Phase Results (TwiBot-20, RGCN backbone — informational)

| Phase | Key Finding |
|-------|-------------|
| Phase 1 | prop_corruption 1.84x error enrichment; entropy+disagree AUROC=0.820 |
| Phase 2 | C&S +1.18% acc; GNNGuard blocked (RGCN incompat) |
| Phase 3 | Graph rewrite marginal (+0.003 F1, within noise) |
| Phase 4 | Semantic override +8.3pp target acc; repair +2.7pp on prop_corruption |
| Phase 5 | Embedding effect (+0.0072) >> graph effect (±0.003) |
