# NARRATIVE_REPORT

## Live source-of-truth status

- This file is the primary narrative input for paper writing.
- The active paper-writing markdown set is:
  - `NARRATIVE_REPORT.md`
  - `PAPER_PLAN.md`
  - `claims_evidence_matrix.md`
- Current compiled-paper identity:
  `conformal-routed selective residual detection of social bots`.
- The old router-only / intended-consumer narrative is retired.

## 1. Paper identity

- Working title:
  `Conformal-Routed Selective Residual Detection of Social Bots`
- Venue: `NLPCC 2026`
- Format: `Springer LNCS / LNAI`
- Review mode: `double-blind`
- Page limit: `12 pages total, including references`

## 2. Main thesis

Strong frozen social bot detectors are already reliable for most nodes, but their
remaining errors concentrate on a small hard subset. Higher-order semantic KNN
evidence is useful for these residual failures, but unsafe when propagated to
all nodes. The paper therefore proposes a selective residual detector:

1. a frozen low-order detector supplies the default prediction,
2. a support-aligned conformal-style router selects high-risk hard nodes, and
3. a high-order residual branch corrects only the routed subset while
   non-routed nodes bypass the branch unchanged.

## 3. Problem statement

The research problem is no longer only hard-node localization. It is selective
residual correction under a frozen detector. The router is necessary because
the correction branch should be activated where the detector is likely to fail,
not where the base prediction is already reliable.

## 4. Completed evidence used in the manuscript

### 4.1 Frozen base detector context

Source:
`docs/research/baselines/frozen_simteg_fullgraph_20260614/README.md`

The full detector-context leaderboard table is retired from the main manuscript
because it distracts from the selective-residual thesis. Use the frozen SimTeG
summary only to justify that the base detector is a strong starting point with
remaining residual errors.

Three-seed TwiBot-20 baseline summary:

| Seed | Accuracy | Macro-F1 | Human-F1 | Bot-F1 | Wrong |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.8605 | 0.8591 | 0.8451 | 0.8732 | 165 |
| 2 | 0.8605 | 0.8581 | 0.8397 | 0.8766 | 165 |
| 3 | 0.8681 | 0.8666 | 0.8523 | 0.8809 | 156 |
| Mean | 0.8631 | 0.8613 | 0.8457 | 0.8769 | 162.0 |

### 4.2 Router quality

Source:
`LLMbot/experiments/knn_router_selectedk_compare_20260611/summary_means.csv`

| Router | AUROC-error | AUPRC-error | AURC |
| --- | ---: | ---: | ---: |
| independent_ncp_local_conformal | 0.7097 | 0.3115 | 0.0787 |
| samehyperedge_global_tail | 0.8355 | 0.3727 | 0.0361 |
| samehyperedge_selected_tail | 0.5070 | 0.1384 | 0.1318 |

Safe takeaway:
`samehyperedge_global_tail` is the strongest aligned router and supports the
claim that routing should use the same support prior as residual correction.

### 4.3 Selective residual detector

Source:
`LLMbot/experiments/local_twibot20_routed_residual_20260618_reports/summary_seed1.csv`

Scope:

- completed seed-1 local detector study;
- HyperScan-style sampled-subgraph contract;
- queue manifest records proxy metadata rather than the server official tensor;
- use as detector evidence for the selective residual mechanism, not as a
  three-seed benchmark replacement.

Main result rows:

| Run | Full Acc. | Full Macro-F1 | Routed-10 Macro-F1 | Non-routed-10 Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Low-only baseline | 0.8766 | 0.8741 | 0.7340 | 0.8834 |
| All-node residual | 0.8757 | 0.8732 | 0.7340 | 0.8824 |
| Routed residual, budget 10%, fanout 64 | 0.8715 | 0.8684 | 0.7627 | 0.8753 |
| Risk-gated all-node residual | 0.8783 | 0.8761 | 0.7485 | 0.8845 |
| Shuffled routed residual, budget 10% | 0.8757 | 0.8745 | 0.7771 | 0.8813 |
| Routed residual, budget 10%, fanout 128 | 0.8833 | 0.8817 | 0.7943 | 0.8875 |

Safe takeaway:
selective high-order residual correction is implemented and evaluated. The
best available configuration improves full-test performance and the routed
hard-node slice while preserving the non-routed complement.

## 5. Writing boundaries

- Do not describe the residual branch as future work or an intended consumer.
- Do not say the paper only validates routing.
- Do not hide that the residual detector table is seed-1/local exploratory
  evidence under a documented sampled-subgraph contract.
- Do not claim the residual table is a full three-seed benchmark replacement
  unless a multi-seed residual summary is added.
- Keep the main contribution centered on selective residual correction, with
  router quality as the mechanism that makes correction targeted.
