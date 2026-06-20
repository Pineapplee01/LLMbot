# PAPER_PLAN

## Live source-of-truth status

- This file is the active section, figure, and table contract for the NLPCC paper draft.
- Current compiled-paper identity:
  `conformal-routed selective residual detection of social bots`.
- The old router-only plan is retired.

## Venue

- Target venue: `NLPCC 2026`
- Format: `Springer LNCS`
- Review mode: `double-blind`
- Page limit: `12 pages`, including references

## Working Title

`Conformal-Routed Selective Residual Detection of Social Bots`

## Main Thesis

Higher-order semantic neighborhoods should be used as routed residual evidence,
not as an unrestricted all-node propagation channel. A frozen detector provides
the default prediction; a conformal-style support-aligned router selects the
hard residual subset; a high-order residual branch corrects only routed nodes.

## Section Plan

1. Introduction
2. Related Work
3. Problem Setup and Residual Regime
4. Methodology
5. Experiments
6. Conclusion

## Figure Plan

1. `Framework overview`
   - editable source: `figures/Architecture.vsdx`
   - compiled file: `figures/Architecture_fit.pdf`
   - role: main method figure
   - content:
     - feature encoder over textual/user/social inputs
     - low-order relation learner
     - high-order hypergraph learner
     - risk aggregate
     - residual fusion
     - classifier
   - constraint: do not replace this figure with generated framework variants unless the Visio source is explicitly updated.

## Table Plan

1. `Detector context`
   - role: answer RQ1 and establish the frozen detector as a credible default decision rule
   - content: sampled TwiBot-22 context and TwiBot-20 benchmark context
   - caution: not a cross-dataset leaderboard and not the main contribution

2. `Selective residual detector results`
   - source: `LLMbot/experiments/local_twibot20_routed_residual_20260618_reports/summary_seed1.csv`
   - rows: low-only, all-node residual, routed-only budgets, risk-gated all-node, shuffled routed, high-fanout routed
   - metrics: full Acc, full Macro-F1, routed-10 Macro-F1, non-routed-10 Macro-F1
   - table note must state seed-1/local sampled-subgraph evidence scope

3. `Router quality`
   - source: `LLMbot/experiments/knn_router_selectedk_compare_20260611/summary_means.csv`
   - metrics: AUROC-error, AUPRC-error, AURC
   - purpose: show router is an effective hard-node localization mechanism

4. `Residual sensitivity`
   - source: same seed-1 residual summary
   - rows: k=4, k=16, fanout32, fanout128

## Writing constraints

- Present the residual branch as an evaluated method component.
- Do not use "intended downstream consumer" or "left to future work" for the residual correction branch.
- Keep "conformal-style" precise: the router is a weighted local tail ranking heuristic and does not claim coverage.
- Bind every numerical claim to the listed evidence files.
- Treat seed-1 residual results as strong mechanism evidence with explicit scope, not as a full multi-seed replacement.
- Structure experiments by RQ: frozen detector context, selective residual correction, router quality, residual sensitivity, and evidence boundary.
