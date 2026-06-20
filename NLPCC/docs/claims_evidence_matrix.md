# Claims-Evidence Matrix

## Current mainline

`conformal-routed selective residual detection of social bots`

## Claim 1

`Frozen social bot detectors leave a residual hard-node regime in which global detector replacement is less appropriate than selective correction.`

Evidence:

- three-seed frozen SimTeG baseline on TwiBot-20;
- residual errors remain concentrated enough for routed intervention.

Support status:

- supported for manuscript framing.

## Claim 2

`A support-aligned conformal-style router localizes residual errors better than independent or overly localized alternatives.`

Evidence:

- `LLMbot/experiments/knn_router_selectedk_compare_20260611/summary_means.csv`;
- samehyperedge global-tail improves AUROC-error from 0.7097 to 0.8355 and AURC from 0.0787 to 0.0361 against independent local conformal.

Support status:

- directly supported by completed three-seed router results.

## Claim 3

`Selective high-order residual correction is an evaluated method component, not a future-work consumer.`

Evidence:

- `LLMbot/experiments/local_twibot20_routed_residual_20260618_reports/summary_seed1.csv`;
- best routed residual configuration reaches 0.8833 accuracy and 0.8817 macro-F1;
- routed top-10% macro-F1 improves from 0.7340 under low-only baseline to 0.7943 under high-fanout routed residual.

Support status:

- supported as completed seed-1/local detector evidence under the documented sampled-subgraph contract.
- not yet a three-seed benchmark replacement unless a multi-seed residual summary is added.

## Claim 4

`Higher-order semantic neighborhoods are most useful as routed residual evidence rather than unrestricted all-node propagation.`

Evidence:

- all-node residual does not improve over low-only baseline in the seed-1 residual table;
- routed high-fanout residual improves full-test and routed-slice metrics;
- router quality confirms that the selected nodes are meaningful residual targets.

Support status:

- supported by the combination of router and selective residual evidence, with seed-1 scope for residual detector performance.
