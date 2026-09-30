# Frozen SimTeG Full-Graph Baseline

Date: 2026-06-14

Status: canonical baseline for the current KNN router and routed prompt
experiments.

## Purpose

This directory fixes the unified frozen SimTeG baseline used by the current
`KNN router -> prompt/refiner` branch. The goal is to stop mixing several
historical artifacts that all used similar names but different feature or
training contracts.

The canonical artifact family is:

```text
/root/workspace/LMbot/LLMbot_active_knn_router_20260612/experiments/tmp_iterm1_forward_fullgraph/seed_{1,2,3}/preparation/graph_detector/outputs.pt
```

All three seeds share:

- `contract=frozen_g0_v1`
- `backbone=rgcn`
- `graph_data_variant=full_graph_support`
- `training_scope=train_only_supervised`
- canonical TwiBot-20 train/valid/test split files
- frozen `embeddings_iter_-1_seed_{1,2,3}.pt` labeled-node embeddings
- full-graph support embeddings from `support_roberta_embeddings_new.pt`

## Metrics

Metrics were replayed from `outputs.pt` with the canonical TwiBot-20
`test_idx.pt`, rather than copied from logs or experiment names.

| Seed | Accuracy | Macro-F1 | Human-F1 | Bot-F1 | Wrong |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.8605 | 0.8591 | 0.8451 | 0.8732 | 165 |
| 2 | 0.8605 | 0.8581 | 0.8397 | 0.8766 | 165 |
| 3 | 0.8681 | 0.8666 | 0.8523 | 0.8809 | 156 |
| Mean | 0.8631 | 0.8613 | 0.8457 | 0.8769 | 162.0 |
| Sample std | 0.0044 | 0.0046 | 0.0063 | 0.0039 | 5.1962 |

Machine-readable values are stored in `metrics.json`.

## Comparison Policy

Use this baseline when evaluating:

- KNN conformal router variants
- router-selected routed-node prompt experiments
- GraphEdit-style edge adjudication over router-KNN support
- downstream refiner comparisons that consume the current router artifacts

Do not compare these experiments against historical `0.87` runs unless the
method is rerun on the same stronger feature/training contract. The audit found
that the highest historical runs used different conditions, especially:

- `finetuned_roberta_embeddings_iter_2_seed1.pt`
- `train_supervised_plus_mhlgc`
- labeled-only graph variants
- HyperScan-style dynamic detector ablations

Those artifacts may be useful as separate ablations or stronger-base studies,
but they are not the canonical frozen SimTeG baseline for the current
router/prompt pipeline.

## Current Prompt Context

The partial Qwen edge-usefulness sidecar experiment used seed 1 from this
baseline family:

```text
experiments/tmp_iterm1_forward_fullgraph/seed_1/preparation/graph_detector/outputs.pt
```

Its full-test replay is therefore a seed-1 comparison only. Future full prompt
experiments should either:

1. report seed-1 results explicitly as seed-1 only, or
2. run the same prompt/refiner protocol for seeds 1, 2, and 3 and report the
   aggregate against this directory's three-seed baseline.

## Evidence Files

- `manifest.json`: artifact paths, split hashes, model contract, and comparison
  policy.
- `metrics.json`: replayed seed metrics and aggregate statistics.

