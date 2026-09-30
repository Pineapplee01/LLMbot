# Local Conflict Diagnostic Record 2026-06-06

Status: exploratory diagnostic record

Scope:

- stage: `local_conflict_prune_diag`
- dataset: `TwiBot-20`
- semantic backbone: finetuned RoBERTa / SimTeG-style embedding cache
- graph backbone: `rgcn`
- evidence date: 2026-06-06

This note records what was actually run, what artifacts were produced, what the
completed labeled-graph diagnostic supports, what the incomplete full-graph
attempts do and do not support, and which field names in the current artifact
schema should be interpreted carefully.

## 1. Diagnostic Intent

The diagnostic work had two goals:

1. Evaluate the current `local_conflict_prune_diag` behavior on the labeled
   TwiBot-20 graph.
2. Extend the stage so that the same diagnostic family can later compare
   labeled-prefix behavior against the support-augmented full graph.

The second goal was implemented at the code level, but the full-graph stage did
not complete end-to-end within practical smoke-validation time.

## 2. Evidence Roots

### 2.1 Completed labeled-graph run

Primary artifact root:

- `LLMbot/validation_local_conflict_labeled_20260606/seed_1/stages/local_conflict_prune_diag`

Key files:

- `manifest.json`
- `metrics.json`
- `per_node_test.jsonl`
- `routed_nodes.jsonl`
- `selected_edges.jsonl`
- `conflict_evidence_edges.jsonl`
- `view_outputs.pt`
- rerun directories under `reruns/local_conflict` and `reruns/local_rand`

### 2.2 Full-graph stage attempts that did not finish

Attempted roots:

- `LLMbot/validation_local_conflict_fullgraph_20260606/seed_1`
- `LLMbot/validation_local_conflict_fullgraph_smoke_20260606/seed_1`

Observed state:

- both roots exist
- both contain `runtime/lm_only_head`
- both contain an empty or not-yet-flushed
  `stages/local_conflict_prune_diag` directory
- neither contains completed `manifest.json` or `metrics.json` for the stage

Interpretation:

- the full-graph stage started
- graph-wide semantic-head preparation completed
- the stage did not finish the main local-conflict diagnostic body before the
  timeout window

### 2.3 Full-graph component validation root

Component-validation root:

- `LLMbot/validation_component_fullgraph_20260606/seed_1/runtime/lm_only_head`

This root was used to verify the critical full-graph row-alignment fix for the
LM-only semantic head.

## 3. Commands Actually Run

### 3.1 Engineering surface validation

```bash
python -m py_compile G:\Research\BotDetection\LLMbot\trainer_graph.py G:\Research\BotDetection\LLMbot\main.py
python G:\Research\BotDetection\LLMbot\main.py --help
```

Outcome:

- passed

### 3.2 Completed labeled-graph diagnostic run

```bash
python G:\Research\BotDetection\LLMbot\main.py ^
  --experiment_task local_conflict_prune_diag ^
  --experiment_name validation_local_conflict_labeled_20260606 ^
  --dataset TwiBot-20 ^
  --use_GNN ^
  --graph_backbone rgcn ^
  --embedding_path G:\Research\BotDetection\datasets\TwiBot-20\embeddings_iter_-1_seed_1.pt ^
  --seeds 1 ^
  --disable_wandb ^
  --external_frozen_g0_root G:\Research\BotDetection\LLMbot\experiments\tmp_local_conflict_seed1\seed_1
```

Outcome:

- completed
- stage artifacts and rerun artifacts were written

### 3.3 Full-graph end-to-end attempt

```bash
python G:\Research\BotDetection\LLMbot\main.py ^
  --experiment_task local_conflict_prune_diag ^
  --experiment_name validation_local_conflict_fullgraph_20260606 ^
  --dataset TwiBot-20 ^
  --graph_data_variant full_graph_support ^
  --use_GNN ^
  --graph_backbone rgcn ^
  --embedding_path G:\Research\BotDetection\datasets\TwiBot-20\embeddings_iter_-1_seed_1.pt ^
  --support_embedding_path G:\Research\BotDetection\datasets\TwiBot-20\support_roberta_embeddings_new.pt ^
  --seeds 1 ^
  --disable_wandb ^
  --external_frozen_g0_root G:\Research\BotDetection\LLMbot\experiments\tmp_fullgraph_smoke\seed_1
```

Outcome:

- timed out

### 3.4 Lower-budget full-graph smoke

```bash
python G:\Research\BotDetection\LLMbot\main.py ^
  --experiment_task local_conflict_prune_diag ^
  --experiment_name validation_local_conflict_fullgraph_smoke_20260606 ^
  --dataset TwiBot-20 ^
  --graph_data_variant full_graph_support ^
  --use_GNN ^
  --graph_backbone rgcn ^
  --embedding_path G:\Research\BotDetection\datasets\TwiBot-20\embeddings_iter_-1_seed_1.pt ^
  --support_embedding_path G:\Research\BotDetection\datasets\TwiBot-20\support_roberta_embeddings_new.pt ^
  --conflict_router_budget 0.001 ^
  --seeds 1 ^
  --disable_wandb ^
  --external_frozen_g0_root G:\Research\BotDetection\LLMbot\experiments\tmp_fullgraph_smoke\seed_1
```

Outcome:

- still timed out

### 3.5 Full-graph row-alignment validation

The row-alignment validation confirmed that the full-graph LM-only head was
actually producing graph-wide rows:

- `prob_cal.shape == [229580, 2]`
- `pred.shape == [229580]`

The corresponding manifest at
`LLMbot/validation_component_fullgraph_20260606/seed_1/runtime/lm_only_head/manifest.json`
records:

- `graph_data_variant = full_graph_support`
- `graph_node_count = 229580`
- `labeled_node_count = 11826`
- `support_node_count = 217754`
- `output_rows = 229580`

This validates the key implementation change:

- full-graph semantic-head outputs are graph-wide
- supervision remains on the labeled prefix only

## 4. Completed Labeled-Graph Results

Source:

- `LLMbot/validation_local_conflict_labeled_20260606/seed_1/stages/local_conflict_prune_diag/metrics.json`

### 4.1 Main metrics

| Variant | Valid Acc | Valid Macro-F1 | Test Acc | Test Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Base frozen G0 | 0.8710 | 0.8680 | 0.8707 | 0.8693 |
| local_conflict | 0.8693 | 0.8665 | 0.8690 | 0.8677 |
| local_rand | 0.8693 | 0.8663 | 0.8724 | 0.8710 |

Delta vs base:

- `local_conflict` test accuracy: `-0.00169`
- `local_conflict` test macro-F1: `-0.00162`
- `local_rand` test accuracy: `+0.00169`
- `local_rand` test macro-F1: `+0.00171`

Immediate reading:

- the current conflict-driven local pruning did not improve over the base model
- the matched random baseline slightly outperformed the base model on test
- therefore, the current conflict score is not yet a reliable proxy for useful
  intervention

### 4.2 Routing summary

Routed-node statistics:

- valid routed target count: `236`
- routed counts:
  - train: `187`
  - valid: `236`
  - test: `143`
  - all labeled nodes: `566`

Conflict-score summary on test:

- mean: `0.1190`
- median: `0.0610`
- max: `0.6562`

### 4.3 Edge-selection summary

Candidate and selection statistics:

- candidate edge count: `1634`
- positive delta-conflict candidates: `536`
- positive delta-conflict ratio: `0.3280`
- selected conflict edges: `92`
- selected random edges: `92`
- degree-guard blocked candidates: `380`

Selected-edge relation/role distribution:

| Bucket | Count | Share |
| --- | ---: | ---: |
| `0::incoming` | 65 | 0.7065 |
| `1::incoming` | 18 | 0.1957 |
| `0::outgoing` | 5 | 0.0543 |
| `1::outgoing` | 4 | 0.0435 |

Immediate reading:

- the current stage overwhelmingly removes relation-0 incoming edges
- outgoing buckets contribute very little to the selected intervention set
- if this pattern is stable, the current intervention policy is much less
  direction-balanced than its bucket definition suggests

### 4.4 Test-side fix/break accounting

Derived from `per_node_test.jsonl`:

- test node count: `1183`
- base wrong test count: `153`
- `local_conflict` wrong test count: `155`
- `local_rand` wrong test count: `151`

Fix/break totals:

| Variant | Fix | Break | Net |
| --- | ---: | ---: | ---: |
| local_conflict | 6 | 8 | -2 |
| local_rand | 10 | 8 | +2 |

Important nuance:

- routed test count is `143`
- routed-node direct effects were nearly absent:
  - routed local-conflict fix: `0`
  - routed local-conflict break: `0`
  - routed local-rand fix: `0`
  - routed local-rand break: `1`

Non-routed side effects account for almost all observed changes:

- non-routed local-conflict fix: `6`
- non-routed local-conflict break: `8`
- non-routed local-rand fix: `10`
- non-routed local-rand break: `7`

Immediate reading:

- the observable metric movement is almost entirely propagation spillover onto
  non-routed test nodes
- the routed nodes themselves did not produce measurable direct fix gains in the
  completed labeled run
- this is a key negative finding for the current stage design

## 5. Failure-Mechanism Statistics From the Completed Labeled Run

The current stage writes four mechanism labels into `per_node_test.jsonl`.
For the completed labeled-graph run, these are operational buckets over
base-vs-local-conflict transitions on the labeled graph only.

### 5.1 Mechanism counts

| Failure mechanism | Count | Share | Base wrong rate | local_conflict wrong rate |
| --- | ---: | ---: | ---: | ---: |
| conflict_dominant | 1022 | 0.8639 | 0.0744 | 0.0734 |
| dense_directional_recoverable | 67 | 0.0566 | 1.0000 | 0.9254 |
| sparse_isolated | 86 | 0.0727 | 0.1163 | 0.1163 |
| support_induced_fragile | 8 | 0.0068 | 0.0000 | 1.0000 |

Important caveat:

- in this labeled-graph run, `support_induced_fragile` is a schema carry-over
  name, not literal support-neighbor evidence
- there are no support nodes in this run
- the bucket is populated by base-correct to local-conflict-wrong transitions
  under the unified artifact schema

### 5.2 Mechanism-level fix/break counts

| Mechanism | Routed count | Base wrong | local_conf fix | local_conf break | local_rand fix | local_rand break |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| conflict_dominant | 120 | 76 | 1 | 0 | 4 | 6 |
| dense_directional_recoverable | 8 | 67 | 5 | 0 | 6 | 0 |
| sparse_isolated | 15 | 10 | 0 | 0 | 0 | 0 |
| support_induced_fragile | 0 | 0 | 0 | 8 | 0 | 2 |

Immediate reading:

- nearly all local-conflict fixes come from `dense_directional_recoverable`
- `sparse_isolated` receives routed traffic but yields no gains
- `conflict_dominant` dominates the routed pool (`120 / 143`) but gives only
  `1` fix and no direct routed-node gains
- the base-correct to local-conflict-wrong failures are concentrated in the
  placeholder `support_induced_fragile` bucket

### 5.3 Routed-pool composition

Derived from `per_node_test.jsonl`:

| Mechanism | Routed count | Share of routed pool | Routed share within mechanism | Mean conflict score (all) | Mean conflict score (routed) |
| --- | ---: | ---: | ---: | ---: | ---: |
| conflict_dominant | 120 | 0.8392 | 0.1174 | 0.1171 | 0.4577 |
| dense_directional_recoverable | 8 | 0.0559 | 0.1194 | 0.1193 | 0.4041 |
| sparse_isolated | 15 | 0.1049 | 0.1744 | 0.1393 | 0.3605 |
| support_induced_fragile | 0 | 0.0000 | 0.0000 | 0.1420 | 0.0000 |

Immediate reading:

- the router spends most of its budget on `conflict_dominant`
- the small `dense_directional_recoverable` group is exactly the group with the
  clearest fix signal, but it receives only a small share of routed test nodes
- `sparse_isolated` has the highest routed share within its own bucket, but
  produced no benefit

## 6. Error-Migration And Regime-Shift Fields: Correct Interpretation

The current unified schema writes `error_migration` and `error_regime_shift`
fields even for labeled-only runs.

For the completed labeled run:

- `graph_data_variant = labeled`
- `support_exposure = 0`
- `support_consistency = 0`
- `structural_shock = 0`

Therefore:

- `error_migration` does not mean labeled-graph to support-augmented full-graph
  migration in this run
- it should be read as base prediction to local-conflict prediction migration
  under the current field names

Recorded migration counts:

- base wrong to local-conflict correct: `90`
- base correct to local-conflict wrong: `51`
- base wrong stays wrong: `1337`
- base correct stays correct: `10348`

The corresponding `error_regime_shift` table should also be interpreted as a
base-vs-local-conflict regime-change table under the labeled graph, not as a
support-graph regime-shift table.

This naming issue should be remembered in later analysis so that the labeled
diagnostic is not over-interpreted as full-graph support evidence.

## 7. Completed Labeled-Graph Conclusions

These conclusions are directly supported by the completed labeled-graph run:

1. The current conflict-driven local pruning does not improve over the base
   frozen G0 model on test.
2. The matched local random baseline slightly improves over the base model on
   test, which weakens the claim that the current conflict score is aligned with
   useful intervention.
3. Most routed nodes come from `conflict_dominant`, but that bucket produces
   almost no realized gain.
4. The small `dense_directional_recoverable` bucket is the only bucket with a
   clear positive fix signal.
5. `sparse_isolated` receives some routed budget but yields no benefit.
6. Nearly all measurable metric change comes from propagation side effects on
   non-routed nodes, not direct routed-node fixes.

## 8. Full-Graph Validation Status

### 8.1 What is validated

The following full-graph implementation facts are validated:

1. `graph_data_variant=full_graph_support` now passes the stage gate for
   `local_conflict_prune_diag`.
2. The LM-only semantic head now writes graph-wide outputs under
   `full_graph_support`.
3. The graph-wide semantic head uses:
   - `graph_node_count = 229580`
   - `labeled_node_count = 11826`
   - `support_node_count = 217754`
4. The output row count matches the full graph:
   - `output_rows = 229580`
   - `prob_cal.shape == [229580, 2]`
   - `pred.shape == [229580]`

### 8.2 What is not validated

The following are not yet validated by completed stage artifacts:

1. full-graph `metrics.json`
2. full-graph `manifest.json`
3. full-graph `per_node_test.jsonl`
4. any real non-zero support-exposure, support-consistency, or structural-shock
   statistics
5. any real evidence for literal support-induced fragility

### 8.3 Most likely bottleneck

The full-graph stage appears to be computationally heavy because the current
diagnostic tests many single-edge counterfactual deletions on the
support-augmented graph.

Observed evidence:

- both full-graph stage attempts reached graph-wide semantic-head preparation
- neither finished the stage write-out
- lowering `conflict_router_budget` from the default to `0.001` was not enough
  to finish within the smoke-timeout window

This suggests the main bottleneck is not parser/shape failure, but runtime cost
inside the full-graph local-conflict diagnostic body.

## 9. Unsupported Claims

The current evidence does not support the following claims:

1. support-neighbor exposure helps or hurts bot detection in TwiBot-20
2. support-induced fragility is a confirmed full-graph phenomenon
3. structural shock on the support-augmented graph explains the current
   failures
4. the current full-graph local-conflict stage is end-to-end validated
5. the current failure-mechanism names can be read literally in the labeled-only
   run

## 10. Practical Use Of This Record

For future discussion:

- use the completed labeled run as the current evidence source for method
  diagnosis
- treat the full-graph work as an implementation-and-validation boundary result,
  not yet as a result-bearing experiment
- do not interpret the labeled run's zero-valued support metrics as evidence
  against support-based mechanisms
- do not interpret the labeled run's `support_induced_fragile` bucket literally
  as support-noise evidence

For future implementation:

- if the next step is method redesign, the strongest labeled-run signal is that
  `dense_directional_recoverable` is small but real, while `conflict_dominant`
  is the dominant failure pool and is not repaired by the current conflict score
- if the next step is full-graph diagnosis, the immediate engineering issue is
  throughput/runtime, not row alignment
