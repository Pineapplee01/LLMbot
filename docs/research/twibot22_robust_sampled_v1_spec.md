# TwiBot-22 Robust Sampled v1 Specification

Date: 2026-06-14

Status: superseded for formal comparison. This document records the historical `TwiBot-22-sampled-robust-v1` proxy protocol and remains useful for diagnosing prior runs, but formal sampled TwiBot-22 baseline work should use [`TwiBot-22-official-prior-sampled-v1`](twibot22_official_prior_sampled_v1_spec.md). The key reason is that `robust_v1` samples from the full labeled pool before constructing a new split, while the official-prior protocol samples inside the released train/val/test pools and preserves each split's own bot/human ratio.

Historical status: executable research specification. This document defines the target sampling protocol, acceptance gates, and artifact contract for a robust `TwiBot-22` sampled subset aligned to the current `LLMbot` mainline. It is a design and implementation target, not an experimental result report.

## 1. Objective

Construct a stable sampled `TwiBot-22` subset that:

- is close to `TwiBot-20` scale for practical full-pipeline runs;
- preserves local social structure and community context instead of acting like a global iid sample;
- avoids the split-prior drift that caused the current sampled run to collapse toward the majority `human` class;
- is transparent enough that later failures can be audited against the sampling and split procedure.

The canonical dataset id for this spec is:

```text
TwiBot-22-sampled-robust-v1
```

The target labeled user count is:

```text
11,826 users
```

with an allowed tolerance of `+/- 5%` during construction and `+/- 2%` for the final released artifact.

## 2. Literature Basis

This protocol combines three evidence sources with different roles.

| Source | What it supports | How it constrains this spec |
| --- | --- | --- |
| `TwiBot-22` ([Feng 等 - TwiBot-22 Towards Graph-Based Twitter Bot Detection.pdf](H:/Zotero/attenger/Social%20bot%20detection/Feng%20%E7%AD%89%20-%20TwiBot-22%20Towards%20Graph-Based%20Twitter%20Bot%20Detection.pdf)) | Large-scale graph-oriented user collection should preserve diversity and local structure. | Use diversity-aware BFS and structure-preserving expansion instead of plain random downsampling. |
| `BotPercent` ([Tan 等 - 2023 - BotPercent estimating bot populations in Twitter communities.pdf](H:/Zotero/attenger/Social%20bot%20detection/Tan%20%E7%AD%89%20-%202023%20-%20BotPercent%20estimating%20bot%20populations%20in%20Twitter%20communities.pdf)) | Evaluation and resampling should respect community context and network proximity. | Build the sampled subset community-first and keep auxiliary prevalence-shift evaluation packs separate from the main canonical split. |
| `Simplistic Collection and Labeling Practices Limit the Utility of Benchmark Datasets for Twitter Bot Detection` ([Hays 等 - 2023 - Simplistic collection and labeling practices limit the utility of benchmark datasets for Twitter bot.pdf](H:/Zotero/attenger/Social%20bot%20detection/Hays%20%E7%AD%89%20-%202023%20-%20Simplistic%20collection%20and%20labeling%20practices%20limit%20the%20utility%20of%20benchmark%20datasets%20for%20Twitter%20bot.pdf)) | Benchmark datasets can look strong while encoding collection and labeling artifacts. | Sampling and splitting must be auditable, transparent, and stress-tested against community and label-prior artifacts. |

## 3. Scope and Non-Goals

### In scope

- Sampling labeled users from the full `TwiBot-22` user graph.
- Preserving a user-user graph suitable for current `LLMbot` graph baselines.
- Producing a canonical `train/valid/test` split for transductive graph training.
- Producing manifest and audit artifacts that record how the subset was built.

### Out of scope for v1

- Re-annotating users or replacing `TwiBot-22` released labels.
- Reproducing the full `TwiBot-22` heterogeneous graph with every entity type.
- Claiming this subset is a new benchmark superior to the full release.
- Forcing a balanced 50/50 bot-human label prior in the main canonical split.

Balanced or prevalence-shift evaluation packs are allowed as auxiliary artifacts, but they do not replace the main canonical sampled dataset.

## 4. Source Data Assumptions

The source dataset is the full local `TwiBot-22` release rooted at:

```text
F:\Datasets\Twibot22
```

The protocol assumes the source contains at minimum:

- user identifiers;
- released bot/human labels;
- user metadata;
- user-user relations with direction;
- user text or tweet content sufficient to build `norm_user_text`.

If a required field is missing, the run must fail with an explicit manifest note. It must not silently downgrade to a weaker protocol.

## 5. Design Principles

1. **Community-first, not global-random.**  
   Users are sampled through community-local expansion, then merged globally.

2. **Diversity-aware expansion, not degree-only crawling.**  
   Expansion must preserve variation in user metadata and neighborhood type.

3. **Matched canonical split, not accidental prevalence shift.**  
   The main `train/valid/test` split must preserve label prior and structural marginals closely enough that a majority-class collapse is not baked into the data.

4. **Main sample preserves real prevalence.**  
   The canonical sampled dataset should stay close to the sampled population's natural bot/human ratio. Artificially balanced subsets belong in auxiliary stress packs, not the main release.

5. **Auditability is mandatory.**  
   Per-node community assignment, sampling provenance, split assignment, and final audit statistics must be recorded.

## 6. Canonical Output

The canonical output has two layers.

### 6.1 Raw sampled subset

Stable root:

```text
F:\Datasets\Twibot22\sampled\robust_v1
```

This layer contains the sampled user ids, raw relations among sampled users, community assignments, split assignment, and sampling manifests.

### 6.2 Prepared `LLMbot` dataset

Stable root:

```text
G:\Research\BotDetection\datasets\TwiBot-22-sampled-robust-v1
```

This layer contains the prepared artifacts expected by current `LLMbot` entrypoints, such as:

- `norm_user_text.json`
- `labels.pt`
- `train_idx.pt`
- `valid_idx.pt`
- `test_idx.pt`
- `edge_index.pt`
- `edge_type.pt`
- `prepare_manifest.json`
- `sample_manifest.json`
- `split_audit.json`

## 7. Community Construction

The subset must be built from explicit community buckets before global merging.

### 7.1 Community count

The default target is:

```text
10 communities
```

with soft per-community quotas and a hard requirement that no community is only represented in train.

### 7.2 Community types

The default executable `robust_v1` runtime uses:

- `5` anchor-account communities inspired by BotPercent-style closely connected sub-communities;
- `5` graph-diverse fallback communities selected to expand coverage away from the anchor neighborhoods while preserving user-user structure.

This is an intentional operational simplification for v1. Topic or hashtag-driven
community construction remains a valid future extension, but it is not required
for a run to qualify as `robust_v1`.

### 7.3 Anchor-account communities

Default seeds follow the BotPercent discussion when present in the local graph:

- `@BarackObama`
- `@elonmusk`
- `@CNN`
- `@NeurIPSConf`
- `@ladygaga`

If a seed is absent or unusable in the local snapshot, it may be replaced, but the replacement must be documented in the manifest with:

- original seed name;
- replacement seed id;
- replacement reason.

### 7.4 Graph-diverse fallback communities

The executable v1 runtime selects the non-anchor communities by:

1. ranking labeled users by graph activity and availability of local user-user structure;
2. excluding users already covered by anchor seeds and their immediate neighborhoods when possible;
3. preferring diverse account-age and activity buckets so that one narrow user slice does not dominate the sample;
4. using these fallback seeds for the same network-proximity expansion as the anchor communities.

If a future implementation upgrades these fallback communities to topic or
hashtag-derived communities, that change must be recorded in the manifest and
version notes.

## 8. Node Sampling Protocol

Sampling is performed in four stages.

### 8.1 Stage A: Seed initialization

For each community:

- initialize a seed set from the anchor account or topic cluster prototype;
- include a small seed neighborhood to avoid a single-center artifact;
- record the seed source for every initial node.

### 8.2 Stage B: Diversity-aware BFS expansion

Expansion follows the `TwiBot-22` diversity-aware BFS idea on user-user relations.

Allowed relation directions:

- `following`
- `follower`

Both directions must be considered during expansion. The protocol must not crawl only one direction unless a failure reason is recorded.

At each expansion step, choose:

- one metadata dimension from the active metadata pool;
- one diversity strategy from:
  - `distribution_diversity`
  - `value_diversity`

#### Metadata pool

The default metadata pool is:

- `followers_count`
- `following_count`
- `tweet_count`
- `account_age`
- `verified`
- `protected`
- `listed_count` when available

#### Distribution diversity

For numeric metadata:

- select from the top range;
- select from the bottom range;
- select from the middle or remainder range.

For boolean metadata:

- prefer selecting from both truth values when available.

#### Value diversity

For numeric metadata:

- sample neighbors with probability proportional to metadata difference from the expanding node.

For boolean metadata:

- prefer the opposite value when available.

#### Default expansion caps

These are v1 defaults and may be tuned only if recorded in the manifest:

- `max_hops_per_community = 2`
- `max_frontier_expansions_per_node = 6`
- `relation_direction_balance = roughly_even`

### 8.3 Stage C: Community-local pruning

After expansion, each community is pruned to a soft quota using:

- proximity to the seed or topic center;
- internal edge density;
- metadata diversity retention;
- text availability;
- label availability.

Default soft quota range per community:

```text
800 to 1,400 users
```

No single community may occupy more than `20%` of the final sampled user set unless explicitly justified.

### 8.4 Stage D: Global merge and deduplication

After local pruning:

- merge all community user sets;
- deduplicate globally by user id;
- assign each node a `primary_community_id`.

If a node belongs to multiple candidate communities, choose the primary community by this priority:

1. highest internal edge density to that community;
2. shortest path or closest expansion depth to that community seed;
3. highest text/topic membership score for topic communities;
4. deterministic tie-break by user id.

The raw manifest must retain secondary community membership candidates for auditability.

## 9. Split Construction

The canonical split is:

```text
train / valid / test = 7 / 2 / 1
```

matching the current practical shape used in `LLMbot`.

For the canonical `11,826`-user target, the default split sizes are:

- `train = 8,278`
- `valid = 2,365`
- `test = 1,183`

### 9.1 Split is supervision-only, not graph cutting

This is a transductive graph setting.

- Edges across split boundaries are allowed.
- The graph is built over the entire sampled user set.
- Split assignment controls supervision and evaluation only.

### 9.2 Stratification axes

Split assignment must be stratified jointly over:

- label (`bot`, `human`);
- primary community id;
- log-degree bucket;
- account-age bucket;
- activity bucket.

Recommended defaults:

- `log_degree_bucket`: quantile buckets over `log1p(total_degree)`
- `account_age_bucket`: quantile buckets or interpretable age bins
- `activity_bucket`: quantile buckets over `tweet_count` or posting proxy

### 9.3 Hard split constraints

The canonical split is accepted only if all conditions below hold:

1. **Overall label prior alignment**
   - maximum absolute bot-ratio difference across `train`, `valid`, `test` is `<= 3 percentage points`

2. **Per-community coverage**
   - every community with at least `100` sampled users must appear in all three splits

3. **Per-community label alignment**
   - for communities large enough to support it, maximum bot-ratio difference across splits is `<= 5 percentage points`
   - executable `robust_v1` interprets "large enough to support it" as:
     - community sampled size `>= 100`, and
     - minority-label sampled count `>= 30`

4. **Structural marginal alignment**
   - median `log1p(total_degree)` deviation across splits is `<= 10%` of the pooled sampled median

5. **Activity marginal alignment**
   - median `log1p(tweet_count)` deviation across splits is `<= 10%` of the pooled sampled median

### 9.4 Soft optimization target

If multiple split assignments satisfy the hard constraints, prefer the one that minimizes aggregate divergence across:

- label prior;
- community mix;
- degree buckets;
- activity buckets;
- account-age buckets.

## 10. Graph Preservation Rules

The final graph artifact is the induced user-user subgraph over the sampled users.

### Required graph properties

- preserve direction labels for `following` and `follower`;
- preserve all observed user-user edges among sampled users;
- do not rebalance or rewrite graph labels after split assignment.

### Acceptance targets

The sampled induced subgraph should satisfy:

- largest connected component covers at least `70%` of sampled users;
- isolated-node ratio is at most `10%`;
- reciprocity ratio is reported globally and per split;
- community-local edge density is reported for every community.

These are acceptance-audit targets, not optimization objectives during training.

## 11. Mainline vs Auxiliary Evaluation Packs

### 11.1 Canonical mainline dataset

The mainline dataset must:

- preserve the sampled population's natural label prevalence;
- use the matched `7/2/1` split above;
- be the default dataset for `LLMbot` baseline and mainline experiments.

### 11.2 Auxiliary balanced pack

An optional balanced pack may be built for stress testing, but:

- it must use a separate `split_id`;
- it must not replace the mainline dataset;
- it must be explicitly labeled as auxiliary.

### 11.3 Auxiliary prevalence-shift community packs

Following BotPercent-style evaluation, optional community-local packs may be created with bot prevalence sweeps such as:

```text
10%, 20%, ..., 90%
```

These are evaluation stress packs only. They are not the canonical training split.

### 11.4 Optional held-out community pack

An optional stronger robustness pack may reserve one or more whole communities for validation or test.

If created, it must be versioned separately because it changes the generalization task.

## 12. Leakage and Audit Rules

The protocol must explicitly guard against benchmark leakage.

### Required checks

- no duplicate user id in the final sampled set;
- no split overlap by user id;
- no overlap between auxiliary evaluation-only community packs and the training portion of another pack unless documented;
- if known benchmark communities or expert-labeled subsets are reused for evaluation, record any removed user ids in the manifest.

### Required per-node provenance

Each sampled user must record:

- `user_id`
- `primary_community_id`
- `sampling_stage`
- `seed_source`
- `expansion_depth`
- `relation_direction_used`
- `diversity_strategy_last_used`
- `metadata_dimension_last_used`
- `dedup_resolution_status`
- `split`

## 13. Required Artifacts

The raw sampled subset root must contain at minimum:

- `sample_manifest.json`
- `community_manifest.json`
- `community_assignments.jsonl`
- `sampled_users.csv`
- `sampled_edges.csv`
- `split.csv`
- `label.csv`
- `split_audit.json`
- `robustness_audit.json`

The prepared `LLMbot` root must contain at minimum:

- `norm_user_text.json`
- `labels.pt`
- `train_idx.pt`
- `valid_idx.pt`
- `test_idx.pt`
- `edge_index.pt`
- `edge_type.pt`
- `prepare_manifest.json`
- `sample_manifest.json`
- `split_audit.json`

## 14. Required Manifest Fields

`sample_manifest.json` must include:

- `dataset_id`
- `source_dataset`
- `source_root`
- `protocol_version`
- `target_user_count`
- `final_user_count`
- `community_count`
- `community_types`
- `seed_accounts`
- `topic_clustering_method`
- `metadata_pool`
- `max_hops_per_community`
- `max_frontier_expansions_per_node`
- `split_ratio`
- `overall_label_counts`
- `overall_label_ratio`
- `per_split_label_counts`
- `per_split_label_ratio`
- `per_community_counts`
- `per_community_label_ratio`
- `graph_edge_counts`
- `relation_counts`
- `dedup_summary`
- `leakage_checks`
- `rejection_checks`
- `notes`

## 15. Acceptance Gates

The dataset is only accepted as `robust_v1` if all gates below pass.

### Gate A: Size and coverage

- final labeled user count within the allowed tolerance;
- exactly one primary community per sampled user;
- every large community represented in all splits.

### Gate B: Prior-shift control

- overall split bot-ratio gap `<= 3 pp`;
- no split differs from the pooled sampled bot ratio by more than `3 pp`.

### Gate C: Structural sanity

- induced graph built successfully;
- connected-component and isolation targets recorded and within bounds;
- direction-specific relation counts are non-zero unless the source data truly lacks them.

### Gate D: Auditability

- required manifests and audits exist;
- every sampled node has provenance fields;
- rejection checks are explicitly marked pass or fail.

### Gate E: Training-readiness

- current `LLMbot` loader can consume the prepared dataset without ad hoc path overrides;
- majority-class baseline on `train`, `valid`, and `test` is reported in `split_audit.json`.

## 16. Explicit Rejection Conditions

The run must be rejected and not promoted to `robust_v1` if any condition below occurs:

1. `train`, `valid`, and `test` have large label-prior drift, such as the current failure regime where `train` bot ratio is much lower than `valid/test`.
2. A single community dominates the sample to the point that global metrics are effectively community-specific.
3. The sample is nearly disconnected or structurally too sparse for graph baselines.
4. Split assignment was performed only on label and ignored community/degree/activity structure.
5. The run cannot explain why a node was selected or how it was assigned to a community and split.

## 16.1 Current executable boundary

The current executable implementation under
[`scripts/twibot22_sample_botpercent_aligned.py`](G:/Research/BotDetection/scripts/twibot22_sample_botpercent_aligned.py)
operationalizes community construction as:

- `5` BotPercent-style anchor-account seeds;
- `5` graph-diverse fallback seeds;
- full-pool sampling before split assignment;
- post-sampling split optimization with audit gates.

This is the authoritative runtime interpretation for `robust_v1` until a newer
spec version replaces it.

## 17. Recommended Implementation Order

For the first implementation pass:

1. build community buckets;
2. run diversity-aware BFS expansion;
3. prune to community quotas;
4. merge and deduplicate globally;
5. optimize split assignment under the hard constraints;
6. write raw sampled artifacts;
7. prepare the `LLMbot` dataset;
8. run loader validation and split audit.

## 18. Validation Checklist

Before using the dataset in experiments, verify:

1. community count and community quotas;
2. per-split label ratios;
3. majority-class accuracy per split;
4. per-split degree and activity bucket alignment;
5. induced graph node/edge counts;
6. loader compatibility with current `LLMbot`.

## 19. v1 Boundary

This protocol is intentionally conservative.

It does **not** claim to solve:

- `TwiBot-22` label quality limitations inherited from the source release;
- cross-platform or cross-time generalization;
- heterogeneous-entity preservation beyond what current `LLMbot` baselines need.

What it does claim to solve is narrower and operational:

```text
Build a TwiBot-20-scale sampled TwiBot-22 subset that preserves community-local graph structure,
keeps split priors aligned, and exposes enough provenance to audit future model failures.
```
