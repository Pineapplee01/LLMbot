# TwiBot-22 Official-Prior Sampled v1 Specification

Date: 2026-06-15

Status: executable sampling specification for formal sampled TwiBot-22 baseline work.

## Objective

Construct a TwiBot-20-scale sampled TwiBot-22 subset without changing the released train/val/test task:

- sample inside the official TwiBot-22 `train`, `val`, and `test` pools;
- preserve each official split's own bot/human ratio;
- constrain selection by community coverage, degree distribution, text availability, and graph connectivity;
- emit manifests that make split priors, graph structure, and sampling decisions auditable.

The canonical dataset id is:

```text
TwiBot-22-official-prior-sampled-v1
```

Default sampled split sizes mirror the current TwiBot-20-scale baseline harness:

```text
train = 8278
val = 2365
test = 1183
```

## Why This Replaces `sampled-robust-v1` For Formal Comparison

The previous `TwiBot-22-sampled-robust-v1` script sampled from the full labeled pool and then constructed a new train/val/test split. That design was useful as a diagnostic proxy, but it changed the official TwiBot-22 split semantics and produced sampled split priors that did not match either official TwiBot-22 or a balanced benchmark.

This protocol keeps the released split boundaries fixed. Community, degree, text, and connectivity constraints are applied only inside fixed `official_split x label` pools, so the sampled dataset remains a downsampled version of the official task rather than a newly defined benchmark.

## Default Target Counts

Targets are computed with largest-remainder rounding from each official split's own bot/human ratio.

Using the verified local full TwiBot-22 distribution:

| Split | Official bot ratio | Target total | Target human | Target bot |
| --- | ---: | ---: | ---: | ---: |
| train | 7.80% | 8278 | 7632 | 646 |
| val | 27.96% | 2365 | 1704 | 661 |
| test | 29.44% | 1183 | 835 | 348 |

These numbers are intentionally not globally balanced. A balanced-test pack may be kept as an auxiliary stress benchmark, but it is not the formal sampled TwiBot-22 mainline.

## Sampling Algorithm

The executable implementation is:

```text
scripts/twibot22_sample_official_prior.py
```

The high-level flow is:

1. Load `label.csv`, `split.csv`, and labeled user profiles from the full local TwiBot-22 release.
2. Build the labeled user-user graph from `following` and `followers` edges.
3. Assign community strata from connected components: top components are kept separately, smaller components are bucketed by component size.
4. Compute global degree, activity, and account-age quantile buckets.
5. For each fixed `split x label` pool, allocate quota across `community x degree_bucket x text_available` strata.
6. Rank candidates inside each stratum by user-user degree, tweet count, account age, and deterministic seed jitter.
7. Run same-stratum connectivity repair: isolated selected users may be replaced only by candidates with the same split, label, community, degree bucket, and text-availability stratum.
8. Filter raw TwiBot-22 users, edges, lists, hashtags, and tweets for the sampled users.
9. Write `sample_manifest.json`, `split_audit.json`, `robustness_audit.json`, and `community_manifest.json`.

## Acceptance Gates

The sample is accepted only if:

- every sampled user keeps its official split;
- per-split bot/human counts match the computed official-prior targets;
- every sampled node has community and provenance metadata;
- the induced user-user graph has nonzero user-user edges;
- entity filtering completes and `user.json` contains exactly the sampled user count.

Connectivity metrics such as largest connected component fraction and isolated-node ratio are reported in `robustness_audit.json`. They are optimization signals, but they must not override official split or label constraints.

## Artifact Contract

Raw sampled root:

```text
F:\Datasets\Twibot22\sampled\official_prior_v1.__staging__
```

Prepared LLMbot root:

```text
G:\Research\BotDetection\datasets\TwiBot-22-official-prior-sampled-v1
```

Required raw artifacts:

- `sampled_users.csv`
- `split.csv`
- `label.csv`
- `edge.csv`
- `sampled_edges.csv`
- `edge_user_user_induced.csv`
- `community_assignments.jsonl`
- `sample_manifest.json`
- `split_audit.json`
- `robustness_audit.json`
- `community_manifest.json`
- filtered `user.json`, `list.json`, `hashtag.json`, and optional sampled `tweet_*.json`

The current sampled root is still named `official_prior_v1.__staging__` locally, and its sampled tweet snapshot is incomplete. For LMBot-paper-aligned preprocessing, always stream tweets from the complete TwiBot-22 release root instead of relying on the sampled snapshot.

After raw sampling, prepare the dataset with metadata, description, and up to 50 tweets per user:

```bash
python scripts/twibot22_prepare_sampled_for_llmbot.py \
  --sampled_root F:\Datasets\Twibot22\sampled\official_prior_v1.__staging__ \
  --output_root G:\Research\BotDetection\datasets\TwiBot-22-official-prior-sampled-v1 \
  --tweet_source_root F:\Datasets\Twibot22 \
  --max_tweets_per_user 50 \
  --drop_users_without_tweets \
  --source_dataset_id TwiBot-22-official-prior-sampled-v1 \
  --source_protocol_version official_prior_v1
```

The prepared `norm_user_text.json` follows the LMBot paper/code textual-sequence protocol:

```text
METADATA: ... </s> ... DESCRIPTION: ... TWEET: tweet1 </s> tweet2 ...
```

Descriptions and tweets are denoised by mapping URLs, mentions, hashtags, and emoji-like symbols to `HTTPURL`, `@USER`, `#HASHTAG`, and `EMOJI`, then tokenized with NLTK `TweetTokenizer`, matching the LMBot paper's social-text denoising intent and the official code's special-token vocabulary.

This same prepared dataset is also the required text/graph input surface for LGB-style experiments. LGB constructs unified user textual sequences from account attributes, personal descriptions, and tweets, then combines LM-derived node representations with directed social graph information. Therefore, `TwiBot-22-official-prior-sampled-v1` must not be treated as LMBot/LGB-aligned if `norm_user_text.json` is profile-only or if the tweet block is empty for most users.

For LMBot/LGB reproduction on sampled TwiBot-22, users without retained tweets are removed from the prepared graph/text dataset with `--drop_users_without_tweets`. This keeps the experiment aligned with both papers' assumption that each retained account has tweet evidence in the unified textual sequence. The script rewrites `split_new.json`, `label_new.json`, `labels.pt`, split index tensors, and `edge_index.pt`/`edge_type.pt` after filtering, and records removed users in `dropped_no_tweet_users.json`.

The prepare script writes into a `.__preparing__` directory and replaces the canonical prepared root only after all tensors, JSON files, and manifests are produced. If preprocessing is interrupted, the canonical dataset root must be restored from `.__previous__` or from `datasets/_backups/` before any experiment uses it.

## Reporting Rule

Use `TwiBot-22-official-prior-sampled-v1` as the formal sampled TwiBot-22 comparison dataset.

Use `TwiBot-22-sampled-robust-v1` only as a historical proxy/diagnostic result unless a paper section explicitly labels it as such.

Use BotPercent-style balanced TwiBot-22 only as an auxiliary prior-shift stress benchmark.
