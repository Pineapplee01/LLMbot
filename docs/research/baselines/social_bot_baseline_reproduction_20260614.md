# Social Bot Baseline Reproduction Ledger, 2026-06-14

This ledger tracks official-code baseline status for the current BotDetection comparison plan.
It separates reusable server evidence from runs that need extra preprocessing or are unsupported by
the official repository.

## Scope

- Target methods: BotRGCN, RGT, BotMoE, BIC, SEBot, BotBR, BotHunter, FriendBot.
- Target datasets requested: TwiBot-20 and TwiBot-22.
- Server root: `/root/workspace/LMbot`.
- Local root: `G:\Research\BotDetection`.
- Primary comparison metrics for this project: Accuracy and Macro-F1.

## Unified Metric Table Update, 2026-06-17

The current unified metric artifacts are:

- Server JSON: `/root/workspace/LMbot/repro_baselines_20260614/results/unified_metric_tables_20260617/unified_metrics.json`
- Server Markdown: `/root/workspace/LMbot/repro_baselines_20260614/results/unified_metric_tables_20260617/unified_metrics.md`
- Local JSON mirror: `G:\Research\BotDetection\repro_baselines_20260614\unified_metrics.json`
- Local Markdown mirror: `G:\Research\BotDetection\repro_baselines_20260614\unified_metrics.md`

Claim boundary:

- TwiBot-20 rows are reusable reproduced/adapted baseline evidence under the canonical TwiBot-20 split.
- `TwiBot-22-official-prior-sampled-v1` rows are corrected sampled/profile-only diagnostics, not full official TwiBot-22 reproduction.
- The historical `TwiBot-22-sampled` and `TwiBot-22-sampled-robust-v1` proxy results remain diagnostic only because their test bot prior is too low.
- SEBot is excluded from the main comparison table per the current comparison plan; its adapted TwiBot-20 diagnostic is retained in `excluded_diagnostics` inside the unified JSON.
- BotHunter and FriendBot now have a sampled TwiBot-22 adapter at `scripts/run_twibot22_official_feature_baselines_sampled.py`. The adapter reconstructs official-baseline-readable `user.json` / `tweet_*.json` / `label.csv` / `split.csv` / `edge.csv` from the corrected prepared sample when raw sampled artifacts are unavailable. This is a sampled-dataset adapter diagnostic, not full official TwiBot-22 reproduction.

Current TwiBot-20 unified table:

| Method | Status | Accuracy | Macro-F1 | Binary-F1 | Metric source |
| --- | --- | ---: | ---: | ---: | --- |
| BotRGCN | completed | `0.8568 +/- 0.0057` | `0.8549 +/- 0.0055` | `0.8714 +/- 0.0063` | native summary |
| RGT | completed | `0.8639 +/- 0.0025` | `0.8620 +/- 0.0026` | `0.8782 +/- 0.0023` | native summary |
| BIC | completed | `0.8555 +/- 0.0050` | `0.8534 +/- 0.0051` | `0.8707 +/- 0.0047` | native summary |
| BotMoE | completed | `0.8612 +/- 0.0056` | `0.8592 +/- 0.0053` | `0.8759 +/- 0.0062` | macro rerun summary |
| BotBR | completed | `0.8691 +/- 0.0020` | `0.8672 +/- 0.0021` | `0.8831 +/- 0.0018` | macro-F1 derived from rounded acc/binary-F1 logs |
| LMBot-GNN | completed | `0.8511 +/- 0.0039` | `0.8465 +/- 0.0039` | `0.8723 +/- 0.0042` | macro-F1 derived from exact acc/binary-F1 result JSON |
| LMBot-LM | completed | `0.8541 +/- 0.0010` | `0.8487 +/- 0.0021` | `0.8771 +/- 0.0012` | macro-F1 derived from exact acc/binary-F1 result JSON |
| Frozen SimTeG | completed | `0.8631 +/- 0.0036` | `0.8613 +/- 0.0038` | `0.8769 +/- 0.0032` | canonical replayed `outputs.pt`, 3 seeds |
| BotHunter | completed | `0.7528 +/- 0.0042` | `0.7439 +/- 0.0042` | `0.7918 +/- 0.0039` | TwiBot-22 packaged feature baseline adapter |
| FriendBot | completed | `0.7719 +/- 0.0046` | `0.7663 +/- 0.0048` | `0.8026 +/- 0.0038` | TwiBot-22 packaged feature baseline adapter |

Current corrected sampled TwiBot-22 table:

| Method | Status | Accuracy | Macro-F1 | Binary-F1 | Metric source |
| --- | --- | ---: | ---: | ---: | --- |
| BotRGCN | completed | `0.7221 +/- 0.0229` | `0.6055 +/- 0.0210` | `0.3919 +/- 0.0411` | native summary |
| RGT | completed | `0.7224 +/- 0.0238` | `0.5673 +/- 0.0691` | `0.3090 +/- 0.1292` | native summary |
| BIC | completed | `0.7856 +/- 0.0097` | `0.7342 +/- 0.0095` | `0.6173 +/- 0.0135` | native summary |
| BotBR | completed | `0.7609 +/- 0.0399` | `0.6034 +/- 0.1220` | `0.3551 +/- 0.2265` | macro-F1 derived from rounded acc/binary-F1 logs |
| BotMoE | completed | `0.6499 +/- 0.0160` | `0.6364 +/- 0.0136` | `0.5669 +/- 0.0111` | reparsed epoch logs; macro-F1 derived from acc/binary-F1 |
| Frozen SimTeG | completed | `0.6583 +/- 0.0620` | `0.5475 +/- 0.0178` | `0.3438 +/- 0.1162` | native summary |
| LMBot-GNN | running partial | `0.7067 +/- 0.0000` | `0.4168 +/- 0.0000` | `0.0057 +/- 0.0000` | seed-1 result JSON; macro-F1 derived from exact acc/binary-F1 |
| LMBot-LM | running partial | `0.7109 +/- 0.0000` | `0.4522 +/- 0.0000` | `0.0757 +/- 0.0000` | seed-1 result JSON; macro-F1 derived from exact acc/binary-F1 |
| BotHunter | completed locally | `0.7731 +/- 0.0019` | `0.6323 +/- 0.0031` | `0.4047 +/- 0.0062` | prepared-sample reconstruction adapter; local 5-seed run |
| FriendBot | running locally | pending | pending | pending | prepared-sample reconstruction adapter; official CPU ego-network feature extraction still running |

Active corrected sampled TwiBot-22 LMBot jobs:

- Main seed-1/root LMBot run PID file: `/root/workspace/LMbot/repro_baselines_20260614/logs/lmbot_twibot22_official_prior_sampled_5seed_20260617.pid`
- Parallel seed-2-to-5 run PID file: `/root/workspace/LMbot/repro_baselines_20260614/logs/lmbot_twibot22_official_prior_sampled_parallel_seed2_5_20260617.pid`
- Output root: `/root/workspace/LMbot/repro_baselines_20260614/results/lmbot_twibot22_official_prior_sampled_5seed_20260617`
- As of the 2026-06-17 check, seed 1 had written `lmbot_seed_1/results_GNN.json` and `lmbot_seed_1/results_LM.json`.
- The main seed-1 job then began duplicate seed-2 work because its original command used `--seeds 1,2,3,4,5`; it was stopped after seed 1 completed to avoid duplicating the parallel seed-2-to-5 run.
- The parallel seed-2-to-5 job remains active and is the owner for the remaining LMBot corrected sampled TwiBot-22 results.

## Current Completion Matrix, 2026-06-15

This table is a reproduction audit matrix, not a single fair-comparison table. It preserves the
claim boundary for each row so adapted diagnostics and proxy runs are not confused with official
full-dataset reproductions.

| Method | Paper / code verified | TwiBot-20 server evidence | Full TwiBot-22 official status | TwiBot-22-sampled proxy status |
| --- | --- | --- | --- | --- |
| BotRGCN | Yes | Completed official-code 5-seed: Acc `0.8568 +/- 0.0057`, Macro-F1 `0.8549 +/- 0.0055` | Blocked: no full ready method-specific processed tensors on server | Completed 5-seed proxy: Acc `0.8920 +/- 0.0042`, Macro-F1 `0.6918 +/- 0.0152` |
| RGT | Yes | Completed adapted-preprocessing 5-seed: Acc `0.8639 +/- 0.0025`, Macro-F1 `0.8620 +/- 0.0026` | Blocked: official code expects Google Drive-style preprocessed files not present on server | Completed 5-seed proxy: Acc `0.8948 +/- 0.0060`, Macro-F1 `0.6833 +/- 0.0291` |
| BotMoE | Yes | Reused checkpoint summary: best checked Acc `0.8597`, binary F1 `0.8752`; Macro-F1 not logged | Blocked: referenced full TwiBot-22 processed path is missing | Completed runtime-only proxy diagnostic: Acc `0.8269 +/- 0.0259`, binary F1 `0.3485 +/- 0.1173`; Macro-F1 not emitted |
| BotHunter | Yes | Completed TwiBot-22 official-code feature baseline, 5 RF seeds: Acc `0.7528 +/- 0.0042`, Macro-F1 `0.7439 +/- 0.0042`, binary F1 `0.7918 +/- 0.0039` | Not targeted in this ledger update | Not targeted |
| FriendBot | Yes | Official TwiBot-22 code launched on TwiBot-20; still running because official ego-network feature extraction is expensive on the full user/support-node graph | Not targeted in this ledger update | Not targeted |
| BIC | Yes | Completed adapted/proxy diagnostic 5-seed: Acc `0.8555 +/- 0.0050`, Macro-F1 `0.8534 +/- 0.0051` | Blocked: official TwiBot-22 generated data artifacts are missing | Completed 5-seed proxy diagnostic: Acc `0.8916 +/- 0.0029`, Macro-F1 `0.7092 +/- 0.0210` |
| SEBot | Yes | Completed runtime-only labeled-only adapted diagnostic: Acc `0.8629 +/- 0.0036`, binary F1 `0.8772 +/- 0.0030`; Macro-F1 not emitted | Blocked: author structural-entropy tree/subgraph artifacts unavailable non-interactively | Completed runtime-only proxy diagnostic after `oomfix`: Acc `0.8793 +/- 0.0029`, binary F1 `0.1892 +/- 0.1187`; Macro-F1 not emitted |
| BotBR | Yes | Reused 5-run server result: Acc `0.8691 +/- 0.0020`, binary F1 `0.8831 +/- 0.0018`; Macro-F1 not logged | Blocked: no ready full TwiBot-22 official support path found | Completed runtime-only proxy diagnostic: Acc `0.8810 +/- 0.0010`, binary F1 `0.1719 +/- 0.1841`; Macro-F1 not emitted |

Completion boundary:

- The paper/code retrieval portion is complete for all six methods; official repository URLs and paper evidence are recorded below.
- TwiBot-20 has server evidence for all six methods, but only BotRGCN is a clean official-code reproduction with the shared available tensors. RGT, BIC, and SEBot are adapted diagnostics; BotMoE and BotBR are reused server evidence with Macro-F1 missing from the logged metrics.
- BotHunter has now been reproduced from the TwiBot-22 official baseline code on TwiBot-20 with a dataset-order adapter required by the official feature scripts. FriendBot official-code reproduction is running from the same adapter; do not replace it with the earlier `FriendBot-style` proxy.
- Full official TwiBot-22 experiments remain blocked for all six methods because the server has raw TwiBot-22 but not the method-specific processed artifacts required by the official repositories.
- The `TwiBot-22-sampled proxy` table is diagnostic evidence only; it must not be reported as full official TwiBot-22 performance.
- For future formal sampled TwiBot-22 comparisons, replace the historical `TwiBot-22-sampled-robust-v1` proxy with `TwiBot-22-official-prior-sampled-v1`, whose construction samples inside the official train/val/test pools and preserves each split's own bot/human ratio. See `docs/research/twibot22_official_prior_sampled_v1_spec.md`.

## Code Sync Status

All six official repositories are present locally and on the server.

| Method | Official repository | Local path | Server path | Server commit |
| --- | --- | --- | --- | --- |
| BotRGCN | `https://github.com/BunsenFeng/BotRGCN` | `G:\Research\BotDetection\BotRGCN` | `/root/workspace/LMbot/BotRGCN` | `eabe7fd` |
| RGT | `https://github.com/BunsenFeng/BotHeterogeneity` | `G:\Research\BotDetection\RGT` | `/root/workspace/LMbot/RGT` | `0f74fc4` |
| BotMoE | `https://github.com/lyh6560new/BotMoE` | `G:\Research\BotDetection\BotMoE` | `/root/workspace/LMbot/BotMoE` | `a423930` |
| BIC | `https://github.com/LzyFischer/BIC` | `G:\Research\BotDetection\BIC` | `/root/workspace/LMbot/BIC` | `b7d4592` |
| SEBot | `https://github.com/846468230/SEBot.git` | `G:\Research\BotDetection\SEBot` | `/root/workspace/LMbot/SEBot` | `59b5407` |
| BotBR | `https://github.com/GauwinSesi1/botbr.git` | `G:\Research\BotDetection\botbr` | `/root/workspace/LMbot/botbr` | `08d102c` |
| TwiBot-22 feature baselines | `https://github.com/LuoUndergradXJTU/TwiBot-22` | `G:\Research\BotDetection\tmp_external\TwiBot-22` | `/root/workspace/LMbot/tmp_external/TwiBot-22` | `54adf9a` |

## Paper And Code Evidence

| Method | Paper | Venue / year | Method family | Official code evidence |
| --- | --- | --- | --- | --- |
| BotRGCN | BotRGCN: Twitter Bot Detection with Relational Graph Convolutional Networks | ASONAM 2021 | Multimodal RGCN over follow relations | GitHub README names the ASONAM 2021 paper and lists required preprocessed tensors: `des_tensor.pt`, `tweets_tensor.pt`, `num_properties_tensor.pt`, `cat_properties_tensor.pt`, `edge_index.pt`, `edge_type.pt`, `label.pt`. |
| RGT | Heterogeneity-aware Twitter Bot Detection with Relational Graph Transformers | AAAI 2022 | Relational graph transformer with semantic attention over relation-specific subgraphs | GitHub README identifies the AAAI 2022 paper and arXiv `2109.02927`; local code entry is `TBD.py`. |
| BotMoE | BotMoE: Twitter Bot Detection with Community-Aware Mixtures of Modal-Specific Experts | SIGIR 2023 | Metadata/text/graph modal experts with community-aware MoE fusion | GitHub README identifies the SIGIR 2023 paper and training/testing commands. |
| BIC | BIC: Twitter Bot Detection with Text-Graph Interaction and Semantic Consistency | ACL 2023 | Text-graph interaction plus semantic-consistency modeling over tweet sequences | GitHub README provides the preprocessed-data link and states BIC uses text-graph interaction and semantic consistency. |
| SEBot | SeBot: Structural Entropy Guided Multi-View Contrastive Learning for Social Bot Detection | KDD 2024 | Structural-entropy node/subgraph views plus multi-view contrastive learning | GitHub README identifies the KDD 2024 paper and provides dataset, encoding tree, and subgraph artifact links. |
| BotBR | BotBR: Social Bot Detection with Balanced Feature Fusion and Reliability-Enhanced Graph Learning | SIGIR 2025 | Balanced feature fusion and reliability-enhanced graph learning | GitHub README lists TwiBot-20 and MGTAB train/test commands and the SIGIR 2025 citation. |
| BotHunter | Bot-hunter: A Tiered Approach to Detecting & Characterizing Automated Activity on Twitter | 2018 technical report / baseline as packaged by TwiBot-22 | Random forest over profile, network, content, and timing features | TwiBot-22 official `src/BotHunter/readme.md` specifies `preprocess.py --dataset DATASETNAME` then `train.py --dataset DATASETNAME`; its TwiBot-20 table reports Acc `0.7522 +/- 0.0044` and binary F1 `0.7909 +/- 0.0036`. |
| FriendBot | You Are Known by Your Friends: Leveraging Network Metrics for Bot Detection in Twitter | SBP-BRiMS 2020 / baseline as packaged by TwiBot-22 | Random forest over ego-network, profile, and text/network features | TwiBot-22 official `src/FriendBot/readme.md` specifies `rand_forest.py --dataset Twibot-20` with seeds `0, 100, 200, 300, 400`; its TwiBot-20 table reports Acc `0.7589 +/- 0.0047` and binary F1 `0.7997 +/- 0.0034`. |

External evidence checked:

- BotRGCN official code: `https://github.com/BunsenFeng/BotRGCN`; paper: `https://arxiv.org/abs/2106.13092`.
- RGT official code: `https://github.com/BunsenFeng/BotHeterogeneity`; paper: `https://arxiv.org/abs/2109.02927`.
- BotMoE official code: `https://github.com/lyh6560new/BotMoE`; paper: `https://arxiv.org/abs/2304.06280`.
- BIC official code: `https://github.com/LzyFischer/BIC`; paper: `https://arxiv.org/abs/2208.08320` and ACL PDF `https://aclanthology.org/2023.acl-long.575.pdf`.
- SEBot official code: `https://github.com/846468230/SEBot`; paper: `https://arxiv.org/abs/2405.11225`.
- BotBR official code: `https://github.com/GauwinSesi1/botbr`.
- BotHunter and FriendBot official packaged code: `https://github.com/LuoUndergradXJTU/TwiBot-22/tree/master/src/BotHunter` and `https://github.com/LuoUndergradXJTU/TwiBot-22/tree/master/src/FriendBot`.

## Reusable TwiBot-20 Evidence

| Method | Status | Accuracy | Macro-F1 | Other reported metric | Evidence |
| --- | --- | ---: | ---: | ---: | --- |
| BotRGCN | Completed official-code run, 5 seeds | `0.8568 +/- 0.0057` | `0.8549 +/- 0.0055` | binary F1 `0.8714 +/- 0.0063` | `/root/workspace/LMbot/repro_baselines_20260614/results/botrgcn_twibot20_official_5seed/summary.json` |
| BotBR | Existing 5-run server result, reused | `0.8691 +/- 0.0020` | not logged as macro-F1 | binary F1 `0.8831 +/- 0.0018` | `/root/workspace/LMbot/repro_baselines_20260527/logs/botbr_twibot20_seed1_5.log` |
| BotMoE | Existing checkpoint test summary, reused | best checked `0.8597` | not logged as macro-F1 | binary F1 best checked `0.8752` | `/root/workspace/LMbot/BotMoE/checkpoint_test_summary_20260602.json` |
| BotHunter | Completed TwiBot-22 official feature-baseline code, 5 RF seeds | `0.7528 +/- 0.0042` | `0.7439 +/- 0.0042` | binary F1 `0.7918 +/- 0.0039` | `/root/workspace/LMbot/repro_baselines_20260614/results/twibot22_official_feature_baselines_twibot20_20260615/bothunter/summary.json` |
| FriendBot | Official TwiBot-22 feature-baseline code launched; still running | pending | pending | pending | PID `32578`; `/root/workspace/LMbot/repro_baselines_20260614/logs/twibot22_official_feature_baselines_twibot20_20260615/friendbot_train.log` |
| RGT | Completed adapted-preprocessing run, 5 seeds | `0.8639 +/- 0.0025` | `0.8620 +/- 0.0026` | binary F1 `0.8782 +/- 0.0023` | `/root/workspace/LMbot/repro_baselines_20260614/results/rgt_twibot20_adapted_5seed/summary.json` |
| BIC | Completed adapted-preprocessing / proxy diagnostic run, 5 seeds | `0.8555 +/- 0.0050` | `0.8534 +/- 0.0051` | binary F1 `0.8707 +/- 0.0047` | `/root/workspace/LMbot/repro_baselines_20260614/results/bic_twibot20_adapted_5seed/summary.json` |
| SEBot | Completed runtime-only labeled-only adapted diagnostic, 5 seeds | `0.8629 +/- 0.0036` | not emitted | binary F1 `0.8772 +/- 0.0030` | `/root/workspace/LMbot/repro_baselines_20260614/results/sebot_twibot20_labeled_only_runtime_5seed_20260615/summary.json` |

Notes:
- BotRGCN official code logs binary F1 by default. A runtime-only hook saved macro-F1 without editing the official repository.
- BotHunter/FriendBot use the TwiBot-22 packaged official baseline code. The runtime adapter maps TwiBot-20 `label_new.json` and `split_new.json` to the CSV format expected by that code and rewrites `node.json` so labeled users appear first in `label.csv` order, then other users, then non-user/tweet nodes. This preserves official feature definitions while satisfying the code's node-order assumption.
- BotHunter's reproduced binary F1 and accuracy match the TwiBot-22 README reported TwiBot-20 numbers within normal RF seed variation. Macro-F1 is additionally logged for this project protocol.
- FriendBot official feature extraction is much slower than the earlier `FriendBot-style` proxy because the official code constructs ego networks and computes Louvain/centrality/text-similarity features over the full user/support-node graph. The proxy result must not be used as the official FriendBot row.
- BotBR and BotMoE existing results should not be rerun unless the comparison table requires macro-F1 under the exact project metric protocol.
- Existing internal LLMbot BotRGCN/RGT logs are not official-code reproductions and should be labeled separately if used.

## TwiBot-20 Runs Not Yet Claim-Ready

| Method | Current blocker | Why not run directly |
| --- | --- | --- |
| RGT | Official code hardcodes `/content/drive/MyDrive/` and expects `label_list.pt`, `follower_edge.pt`, `following_edge.pt`, `My_user_feature_ZS.pt`, `My_user_feature_bool.pt`, `user_features2.pt`, and `user_features3.pt`. | These are not the same names or exact preprocessing contract as the shared processed tensors. Running requires a documented adapter or regenerating the official preprocessing files. |
| BIC | Missing TwiBot-20 repo-local files `tweet.pt`, `description.pt`, `numerical.pt`, `categorical.pt`, `labeled_status.npy`, `train.pt`, `val.pt`, `test.pt`. | BIC consumes tweet-sequence status tensors for text-graph interaction. Replacing them with averaged `tweets_tensor.pt` would change the method. |
| SEBot | Official author `trees/twibot-20_4.pickle` and `subgraphs/twibot-20_4.pickle` remain unavailable, but a runtime-only labeled-only adapted diagnostic has completed. | Treat the SEBot TwiBot-20 result as adapted diagnostic evidence, not official artifact reproduction, because it regenerates structural-entropy artifacts from the labeled-only bridge and uses the official `main_twibot20.py` sequential 70/20/10 split instead of the canonical TwiBot-20 split. |

Server artifact search on 2026-06-15 found no hidden copies of the RGT required files, BIC tweet-sequence/status files, or SEBot tree/subgraph pickles under `/root/workspace`, `/data`, or `/data3`.
The only relevant feature tensors found were the existing TwiBot-20 shared tensors under `/root/workspace/LMbot/processed_data` and the BotBR reproduction dataset.

Additional SEBot artifact check on 2026-06-15:

- SEBot README provides SharePoint links for datasets, encoding trees, and subgraphs.
- Server-side `requests` access to those links returned HTML OneDrive pages rather than file streams; direct folder `download.aspx?SourceUrl=...` returned `403 FORBIDDEN`.
- OneDrive public share API attempts returned `userContentMigrated` for `api.onedrive.com` and `InvalidAuthenticationToken` for Microsoft Graph without an access token.
- Therefore the author-provided SEBot artifacts were not downloadable non-interactively from the server in the current environment.
- A runtime-only SEBot `TwiBot-22-sampled proxy` artifact-generation diagnostic was launched at `/root/workspace/LMbot/repro_baselines_20260614/workdirs/sebot_twibot22_sampled_proxy_runtime` using local workdir `pydeps` for `numba==0.56.4`; this does not edit the official SEBot clone and is not an official full TwiBot-22 reproduction. The first artifact-generation PID `41088` ended without required pickles. The first fixed retry created the node-level tree but failed near the end of subgraph generation because PyG inferred one fewer node for `k_hop_subgraph`; the runtime workdir was patched to call `k_hop_subgraph(..., num_nodes=label.size(0))`, and `/root/workspace/LMbot/repro_baselines_20260614/workdirs/sebot_twibot22_sampled_proxy_runtime/resume_subgraphs_fixed_num_nodes_20260615.py` completed artifact generation at `2026-06-15 03:55:55 +0000`. The final artifact summary is `/root/workspace/LMbot/repro_baselines_20260614/results/sebot_twibot22_sampled_proxy_runtime_artifacts_20260615/artifact_summary_fixed.json`; tree size is `10113917` bytes and subgraph size is `46159228` bytes. A 5-seed proxy classification run was then launched from `/root/workspace/LMbot/repro_baselines_20260614/run_sebot_twibot22_sampled_proxy_5seed_20260615.sh`; runner PID `50479` started seed 1.
- A runtime-only SEBot `TwiBot-20 labeled-only adapted` artifact bridge completed at `/root/workspace/LMbot/repro_baselines_20260614/results/sebot_twibot20_labeled_only_runtime_artifacts_20260615/artifact_summary.json`. The first generated `subgraphs/twibot-20_4.pickle` contained raw keys `G/index/label/nodelist/tree`; before training it was converted with the official internal `SEPG.data.load_tree('twibot-20', 4)` path, producing the expected `edges/graph_mats/label/node_degrees/node_features/node_size/nodelist` fields. The 5-seed adapted diagnostic run completed from `/root/workspace/LMbot/repro_baselines_20260614/run_sebot_twibot20_labeled_only_runtime_5seed_20260615.sh`; its summary at `/root/workspace/LMbot/repro_baselines_20260614/results/sebot_twibot20_labeled_only_runtime_5seed_20260615/summary.json` reports Accuracy `0.8629 +/- 0.0036`, binary F1 `0.8772 +/- 0.0030`, binary recall `0.9056 +/- 0.0094`, and binary precision `0.8507 +/- 0.0086`. Macro-F1 was not emitted by this adapted run.

An RGT adapted-preprocessing run was launched on 2026-06-15:

- Data bridge: `/root/workspace/LMbot/repro_baselines_20260614/adapted_data/rgt_twibot20`.
- Code workdir: `/root/workspace/LMbot/repro_baselines_20260614/workdirs/rgt_twibot20_adapted`.
- Log: `/root/workspace/LMbot/repro_baselines_20260614/logs/rgt_twibot20_adapted_5seed.master.log`.
- Output: `/root/workspace/LMbot/repro_baselines_20260614/results/rgt_twibot20_adapted_5seed`.
- Scope label: adapted preprocessing, not official artifact reproduction.
- Adaptation details: shared BotRGCN-style TwiBot-20 tensors were converted to RGT filenames; numeric features were padded from 5 to 7 dimensions, categorical features from 3 to 11 dimensions, tweet/description tensors were reshaped to `[N, 1, 768]`, and edges were restricted to labeled-labeled users because RGT official code constructs features only for the 11826 labeled users.
- Current status: completed. The 5-seed adapted summary reports Accuracy `0.8639 +/- 0.0025`, Macro-F1 `0.8620 +/- 0.0026`, and binary F1 `0.8782 +/- 0.0023`.

An additional BIC adapted-preprocessing diagnostic run was launched on 2026-06-15:

- Code workdir: `/root/workspace/LMbot/repro_baselines_20260614/workdirs/bic_twibot20_adapted`.
- Log: `/root/workspace/LMbot/repro_baselines_20260614/logs/bic_twibot20_adapted_5seed.master.log`.
- Output: `/root/workspace/LMbot/repro_baselines_20260614/results/bic_twibot20_adapted_5seed`.
- Scope label: adapted preprocessing / proxy diagnostic, not official artifact reproduction.
- Adaptation details: the available pooled `tweets_tensor.pt` is wrapped as a length-1 tweet/status sequence for BIC's text module; the graph is restricted to labeled-labeled users; numerical features are padded from 5 to 6 dimensions. This diagnostic tests the BIC architecture family under shared TwiBot-20 tensors, but it does not reproduce BIC's original tweet-sequence preprocessing.
- Current status: completed. At the 2026-06-15 02:39 CST check, the 5-seed `summary.json` existed and reported Accuracy `0.8555 +/- 0.0050`, Macro-F1 `0.8534 +/- 0.0051`, and binary F1 `0.8707 +/- 0.0047`.
- Seed metrics:
  - Seed 1: Accuracy `0.8521`, Macro-F1 `0.8491`, binary F1 `0.8703`.
  - Seed 2: Accuracy `0.8563`, Macro-F1 `0.8544`, binary F1 `0.8710`.
  - Seed 3: Accuracy `0.8529`, Macro-F1 `0.8511`, binary F1 `0.8676`.
  - Seed 4: Accuracy `0.8512`, Macro-F1 `0.8496`, binary F1 `0.8652`.
  - Seed 5: included in the summary rows; see `/root/workspace/LMbot/repro_baselines_20260614/results/bic_twibot20_adapted_5seed/summary.json` for full history.

## TwiBot-22 Status

Current server evidence:

- Raw TwiBot-22 exists at `/root/workspace/LMbot/datasets/TwiBot-22`.
- Sampled TwiBot-22 exists at `/root/workspace/LMbot/datasets/TwiBot-22-sampled`, with `labels.pt`, split indices, and graph tensors.
- Official baseline feature/preprocessing outputs for TwiBot-22 are missing for these repositories.
- Raw TwiBot-22 currently contains `edge.csv`, `label.csv`, `split.csv`, `tweet_0.json`, `hashtag.json`, `list.json`, and `readme.md`; it does not contain `user.json`, `node.json`, or `tweet_1.json` through `tweet_8.json`.
- The raw full files are large enough to be the full labeled graph skeleton: `label.csv` has `1,000,000` data rows, `split.csv` has `1,000,000` data rows, and `edge.csv` has `170,185,937` data rows. However, the user-profile file and eight tweet shards required by the documented TwiBot-22 structure are absent.
- The official TwiBot-22 README says the dataset contains `tweet_i.json` for `i=0..8` and `user.json`. BIC's included `TwiBot22/data-process/preprocess_1.py` also directly reads `user.json` and loops over `tweet_0.json` through `tweet_8.json`. Therefore the current server raw directory is insufficient for official TwiBot-22 preprocessing.
- No ready user-profile/node feature tensor directory was found for the six official baselines.
- The sampled TwiBot-22 directory is a project-local robust sample, not a full official TwiBot-22 baseline-preprocessing output.

| Method | TwiBot-22 support status |
| --- | --- |
| BotRGCN | README mentions a pre-trained TwiBot-22 weight, but no full ready-to-run TwiBot-22 training data pipeline is present on the server. |
| RGT | No ready TwiBot-22 path found; official code expects Google Drive-style preprocessed files. |
| BotMoE | Repository has TwiBot-22 references, but `/data3/whr/lyh/MoE/mixture-of-experts/BotRGCN/twibot_22/processed_data` is missing. |
| BIC | Repository contains `TwiBot22/data-process/*`, but generated `BIC/TwiBot22/data` artifacts are missing. |
| SEBot | No ready TwiBot-22 support path found in the official code. |
| BotBR | Current official code supports TwiBot-20 and MGTAB-style commands; no ready TwiBot-22 support path found. |

Conclusion: do not launch TwiBot-22 official-code baseline runs from the current server state. They require a separate preprocessing/adaptation task and should be labeled adapted/proxy unless they reproduce each repository's documented preprocessing contract.

If TwiBot-22 numbers are required in the same table, the safest layout is:

- Main official-reproduction table: TwiBot-20 only for rows with verified official-code data.
- TwiBot-22 diagnostic/proxy table: explicitly labeled adapted runs on `TwiBot-22-sampled`, using a common feature extractor, not official repository preprocessing.
- Paper-reported table: include any TwiBot-22 results reported by the original papers only if the paper actually evaluates TwiBot-22 and the dataset/split matches.

Current decision for 2026-06-15:

- Do not launch an official TwiBot-22 run for any of the six baselines from the current server state.
- A shared `TwiBot-22-sampled proxy` feature protocol was defined and launched for BotRGCN, RGT, and BIC only.
- The proxy feature protocol uses `/root/workspace/LMbot/datasets/TwiBot-22-sampled-robust-v1`, converts one-hot labels with `argmax`, reuses the existing unified RoBERTa embeddings from `/root/workspace/LMbot/LLMbot/experiments/twibot22_robust_v1__roberta_rgcn__seed1/seed_1/preparation/semantic_encoder/embeddings.pt` as both `des_tensor.pt` and `tweets_tensor.pt`, parses profile metadata from `norm_user_text.json`, and writes explicit proxy manifests.
- The proxy run is not an official TwiBot-22 reproduction and must be reported in a separate diagnostic table named `TwiBot-22-sampled proxy`.
- BotMoE, BotBR, and SEBot were later added as runtime-only `TwiBot-22-sampled proxy` diagnostics after method-specific bridges over the common proxy tensors or generated proxy artifacts. This does not change the official full TwiBot-22 blocked status.
- As of the final 2026-06-15 check, all six methods have `TwiBot-22-sampled proxy` diagnostic evidence, while full official TwiBot-22 remains blocked by missing method-specific processed artifacts and incomplete raw files (`user.json` and `tweet_1..8.json` absent).
- New sampling protocol note: `scripts/twibot22_sample_official_prior.py` now defines `TwiBot-22-official-prior-sampled-v1`. It should be the next sampled TwiBot-22 dataset used for formal baseline reruns after raw data is synced to a machine with complete `user.json` and `tweet_0..8.json` shards.

## TwiBot-22-Sampled Proxy Run

Status on 2026-06-15:

- Script uploaded to server: `/root/workspace/LMbot/repro_baselines_20260614/run_twibot22_sampled_proxy_server.sh`.
- Launcher: `/tmp/remote_launch_twibot22_sampled_proxy.sh`.
- Log: `/root/workspace/LMbot/repro_baselines_20260614/logs/twibot22_sampled_proxy_5seed.master.log`.
- PID file: `/root/workspace/LMbot/repro_baselines_20260614/logs/twibot22_sampled_proxy_5seed.pid`.
- Current PID at launch: `35678`.
- Common proxy data: `/root/workspace/LMbot/repro_baselines_20260614/proxy_data/twibot22_sampled_common`.
- Common manifest: `/root/workspace/LMbot/repro_baselines_20260614/proxy_data/twibot22_sampled_common/manifest.json`.
- Planned result roots:
  - `/root/workspace/LMbot/repro_baselines_20260614/results/botrgcn_twibot22_sampled_proxy_5seed`.
  - `/root/workspace/LMbot/repro_baselines_20260614/results/rgt_twibot22_sampled_proxy_5seed`.
  - `/root/workspace/LMbot/repro_baselines_20260614/results/bic_twibot22_sampled_proxy_5seed`.

Current proxy results:

| Method | Scope | Status | Accuracy | Macro-F1 | Other metric | Evidence |
| --- | --- | --- | ---: | ---: | ---: | --- |
| BotRGCN | `TwiBot-22-sampled proxy`, 5 seeds | Completed | `0.8920 +/- 0.0042` | `0.6918 +/- 0.0152` | binary F1 `0.4434 +/- 0.0284` | `/root/workspace/LMbot/repro_baselines_20260614/results/botrgcn_twibot22_sampled_proxy_5seed/summary.json` |
| RGT | `TwiBot-22-sampled proxy`, 5 seeds | Completed | `0.8948 +/- 0.0060` | `0.6833 +/- 0.0291` | binary F1 `0.4246 +/- 0.0556` | `/root/workspace/LMbot/repro_baselines_20260614/results/rgt_twibot22_sampled_proxy_5seed/summary.json` |
| BIC | `TwiBot-22-sampled proxy`, 5 seeds | Completed proxy diagnostic | `0.8916 +/- 0.0029` | `0.7092 +/- 0.0210` | binary F1 `0.4789 +/- 0.0429` | `/root/workspace/LMbot/repro_baselines_20260614/results/bic_twibot22_sampled_proxy_5seed/summary.json` |
| BotBR | `TwiBot-22-sampled proxy`, 5 seeds | Completed runtime-only proxy diagnostic | `0.8810 +/- 0.0010` | not logged as macro-F1 | binary F1 `0.1719 +/- 0.1841` | `/root/workspace/LMbot/repro_baselines_20260614/results/botbr_twibot22_sampled_proxy_5seed/summary.json` |
| BotMoE | `TwiBot-22-sampled proxy`, 5 seeds | Completed runtime-only proxy diagnostic | `0.8269 +/- 0.0259` | not logged as macro-F1 | binary F1 `0.3485 +/- 0.1173` | `/root/workspace/LMbot/repro_baselines_20260614/results/botmoe_twibot22_sampled_proxy_runtime_adapter_5seed/summary.json` |
| SEBot | `TwiBot-22-sampled proxy`, 5 seeds | Completed runtime-only proxy diagnostic after `oomfix` | `0.8793 +/- 0.0029` | not logged as macro-F1 | binary F1 `0.1892 +/- 0.1187` | `/root/workspace/LMbot/repro_baselines_20260614/results/sebot_twibot22_sampled_proxy_runtime_5seed_oomfix_20260615/summary.json` |

Current BIC proxy note:

- On the 2026-06-15 06:23 CST check, the BIC proxy worker had completed 5 seeds and wrote `/root/workspace/LMbot/repro_baselines_20260614/results/bic_twibot22_sampled_proxy_5seed/summary.json`. The completed 5-seed result is Accuracy `0.8916 +/- 0.0029`, Macro-F1 `0.7092 +/- 0.0210`, and binary F1 `0.4789 +/- 0.0429`.
- The inspected BIC training path uses `NeighborLoader(num_neighbors=[-1] * 4)`, so this diagnostic proxy was much slower than BotRGCN and RGT on the 343,605-edge sampled graph.
- A separate bounded-neighbor diagnostic completed without stopping the exact proxy: `/root/workspace/LMbot/repro_baselines_20260614/results/bic_twibot22_sampled_proxy_bounded_neighbor_diag_seed1_20260615/summary.json`. It uses `num_neighbors=[15, 10]`, seed 1 only, and is not a replacement for exact BIC. Its seed 1 diagnostic result is Accuracy `0.8943`, Macro-F1 `0.7308`, and binary F1 `0.5211`.
- A micro smoke subset diagnostic completed at `/root/workspace/LMbot/repro_baselines_20260614/results/bic_twibot22_sampled_proxy_smoke_subset_diag_20260615/summary.json`, proving the adapted BIC code path can complete. It used only 256 train / 128 valid / 128 test nodes, `num_neighbors=[5, 5]`, and `max_epochs=2`, so it must not be reported as baseline performance.

Current BotBR proxy note:

- On the 2026-06-15 04:59 CST check, the BotBR `TwiBot-22-sampled proxy` worker completed from `/root/workspace/LMbot/repro_baselines_20260614/workdirs/botbr_twibot22_sampled_proxy/run_twibot22_sampled_proxy_5seed.sh`.
- Runtime-only data bridge: `/root/workspace/LMbot/repro_baselines_20260614/proxy_data/twibot22_sampled_botbr`.
- Result root: `/root/workspace/LMbot/repro_baselines_20260614/results/botbr_twibot22_sampled_proxy_5seed`.
- Manifest: `/root/workspace/LMbot/repro_baselines_20260614/results/botbr_twibot22_sampled_proxy_5seed/manifest.json`.
- Summary: `/root/workspace/LMbot/repro_baselines_20260614/results/botbr_twibot22_sampled_proxy_5seed/summary.json`.
- 5-seed proxy diagnostic result: Accuracy `0.8810 +/- 0.0010`; binary F1 `0.1719 +/- 0.1841`.
- Macro-F1 is not available from this runtime proxy because the BotBR official script/log reports accuracy, binary precision, binary recall, and binary F1 only.
- This run reuses BotBR's `TwiBot20` loader class against a BotBR-shaped directory containing the common TwiBot-22-sampled proxy tensors. It is a proxy diagnostic only and must not be reported as official BotBR TwiBot-22.

Current BotMoE proxy note:

- On the 2026-06-15 05:26 CST check, the BotMoE `TwiBot-22-sampled proxy` worker completed 5 seeds from a runtime-only workdir at `/root/workspace/LMbot/repro_baselines_20260614/workdirs/botmoe_twibot22_sampled_proxy`.
- Result root: `/root/workspace/LMbot/repro_baselines_20260614/results/botmoe_twibot22_sampled_proxy_runtime_adapter_5seed`.
- Manifest: `/root/workspace/LMbot/repro_baselines_20260614/results/botmoe_twibot22_sampled_proxy_runtime_adapter_5seed/manifest.json`.
- Summary: `/root/workspace/LMbot/repro_baselines_20260614/results/botmoe_twibot22_sampled_proxy_runtime_adapter_5seed/summary.json`.
- 5-seed proxy diagnostic result: Accuracy `0.8269 +/- 0.0259`; binary F1 `0.3485 +/- 0.1173`.
- Macro-F1 is not available from this runtime proxy because the current BotMoE adapter logs accuracy, binary precision, binary recall, and binary F1 only.
- This run uses the shared TwiBot-22-sampled proxy tensor protocol and a BotMoE-shaped runtime adapter. It is a proxy diagnostic only and must not be reported as official full BotMoE TwiBot-22.

Current SEBot proxy classification note:

- The first SEBot `TwiBot-22-sampled proxy` 5-seed diagnostic was launched from `/root/workspace/LMbot/repro_baselines_20260614/run_sebot_twibot22_sampled_proxy_5seed_20260615.sh`, but failed at seed 1 epoch 43 with CUDA OOM in the full-graph InfoNCE denominator allocation. The failed evidence remains in `/root/workspace/LMbot/repro_baselines_20260614/results/sebot_twibot22_sampled_proxy_runtime_5seed_20260615/seed_1.log`, and no `summary.json` was written.
- A runtime-only recovery rerun completed from `/root/workspace/LMbot/repro_baselines_20260614/run_sebot_twibot22_sampled_proxy_5seed_oomfix_20260615.sh`, writing `/root/workspace/LMbot/repro_baselines_20260614/results/sebot_twibot22_sampled_proxy_runtime_5seed_oomfix_20260615/summary.json`. It reports Accuracy `0.8793 +/- 0.0029`, binary F1 `0.1892 +/- 0.1187`, binary recall `0.1305 +/- 0.0935`, and binary precision `0.5803 +/- 0.2351`; Macro-F1 was not emitted. This does not edit the official SEBot clone. The workdir-only fixes are: wrap eval in `torch.no_grad()`, remove an extra `return_attention=True` backbone call used only when logging high test accuracy, release per-epoch references, call `torch.cuda.empty_cache()`, and set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`. The claim boundary remains `TwiBot-22-sampled proxy diagnostic`, not official full TwiBot-22.

The common proxy manifest printed at launch records:

- `label_shape`: `[11826]`.
- `label_counts`: `[10171, 1655]`.
- `edge_index_shape`: `[2, 343605]`.
- `edge_type_counts`: relation `0`: `69374`, relation `1`: `274231`.
- `split_sizes`: train `8278`, valid `2365`, test `1183`.
- `text_proxy_note`: both `des_tensor.pt` and `tweets_tensor.pt` reuse the unified RoBERTa embedding, so this does not reproduce per-method official TwiBot-22 text preprocessing.

## TwiBot-22 Sampling Audit, 2026-06-15

This audit separates the historical sampled proxy from the newer official-prior sample.
The two datasets have the same total labeled size (`11826`) and overall bot count (`1655`),
but very different split-level label priors.

| Dataset | Split | Human | Bot | Bot ratio | Majority-human baseline |
| --- | --- | ---: | ---: | ---: | ---: |
| `TwiBot-22-sampled` / `TwiBot-22-sampled-robust-v1` | train | `7089` | `1189` | `0.1436` | `0.8564` |
| `TwiBot-22-sampled` / `TwiBot-22-sampled-robust-v1` | valid | `2040` | `325` | `0.1374` | `0.8626` |
| `TwiBot-22-sampled` / `TwiBot-22-sampled-robust-v1` | test | `1042` | `141` | `0.1192` | `0.8808` |
| `TwiBot-22-official-prior-sampled-v1` | train | `7632` | `646` | `0.0780` | `0.9220` |
| `TwiBot-22-official-prior-sampled-v1` | valid | `1704` | `661` | `0.2795` | `0.7205` |
| `TwiBot-22-official-prior-sampled-v1` | test | `835` | `348` | `0.2942` | `0.7058` |
| Full official `TwiBot-22` | train | `645414` | `54586` | `0.0780` | `0.9220` |
| Full official `TwiBot-22` | val | `144087` | `55913` | `0.2796` | `0.7204` |
| Full official `TwiBot-22` | test | `70556` | `29444` | `0.2944` | `0.7056` |

Conclusion:

- The historical `TwiBot-22-sampled` / `TwiBot-22-sampled-robust-v1` still has a sampling problem for method comparison: test bot ratio is only `0.1192`, so a majority-human classifier already reaches `0.8808` accuracy. Accuracy near `0.88-0.895` on this sample is therefore inflated and must be treated as diagnostic/proxy evidence only.
- `TwiBot-22-official-prior-sampled-v1` fixes the split-prior mismatch relative to full TwiBot-22: train, validation, and test bot ratios match the official split priors. It should replace the old sampled proxy for future sampled TwiBot-22 comparisons.
- Full official TwiBot-22 itself has a strong train/test prior shift: train bot ratio is `0.0780`, while validation/test bot ratios are about `0.2796/0.2944`. Lower accuracy on the official-prior sample is expected and is more meaningful than inflated old-proxy accuracy.

Completed sampled TwiBot-22 diagnostic results:

| Method | Dataset | Scope | Accuracy | Macro-F1 | Binary F1 | Evidence |
| --- | --- | --- | ---: | ---: | ---: | --- |
| BotRGCN | old sampled proxy | 5 seeds | `0.8920 +/- 0.0042` | `0.6918 +/- 0.0152` | `0.4434 +/- 0.0284` | `/root/workspace/LMbot/repro_baselines_20260614/results/botrgcn_twibot22_sampled_proxy_5seed/summary.json` |
| RGT | old sampled proxy | 5 seeds | `0.8948 +/- 0.0060` | `0.6833 +/- 0.0291` | `0.4246 +/- 0.0556` | `/root/workspace/LMbot/repro_baselines_20260614/results/rgt_twibot22_sampled_proxy_5seed/summary.json` |
| BIC | old sampled proxy | 5 seeds | `0.8916 +/- 0.0029` | `0.7092 +/- 0.0210` | `0.4789 +/- 0.0429` | `/root/workspace/LMbot/repro_baselines_20260614/results/bic_twibot22_sampled_proxy_5seed/summary.json` |
| BotBR | old sampled proxy | runtime adapter, 5 seeds | `0.8810 +/- 0.0010` | not emitted | `0.1719 +/- 0.1841` | `/root/workspace/LMbot/repro_baselines_20260614/results/botbr_twibot22_sampled_proxy_5seed/summary.json` |
| BotMoE | old sampled proxy | runtime adapter, 5 seeds | `0.8269 +/- 0.0259` | not emitted | `0.3485 +/- 0.1173` | `/root/workspace/LMbot/repro_baselines_20260614/results/botmoe_twibot22_sampled_proxy_runtime_adapter_5seed/summary.json` |
| SEBot | old sampled proxy | runtime adapter, 5 seeds, OOM-fix workdir | `0.8793 +/- 0.0029` | not emitted | `0.1892 +/- 0.1187` | `/root/workspace/LMbot/repro_baselines_20260614/results/sebot_twibot22_sampled_proxy_runtime_5seed_oomfix_20260615/summary.json` |
| BotRGCN | official-prior sample | profile-only diagnostic, 5 seeds | `0.7221 +/- 0.0229` | `0.6055 +/- 0.0210` | `0.3919 +/- 0.0411` | `/root/workspace/LMbot/repro_baselines_20260614/results/botrgcn_twibot22_official_prior_profileonly_5seed/summary.json` |
| RGT | official-prior sample | profile-only diagnostic, 5 seeds | `0.7224 +/- 0.0238` | `0.5673 +/- 0.0691` | `0.3090 +/- 0.1292` | `/root/workspace/LMbot/repro_baselines_20260614/results/rgt_twibot22_official_prior_profileonly_5seed/summary.json` |
| BIC | official-prior sample | profile-only diagnostic, 5 seeds | `0.7856 +/- 0.0097` | `0.7342 +/- 0.0095` | `0.6173 +/- 0.0135` | `/root/workspace/LMbot/repro_baselines_20260614/results/bic_twibot22_official_prior_profileonly_5seed/summary.json` |
| BotBR | official-prior sample | profile-only diagnostic, 5 seeds | `0.7609 +/- 0.0446` | not emitted | `0.3551 +/- 0.2533` | `/root/workspace/LMbot/repro_baselines_20260614/results/botbr_twibot22_official_prior_profileonly_5seed/summary.json` |
| BotMoE | official-prior sample | profile-only runtime adapter, 5 seeds | `0.7250 +/- 0.0039` | not emitted | `0.5669 +/- 0.0111` | Parsed from `/root/workspace/LMbot/repro_baselines_20260614/results/botmoe_twibot22_official_prior_profileonly_runtime_adapter_5seed/AllInOne1_rgcn_rgt_gcn.log`; summary JSON lacks parsed metric fields. |

Current run status:

- No sampled TwiBot-22 training process was active at the 2026-06-15 server check.
- Unrelated active processes were a TwiBot-20 HyperScan diagnostic and the official TwiBot-22 feature-baseline FriendBot code running on TwiBot-20.
- On 2026-06-18, `scripts/run_twibot22_official_feature_baselines_sampled.py` was added for BotHunter/FriendBot on `TwiBot-22-official-prior-sampled-v1`. Local validation passed for `py_compile`, adapter construction, id-format checks, and a BotHunter 1-seed smoke on the corrected prepared sample with `max_tweets_per_user=2`.
- A local 5-seed BotHunter run completed at `G:\Research\BotDetection\repro_baselines_20260614\results\twibot22_official_prior_sampled_feature_baselines_20260618\bothunter\summary.json`.
- FriendBot is running locally from the same adapter because SSH is blocked. The first local FriendBot attempt was interrupted after diagnosing a harness issue: long-running subprocess output was captured with `stdout=PIPE`, which can block official `tqdm`-heavy feature extraction. The harness now streams child stdout/stderr directly to the method log while preserving captured stdout only for short commands that need it, such as `git rev-parse`.
- FriendBot is intentionally run on CPU. The packaged official method is profile/text/ego-network feature engineering plus sklearn RandomForest, with the slow path in NetworkX/Louvain/effective-size style features; moving it to GPU would require rewriting the official baseline rather than reproducing it.
- The intended server run is blocked only by SSH protocol availability: TCP to `172.31.106.108:10011` succeeds, but SSH times out during banner exchange before command execution. Do not claim sampled BotHunter/FriendBot server results until the result root below exists.

Sampled TwiBot-22 BotHunter/FriendBot adapter command, once SSH is available:

```bash
cd /root/workspace/LMbot/repro_baselines_20260614
mkdir -p logs
nohup bash -lc 'source /root/mambaforge/etc/profile.d/conda.sh && conda activate lmbot && python run_twibot22_official_feature_baselines_sampled.py \
  --repo /root/workspace/LMbot/tmp_external/TwiBot-22 \
  --prepared-root /root/workspace/LMbot/datasets/TwiBot-22-official-prior-sampled-v1 \
  --run-root /root/workspace/LMbot/repro_baselines_20260614 \
  --methods bothunter friendbot \
  --seeds 0 100 200 300 400 \
  --max-tweets-per-user 50' \
  > logs/twibot22_official_prior_sampled_feature_baselines_20260618.master.log 2>&1 &
echo $! > logs/twibot22_official_prior_sampled_feature_baselines_20260618.pid
```

Expected result root:

```text
/root/workspace/LMbot/repro_baselines_20260614/results/twibot22_official_prior_sampled_feature_baselines_20260618
```

## Commands And Audit Artifacts

- Server audit output: `.tmp/baseline_server_audit.out`.
- TwiBot-22 support audit output: `.tmp/twibot22_support_audit.out`.
- Server artifact search output: `.tmp/server_artifact_search.out`.
- Raw dataset audit output: `.tmp/server_raw_dataset_audit.out`.
- BotRGCN single-run log: `/root/workspace/LMbot/repro_baselines_20260614/logs/botrgcn_twibot20_official_seed_default.log`.
- BotRGCN 5-seed log: `/root/workspace/LMbot/repro_baselines_20260614/logs/botrgcn_twibot20_official_5seed.master.log`.
- BotRGCN 5-seed summary: `/root/workspace/LMbot/repro_baselines_20260614/results/botrgcn_twibot20_official_5seed/summary.json`.

## Comparison Table Guidance

For the main paper table:

- Use official-code reproduced rows where available.
- Report Accuracy and Macro-F1 when both exist.
- If only binary F1 exists in reused logs, either rerun under macro-F1 logging or mark the value as binary F1 in an auxiliary column.
- Keep paper-reported results separate from server-reproduced results.
- Do not mix TwiBot-22 sampled results with full TwiBot-22 official results.
