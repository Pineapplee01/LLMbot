# Mainline Reference Papers

This document records strong references for the active research pipeline. Each entry follows a motivation-method-result logic and states how the paper can or cannot be migrated into this project.

Weak candidates and non-mainline alternatives belong in `docs/candidates.md`, not here.

## Base LM+GNN

### SimTeG: A Frustratingly Simple Approach Improves Textual Graph Learning

- **Motivation.** Text-attributed graph learning often focuses on GNN design while using weak or generic text features.
- **Problem analysis.** If text embeddings are poor, graph learning is bottlenecked before message passing starts; heavy joint LM-GNN training is expensive.
- **Mechanism / method.** Finetune or PEFT an LM on the downstream task, extract hidden-state node embeddings, then train standard GNNs on those features.
- **Result / evidence.** Reports broad gains across textual graph benchmarks while keeping the framework simple.
- **Boundary.** It is a base pipeline reference, not the novelty. It does not provide a post-hoc graph-quality estimator or local rewrite mechanism.
- **Migration to this project.** Use as the Stage-0 / frozen-G0 preparation logic: finetuned LM hidden states become GNN node inputs.
- **Links.** Paper: [OpenReview](https://openreview.net/forum?id=EFGwiZ2pAW), [arXiv](https://arxiv.org/abs/2308.02565). Official repo: not verified.

## Conformal / Graph-Aware Uncertainty

### CF-GNN: Uncertainty Quantification over Graph with Conformalized Graph Neural Networks

- **Motivation.** GNN predictions lack rigorous uncertainty estimates, especially when errors are costly.
- **Problem analysis.** Standard conformal prediction ignores graph dependence and may produce inefficient prediction sets on graph data.
- **Mechanism / method.** Extends conformal prediction to graph-based models and learns topology-aware output correction to improve uncertainty sets.
- **Result / evidence.** Provides conformalized GNN prediction sets/intervals and reports more useful uncertainty estimates.
- **Boundary.** We do not claim CF-GNN coverage theorems after local graph rewriting or under unverified exchangeability assumptions.
- **Migration to this project.** Use prediction-set interface, calibration split discipline, and set-size/margin quality signals.
- **Links.** Paper: [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/54a1495b06c4ee2f07184afb9a37abda-Abstract-Conference.html), [arXiv](https://arxiv.org/abs/2305.14535). Repo: [snap-stanford/conformalized-gnn](https://github.com/snap-stanford/conformalized-gnn).

### DAPS/NAPS: Conformal Prediction Sets for Graph Neural Networks

- **Motivation.** Prediction sets for graph nodes should use local graph information to improve efficiency.
- **Problem analysis.** Neighboring nodes and graph diffusion can help calibrate node-level uncertainty, but direct i.i.d. assumptions are weak on graphs.
- **Mechanism / method.** Aggregates nonconformity information with graph diffusion / neighborhood-aware mechanisms.
- **Result / evidence.** Improves prediction-set efficiency for graph node classification while preserving conformal evaluation goals.
- **Boundary.** The original method is not a social-bot ego repair algorithm, and homophily assumptions may fail under camouflage/heterophily.
- **Migration to this project.** Use local 2-hop nonconformity aggregation as a quality proxy, not a full DAPS/NAPS reproduction.
- **Links.** Paper: [PMLR ICML 2023](https://proceedings.mlr.press/v202/h-zargarbashi23a.html). Repo: [soroushzargar/DAPS](https://github.com/soroushzargar/DAPS).

### SNAPS: Similarity-Navigated Conformal Prediction for Graph Neural Networks

- **Motivation.** Conformal prediction sets should be compact while retaining valid coverage.
- **Problem analysis.** Nodes with high feature similarity or structural proximity can provide useful nonconformity context.
- **Mechanism / method.** Aggregates nonconformity scores using feature similarity and structural neighborhood.
- **Result / evidence.** Improves prediction-set efficiency and singleton hit ratio while targeting valid marginal coverage.
- **Boundary.** Similarity is not edge reliability in social bot graphs; semantic similarity can be camouflage or coordination evidence.
- **Migration to this project.** Use RoBERTa/SimTeG similarity and 2-hop graph context as estimator/retrieval signals with explicit boundaries.
- **Links.** Paper: [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/571c7e164fb1ffbcf2f84a63784451ec-Abstract-Conference.html), [arXiv](https://arxiv.org/abs/2405.14303). Repo: [janqsong/SNAPS](https://github.com/janqsong/SNAPS).

### Conformal Inductive Graph Neural Networks

- **Motivation.** Graph conformal methods must handle inductive settings where new nodes or edges appear after calibration.
- **Problem analysis.** New graph connectivity changes message passing and can shift the score distribution, so transductive graph conformal assumptions do not automatically transfer.
- **Mechanism / method.** Studies conformal prediction for inductive GNNs and isolates how calibration validity interacts with graph changes at test time.
- **Result / evidence.** Provides a high-level warning that graph updates and inductive message passing require explicit assumptions and evaluation.
- **Boundary.** It does not provide a social-bot local rewriter or relation-aware hard-node router.
- **Migration to this project.** Use as a guardrail: after local graph rewrite, the estimator score is a post-hoc quality proxy and rollback signal, not a renewed coverage theorem.
- **Links.** Paper: [OpenReview](https://openreview.net/forum?id=homn1jOKI5). Official repo: not verified.

### RR-GNN: Residual Reweighted Conformal Prediction for Graph Neural Networks

- **Motivation.** Standard graph conformal prediction can be conservative when graph heterogeneity and structural bias vary across nodes.
- **Problem analysis.** A single global score may miss cluster-specific uncertainty and residual difficulty; leakage can also occur when residual predictors reuse training information unsafely.
- **Mechanism / method.** Uses graph-structured Mondrian partitioning, residual-adaptive nonconformity scores, and a cross-training protocol to reduce leakage.
- **Result / evidence.** Reports improved prediction-set efficiency while preserving marginal coverage across multiple graph tasks.
- **Boundary.** It trains an additional residual model and is not a post-hoc SimTeG router implementation in this project.
- **Migration to this project.** Use the logic of regime-conditioned or residual-adaptive conformal scores to justify relation/direction-conditioned router channels selected on `valid_tune` and calibrated on `valid_cal`.
- **Links.** Paper: [arXiv](https://arxiv.org/abs/2506.07854). Official repo: not verified.

### CoRel: Relational Conformal Prediction for Correlated Time Series

- **Motivation.** Conformal prediction for correlated entities should exploit relational dependencies instead of treating samples as independent.
- **Problem analysis.** Related series share uncertainty structure, and ignoring relational context can produce inefficient or misaligned intervals.
- **Mechanism / method.** Uses relational structure to improve conformal prediction for correlated time series.
- **Result / evidence.** Shows that relational dependencies can improve conformal uncertainty estimates in structured prediction settings.
- **Boundary.** It is not graph node classification and not a social-bot detector.
- **Migration to this project.** Use only the principle that relation structure can condition conformal uncertainty; implement relation-conditioned node nonconformity as a domain adaptation of graph conformal scoring.
- **Links.** Paper: [PMLR ICML 2025](https://proceedings.mlr.press/v267/cini25a.html). Repo: [andreacini/corel](https://github.com/andreacini/corel).

### GATS: What Makes Graph Neural Networks Miscalibrated?

- **Motivation.** GNN confidence can be poorly calibrated.
- **Problem analysis.** Miscalibration is affected by graph topology, distance to labeled nodes, neighborhood similarity, and prediction distribution diversity.
- **Mechanism / method.** Graph Attention Temperature Scaling learns node-specific calibration temperatures from graph-aware signals.
- **Result / evidence.** Improves calibration quality across graph benchmarks.
- **Boundary.** GATS is a scalar calibration baseline, not a prediction-set interface or ego rewriter.
- **Migration to this project.** Use as a calibration baseline and as evidence that graph structure matters for uncertainty.
- **Links.** Paper: [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/5975754c7650dfee0682e06e1fec0522-Abstract-Conference.html). Repo: [hans66hsu/GATS](https://github.com/hans66hsu/GATS).

### CaGCN: Be Confident! Towards Trustworthy Graph Neural Networks via Confidence Calibration

- **Motivation.** GNN node classifiers need trustworthy confidence estimates.
- **Problem analysis.** Simple post-hoc calibration is insufficient when node predictions are graph-dependent.
- **Mechanism / method.** Uses a graph convolutional calibration model to learn topology-aware confidence calibration.
- **Result / evidence.** Shows graph-aware calibration improves confidence quality.
- **Boundary.** It does not output conformal prediction sets and should not be the main router mechanism.
- **Migration to this project.** Use as graph-aware scalar calibration baseline.
- **Links.** Paper: [NeurIPS PDF](https://papers.nips.cc/paper/2021/file/c7a9f13a6c0940277d46706c7ca32601-Paper.pdf). Repo: [BUPT-GAMMA/CaGCN](https://github.com/BUPT-GAMMA/CaGCN).

### SimCalib: Graph Neural Network Calibration Based on Similarity between Nodes

- **Motivation.** Graph neural networks can remain miscalibrated even when their accuracy is high.
- **Problem analysis.** Similar nodes often share calibration behavior, so node similarity can help estimate whether confidence is reliable.
- **Mechanism / method.** Calibrates GNN predictions by using similarity relationships between nodes.
- **Result / evidence.** Reports improved GNN calibration on benchmark graph datasets.
- **Boundary.** Similarity-based calibration is not social-bot edge reliability; in bot graphs, semantic similarity can indicate camouflage or coordination rather than trust.
- **Migration to this project.** Use as support for similarity as an ablation or secondary router cue, while keeping relation/direction as the primary social-bot domain signal.
- **Links.** Paper: [AAAI Proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/29450). Official repo: not verified.

## Selective Routing / Risk Control

### Selective Classification for Deep Neural Networks

- **Motivation.** A model should be allowed to abstain or defer on examples where it is likely to be wrong.
- **Problem analysis.** Accuracy alone hides whether uncertainty ranking actually isolates high-risk examples.
- **Mechanism / method.** Formalizes selective prediction with risk-coverage tradeoffs and trains classifiers with a reject option.
- **Result / evidence.** Establishes risk-coverage curves as a standard evaluation lens for selective prediction.
- **Boundary.** It is not graph-specific and does not provide conformal prediction sets or social-bot ego refinement.
- **Migration to this project.** Evaluate the router by hard-node error recall, precision, lift, and budget curves, not by claiming that the router directly improves base Acc/F1.
- **Links.** Paper: [NeurIPS](https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks), [PDF](https://papers.neurips.cc/paper_files/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf). Official repo: not verified.

### SelectiveNet: A Deep Neural Network with an Integrated Reject Option

- **Motivation.** A classifier can be more useful when it learns both prediction and rejection behavior instead of applying only a confidence threshold after training.
- **Problem analysis.** Directly accepting every candidate correction is unsafe when candidate models can fix base-wrong examples but also break base-correct examples.
- **Mechanism / method.** Adds a learned reject option and optimizes the model over the covered domain with a risk-coverage objective.
- **Result / evidence.** Reports improved risk-coverage tradeoffs across classification and regression benchmarks.
- **Boundary.** The original method is an end-to-end selective network, not a post-hoc social-bot correction gate over frozen SimTeG plus LLM candidates.
- **Migration to this project.** Supports the accept/defer framing and validation-locked thresholding for `semantic_correction_gate`, while the candidate semantic predictors remain fixed rather than jointly trained.
- **Links.** Paper: [PMLR ICML 2019](https://proceedings.mlr.press/v97/geifman19a.html), [PDF](https://proceedings.mlr.press/v97/geifman19a/geifman19a.pdf). Code link is listed by PMLR but not locally verified.

### Consistent Estimators for Learning to Defer to an Expert

- **Motivation.** In practical systems, a learned model often works beside another decision maker, so the system must decide whether to predict or defer.
- **Problem analysis.** A candidate LLM/MLP can be interpreted as an auxiliary decision maker, while frozen SimTeG is the high-quality default expert. The core problem is choosing when to defer from the base to the candidate.
- **Mechanism / method.** Formalizes learning a predictor plus rejector/defer rule through a cost-sensitive reduction and a consistent surrogate loss.
- **Result / evidence.** Provides theoretical and empirical support for learning defer decisions instead of forcing a single predictor to own every example.
- **Boundary.** The paper is not graph-specific and assumes the expert-decision setting rather than a routed-node social-bot correction protocol.
- **Migration to this project.** Directly motivates `semantic_correction_gate`: train an accept/defer gate where accepting a candidate is rewarded only when it improves over the base and penalized when it breaks base-correct nodes.
- **Links.** Paper: [PMLR ICML 2020](https://proceedings.mlr.press/v119/mozannar20b.html), [PDF](https://proceedings.mlr.press/v119/mozannar20b/mozannar20b.pdf). Code link is listed by PMLR but not locally verified.

### META-DES: A Dynamic Ensemble Selection Framework using Meta-Learning

- **Motivation.** A fixed ensemble or globally best classifier can fail because different samples need different competent classifiers.
- **Problem analysis.** Routed nodes contain multiple candidate semantic predictors with different fix/break behavior. The selection question is a competence-estimation problem, not just another bot/human classifier.
- **Mechanism / method.** Constructs meta-features that describe classifier competence, then trains a meta-classifier to decide whether a base classifier should participate for a query instance.
- **Result / evidence.** Reports stronger dynamic ensemble selection than earlier single-criterion competence rules.
- **Boundary.** META-DES is a general pattern-recognition dynamic ensemble-selection framework; it does not contain LLM prompts, graph propagation, or a frozen SimTeG base detector.
- **Migration to this project.** Supports building gate inputs from base/candidate confidence, agreement, margin, entropy, and probability deltas, and it cautions that node-specific selection requires competence features that correlate with real correction utility. The `semantic_correction_gate --semantic_gate_feature_family node_attribute` path tests static target-node descriptors; the `local_competence` path is the closer migration of the region-of-competence idea because it appends per-candidate fix, break, net-utility, and support estimates from nearest routed train nodes.
- **Links.** Paper: [Pattern Recognition via DBLP](https://dblp.org/rec/journals/pr/CruzSCR15), [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0031320314004919), [author accepted manuscript](https://www.cin.ufpe.br/~gdcc/papers/2015-pr-cruz.pdf).

### Multicalibration: Calibration for the Computationally-Identifiable Masses

- **Motivation.** A predictor can look calibrated globally while being miscalibrated on important subgroups.
- **Problem analysis.** Routed nodes are a selected, dense, high-risk subset. A single global gate score or threshold can validate on one mixture of routed nodes and fail on another, especially when break-prone and fix-prone nodes share similar global attributes.
- **Mechanism / method.** Requires approximate calibration over many identifiable subpopulations rather than only in aggregate.
- **Result / evidence.** Establishes subgroup-aware calibration as a formal target and shows why global calibration is an insufficient deployment guarantee.
- **Boundary.** The original work is a calibration framework, not a social-bot detector, dynamic selector, or LLM refiner.
- **Migration to this project.** Supports moving from a probability-only or globally thresholded gate toward local/subgroup competence features. `local_competence` remains an exploratory approximation: it estimates candidate utility in routed-train neighborhoods instead of claiming formal multicalibration.
- **Links.** Paper: [PMLR ICML 2018](https://proceedings.mlr.press/v80/hebert-johnson18a.html), [PDF](https://proceedings.mlr.press/v80/hebert-johnson18a/hebert-johnson18a.pdf).

### Conformal Risk Control

- **Motivation.** Many deployment settings need control over task losses beyond simple label coverage.
- **Problem analysis.** Prediction-set coverage does not always align with operational risk, abstention cost, or downstream error budgets.
- **Mechanism / method.** Extends conformal methods to control user-defined risks under distributional assumptions.
- **Result / evidence.** Provides a framework for calibrating risk-controlling prediction sets.
- **Boundary.** It is a general conformal framework; this project should not overclaim formal risk control for rewritten graphs without a dedicated proof and experiment.
- **Migration to this project.** Use only as evaluation vocabulary for budgeted hard-node routing and accept/rollback risk, while keeping v1 claims empirical.
- **Links.** Paper: [arXiv](https://arxiv.org/abs/2208.02814). Repo: [aangelopoulos/conformal-risk](https://github.com/aangelopoulos/conformal-risk).

## Post-hoc Calibration / Reject Option / Localized CP

### Localized Conformal Prediction

- **Motivation.** Global conformal calibration can be too coarse when uncertainty varies across local regions.
- **Problem analysis.** One-size-fits-all calibration may miss heterogeneity, so the calibration object should adapt to the test point's local neighborhood.
- **Mechanism / method.** Introduces a localized conformal framework that weights or localizes calibration around the test sample.
- **Result / evidence.** Shows that local calibration can tighten prediction sets while retaining finite-sample marginal validity under suitable assumptions.
- **Boundary.** It is a general inference framework, not a graph-specific router and not a social-bot detector.
- **Migration to this project.** Use as the cleanest conceptual bridge for local calibration bins and neighborhood-conditioned ranking, without claiming a new theorem for rewrites.
- **Links.** Paper: [Biometrika](https://academic.oup.com/biomet/article/110/1/33/6647831), [arXiv](https://arxiv.org/abs/2106.08460).

### A Model-Agnostic Heuristics for Selective Classification

- **Motivation.** A reject option should work with any base classifier, not only end-to-end selective networks.
- **Problem analysis.** Training a separate selection head is model-specific and can be expensive; a cross-fit quantile view is simpler.
- **Mechanism / method.** Uses cross-fitting and quantile estimation to build a selection function for selective classification.
- **Result / evidence.** Reports that the model-agnostic selector can outperform existing methods on real-world data.
- **Boundary.** It is a general selective-classification method, not graph-specific and not a prediction-set theorem.
- **Migration to this project.** Supports the idea that the router can be a post-hoc calibration + ranking layer over frozen GNN outputs.
- **Links.** Paper: [AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/26133). DOI: [10.1609/aaai.v37i8.26133](https://doi.org/10.1609/aaai.v37i8.26133).

### Classification with reject option: Distribution-free error guarantees via conformal prediction

- **Motivation.** In binary classification, abstention is often better than forcing a wrong decision.
- **Problem analysis.** A reject option needs error guarantees, not only a heuristic threshold.
- **Mechanism / method.** Turns conformal prediction into a reject-option classifier by accepting singleton prediction sets and rejecting ambiguous ones.
- **Result / evidence.** Derives error-reject curves and finite-sample estimates for reject-option behavior.
- **Boundary.** The paper is binary and generic; it is not graph-specific and does not solve social-bot heterogeneity by itself.
- **Migration to this project.** Gives direct theoretical language for turning calibrated scores into accept/reject or rank/abstain decisions.
- **Links.** Paper: [Machine Learning with Applications](https://doi.org/10.1016/j.mlwa.2025.100664), [arXiv](https://arxiv.org/abs/2506.21802).

### Node Classification With Integrated Reject Option

- **Motivation.** GNN node classifiers should be able to abstain when uncertainty is high.
- **Problem analysis.** Reject-option behavior had not been well explored for node classification.
- **Mechanism / method.** Proposes cost-based and coverage-based abstention for GNN node classification.
- **Result / evidence.** Demonstrates node-level reject behavior on citation graphs and a legal citation node-classification setting.
- **Boundary.** It is graph-specific abstention, but not social-bot-specific and not a local rewrite method.
- **Migration to this project.** Strongly supports the router-as-abstention/ranking framing on graph nodes.
- **Links.** Paper: [OpenReview](https://openreview.net/forum?id=4xXJDO8Bvu), [arXiv](https://arxiv.org/abs/2412.03190). Official repo: not verified.

## Graph CP Benchmarks / Tooling

### Conformal Prediction: A Theoretical Note and Benchmarking Transductive Node Classification in Graphs

- **Motivation.** Graph conformal prediction had become fragmented across implementations and baselines.
- **Problem analysis.** The literature had conflicting design choices and unclear evaluation protocols.
- **Mechanism / method.** Benchmarks transductive node classification with conformal prediction and studies implementation tradeoffs.
- **Result / evidence.** Provides recommendations for future graph conformal scholarship and scaling guidance.
- **Boundary.** It is a benchmark and methodology note, not a new social-bot router.
- **Migration to this project.** Use it as direct support for keeping the router post-hoc, split-safe, and benchmarked by calibration/ranking metrics.
- **Links.** Paper: [OpenReview](https://openreview.net/forum?id=Ed1DBB3sBQ). Code: [pranavmaneriker/graphconformal-code](https://github.com/pranavmaneriker/graphconformal-code/tree/main).

### Benchmarking Graph Conformal Prediction: Empirical Analysis, Scalability, and Theoretical Insights

- **Motivation.** Graph conformal prediction was growing quickly but lacked a clean empirical and scalability benchmark.
- **Problem analysis.** Existing graph CP work used conflicting baselines, implementations, and assumptions.
- **Mechanism / method.** Analyzes existing graph CP design choices and scaling tradeoffs.
- **Result / evidence.** Provides empirical and theoretical guidance for future graph conformal work.
- **Boundary.** It is a benchmark paper, not a new estimator.
- **Migration to this project.** Supports treating router work as calibration/ranking under a benchmarked protocol rather than a new predictor.
- **Links.** Paper: [arXiv](https://arxiv.org/abs/2409.18332). 

### RoCP-GNN: Robust Conformal Prediction for Graph Neural Networks in Node-Classification

- **Motivation.** GNNs need robust uncertainty estimates under shift and graph dependence.
- **Problem analysis.** Graph CP must handle dependent node data and maintain useful prediction-set efficiency.
- **Mechanism / method.** Integrates conformal prediction directly into GNN training for node classification.
- **Result / evidence.** Reports robust uncertainty estimates and useful prediction-set behavior on graph benchmarks.
- **Boundary.** It is graph-specific CP, but it is still a prediction-set method rather than a social-bot router.
- **Migration to this project.** Useful as a stronger graph-CP adjacent reference, but the current project should stay post-hoc and hard-node focused.
- **Links.** Paper: [arXiv](https://arxiv.org/abs/2408.13825). Official repo: not verified.

### Conditional Shift-Robust Conformal Prediction for Graph Neural Network

- **Motivation.** GNN uncertainty degrades under conditional shift.
- **Problem analysis.** Standard graph CP assumptions can fail when \(P(Y\mid X)\) changes across graph regimes.
- **Mechanism / method.** Adds a shift-robust conformal objective and latent shift mitigation to graph node classification.
- **Result / evidence.** Reports stronger marginal coverage and smaller sets under conditional shift.
- **Boundary.** It addresses shift-robust graph CP, not social-bot relation-aware routing.
- **Migration to this project.** Supports the idea that the router should be phrased as calibration/ranking under regime shift, not as a new bot classifier.
- **Links.** Paper: [arXiv](https://arxiv.org/abs/2405.11968). Later published in Soft Computing (2026); official repo not verified.

### TorchCP: A Python Library for Conformal Prediction

- **Motivation.** Conformal prediction needed a scalable, reusable library for deep learning models.
- **Problem analysis.** Existing CP toolkits were not well aligned with large-scale DL, GNN, and LLM workflows.
- **Mechanism / method.** Provides a PyTorch-native CP library with GPU-accelerated post-hoc and training methods.
- **Result / evidence.** Reports strong scalability and broad model support, including GNNs.
- **Boundary.** It is infrastructure, not a method claim.
- **Migration to this project.** Confirms that CP for GNNs is now treated as a toolchain problem, which makes a post-hoc router framing natural.
- **Links.** Paper: [JMLR](https://jmlr.org/papers/v26/24-2141.html), [arXiv](https://arxiv.org/abs/2402.12683), [code](https://github.com/ml-stat-Sustech/TorchCP).

### GRAPHLCP: Structure-Aware Localized Conformal Prediction on Graphs

- **Motivation.** Embedding-space proximity alone can be unreliable for graph localization.
- **Problem analysis.** Graph topology and long-range dependencies should influence how calibration is localized.
- **Mechanism / method.** Uses graph-topology-aware localization, including feature-aware densification and Personalized PageRank-style structural kernels.
- **Result / evidence.** Reports finite-sample marginal coverage and better conditional coverage in localized settings.
- **Boundary.** It is a 2026 preprint and not yet a social-bot-specific method.
- **Migration to this project.** Strongly reinforces the idea that the router is a localized calibration/ranking problem, but also raises the bar for any relation/direction novelty claim.
- **Links.** Paper: [arXiv](https://arxiv.org/abs/2605.08074).

## Relation-Aware / Social-Bot Graph Learning

### R-GCN: Modeling Relational Data with Graph Convolutional Networks

- **Motivation.** Many graphs contain multiple relation types, and treating all edges as identical loses task-relevant structure.
- **Problem analysis.** Relation-specific message passing is needed when edge semantics change how information should propagate.
- **Mechanism / method.** Introduces relation-specific graph convolution with parameter sharing for multi-relational graphs.
- **Result / evidence.** Improves representation learning on relational graph tasks.
- **Boundary.** It is a backbone architecture, not a conformal estimator or hard-node router.
- **Migration to this project.** Use as the structural premise for relation-conditioned nonconformity channels: edge type should condition local uncertainty aggregation.
- **Links.** Paper: [arXiv](https://arxiv.org/abs/1703.06103). Official repo: not verified.

### HGT: Heterogeneous Graph Transformer

- **Motivation.** Heterogeneous graphs contain different node and edge types with distinct semantics.
- **Problem analysis.** A single shared aggregation rule can blur type-specific dependencies.
- **Mechanism / method.** Uses meta-relation-aware attention and type-specific transformations for heterogeneous graph learning.
- **Result / evidence.** Shows type-aware attention improves heterogeneous graph tasks.
- **Boundary.** It is an end-to-end heterogeneous GNN, not a post-hoc uncertainty router.
- **Migration to this project.** Use as support that relation and direction should be first-class channels in the router, while keeping the estimator post-hoc and lightweight.
- **Links.** Paper: [arXiv](https://arxiv.org/abs/2003.01332). Repo: [acbull/pyHGT](https://github.com/acbull/pyHGT).

### BotRGCN: Twitter Bot Detection with Relational Graph Convolutional Networks

- **Motivation.** Twitter bot detection must use both account features and follow-relationship structure.
- **Problem analysis.** Bots may look genuine individually but reveal coordinated or community behavior through relational graph patterns.
- **Mechanism / method.** Builds a heterogeneous follow graph and applies R-GCN with multimodal user information.
- **Result / evidence.** Reports stronger TwiBot-20 detection by using semantic, property, and neighborhood information.
- **Boundary.** It is a base social-bot detector, not a post-hoc conformal router.
- **Migration to this project.** Use as domain evidence that relation types and edge direction are not optional details; they should shape hard-node uncertainty aggregation.
- **Links.** Paper: [arXiv](https://arxiv.org/abs/2106.13092). Repo: [BunsenFeng/BotRGCN](https://github.com/BunsenFeng/BotRGCN).

### BotRGT / BotHeterogeneity: Heterogeneity-Aware Twitter Bot Detection with Relational Graph Transformers

- **Motivation.** Social-bot graphs contain heterogeneous user attributes and relations that simple homogeneous GNNs underuse.
- **Problem analysis.** Relation heterogeneity affects bot detection because different interaction types encode different behavioral evidence.
- **Mechanism / method.** Uses a relational graph transformer to model heterogeneous Twitter bot detection signals.
- **Result / evidence.** Demonstrates that heterogeneity-aware relational modeling improves bot detection on TwiBot-style data.
- **Boundary.** It is a supervised detector architecture, not a conformal uncertainty method.
- **Migration to this project.** Use as a social-bot-specific basis for relation/direction-conditioned router channels and relation-shuffle ablations.
- **Links.** Paper: [AAAI PDF](https://www.atailab.cn/seminar2022Spring/pdf/2022_AAAI_Heterogeneity-Aware%20Twitter%20Bot%20Detection%20with%20Relational%20Graph%20Transformers.pdf). Repo: [BunsenFeng/BotHeterogeneity](https://github.com/BunsenFeng/BotHeterogeneity).

### BotMoE: Twitter Bot Detection with Community-Aware Mixtures of Modal-Specific Experts

- **Motivation.** Advanced Twitter bots can manipulate one modality while blending into different communities, so a single shared encoder is brittle.
- **Problem analysis.** Metadata, textual content, and network structure expose different failure modes; fusing them too early can blur evidence that is community- or modality-specific.
- **Mechanism / method.** Uses modal-specific encoders for `metadata`, `text`, and `graph`, then applies a community-aware mixture-of-experts layer and an expert fusion layer.
- **Result / evidence.** Reports state-of-the-art Twitter-bot detection gains and explicitly shows that modality-specific expert decomposition helps detect deceptive bots.
- **Boundary.** It is a supervised detection architecture, not an LLM explanation pipeline, and its experts are neural encoders rather than prompt experts.
- **Migration to this project.** Strong support that `metadata / tweet / graph` should be treated as separable evidence channels before fusion. It supports decomposed prompt experts much more directly than a single monolithic user prompt.
- **Links.** Paper: [SIGIR 2023 via DBLP](https://dblp.org/rec/conf/sigir/LiuTWFZL23.html), [arXiv](https://arxiv.org/abs/2304.06280). Official repo: not verified.

### TwiBot-20: A Comprehensive Twitter Bot Detection Benchmark

- **Motivation.** Prior Twitter-bot benchmarks were small, biased toward a few bot types, and often lacked realistic multi-source user information.
- **Problem analysis.** A meaningful benchmark for social-bot detection should expose the main evidence channels actually used in the field: user properties, tweets, and social-network relations.
- **Mechanism / method.** Builds a large Twitter-bot benchmark with user descriptions, tweets, property items, and follow relationships, then evaluates multiple families of detectors on that shared substrate.
- **Result / evidence.** Reports that Twitter bot detection remains hard even with all these signals, and provides a benchmark where textual, metadata, and graph evidence coexist by construction.
- **Boundary.** TwiBot-20 is a dataset paper, not a decomposition method. It does not tell us which evidence channel should be handled by LLM prompts versus side-channel features.
- **Migration to this project.** Strong support that `tweet / metadata / follow-graph` is a dataset-faithful decomposition of the user state space. It is the right starting split for prompt-expert design, even if later experiments decide that metadata should stay mostly outside the prompt.
- **Links.** Paper: [arXiv](https://arxiv.org/abs/2106.13088). Official repo: not verified.

### TwiBot-22: Towards Graph-Based Twitter Bot Detection

- **Motivation.** Social-bot benchmarks need richer, larger graph structure than early small datasets provide.
- **Problem analysis.** Bot behavior is relational and multi-source; benchmark design should expose graph relations, labels, and realistic splits.
- **Mechanism / method.** Releases a large graph-based Twitter bot detection benchmark with richer social graph context.
- **Result / evidence.** Establishes TwiBot-22 as a graph-centered benchmark for bot detection research.
- **Boundary.** It is a dataset paper, not a router or conformal estimator.
- **Migration to this project.** Use as evidence that relation-aware design is domain-aligned and as a future evaluation target beyond TwiBot-20.
- **Links.** Paper: [NeurIPS Datasets and Benchmarks](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c9dfe6d38d1d5c3ecb162c81a3fbbf2-Abstract-Datasets_and_Benchmarks.html). Repo: [LuoUndergradXJTU/TwiBot-22](https://github.com/LuoUndergradXJTU/TwiBot-22).

### A Decade of Social Bot Detection

- **Motivation.** Social-bot detection research had grown rapidly, but its assumptions, feature families, and evaluation culture were fragmented.
- **Problem analysis.** The survey shows that detection historically relies on combining multiple evidence families, including profile/meta-data, content/activity cues, and network/group behaviors. It also emphasizes that more sophisticated bots increasingly evade detectors that rely on only one family of signals.
- **Mechanism / method.** Reviews the first decade of social-bot detection from early feature-based methods to more group-aware and behavior-aware approaches.
- **Result / evidence.** The central lesson is that social bots are best understood through multiple complementary evidence channels rather than a single feature family.
- **Boundary.** This is a survey and field synthesis, not a prompt-design paper. It supports multi-view decomposition at the task level, but does not by itself justify a particular prompt template.
- **Migration to this project.** Strong survey-level support for decomposing social-bot evidence into at least content/text, profile/meta-data, and network/community channels before fusion.
- **Links.** Article: [Communications of the ACM](https://cacm.acm.org/research/a-decade-of-social-bot-detection/), [arXiv postprint](https://arxiv.org/abs/2007.03604). Official repo: not applicable.

### Detection of Malicious Social Bots: A Survey and a Refined Taxonomy

- **Motivation.** Social-bot detection spans many feature families and threat models, making it hard to compare methods or identify what evidence each one actually uses.
- **Problem analysis.** The survey organizes the literature into a refined taxonomy and highlights that metadata, content, temporal behavior, and network signals play different roles under different bot behaviors.
- **Mechanism / method.** Reviews and categorizes the literature from a social-network perspective, with a focus on the evidence sources used by detectors.
- **Result / evidence.** Reinforces that bot detection is naturally multi-view and that no single feature family is sufficient across bot regimes.
- **Boundary.** It is a taxonomy paper, not a model or benchmark. It does not provide direct guidance on LLM prompting strategy.
- **Migration to this project.** Useful as complementary survey evidence that a decomposed expert design is task-aligned, but also as a warning that the `metadata` channel should be treated carefully because not all metadata cues are equally robust across bot regimes.
- **Links.** Paper: [Expert Systems with Applications](https://www.sciencedirect.com/science/article/pii/S0957417420302074). Official repo: not applicable.

### Integrating Higher-Order Relations for Enhanced Twitter Bot Detection

- **Motivation.** Direct pairwise edges may miss broader interaction patterns used by coordinated bot accounts.
- **Problem analysis.** Higher-order and relation-rich graph context can expose behavioral signals that one-hop homogeneous propagation misses.
- **Mechanism / method.** Integrates higher-order relations into Twitter bot detection models.
- **Result / evidence.** Reports that higher-order relational information can improve bot detection.
- **Boundary.** It does not provide conformal calibration or a post-hoc local rewriter.
- **Migration to this project.** Use as support for evaluating hop-specific and relation-specific uncertainty channels rather than treating 2-hop aggregation as plain smoothing.
- **Links.** Paper: [Springer SNAM](https://link.springer.com/article/10.1007/s13278-024-01372-0). Official repo: not verified.

## Text-Graph Retrieval / Modifier

### GAugLLM: Improving Graph Contrastive Learning for Text-Attributed Graphs with Large Language Models

- **Motivation.** TAG augmentation often perturbs numeric features or topology without preserving raw text semantics.
- **Problem analysis.** Text attributes and graph structure are not naturally aligned, so structure-only or text-only augmentation can create unreliable views.
- **Mechanism / method.** Uses LLM-based text augmentation and a collaborative edge modifier combining structural candidates with textual commonality.
- **Result / evidence.** Improves graph contrastive learning and downstream performance as a plug-in augmentation framework.
- **Boundary.** It targets contrastive TAG augmentation, not post-hoc social-bot ego repair.
- **Migration to this project.** Use "structural propose + text confirm" as the local retrieval/rewriter principle; do not import the GCL objective into the main shared prefix.
- **Links.** Paper: [arXiv](https://arxiv.org/abs/2406.11945). Repo: [NYUSHCS/GAugLLM](https://github.com/NYUSHCS/GAugLLM).

### SKETCH / Taming Language Models for Text-Attributed Graph Learning with Decoupled Aggregation

- **Motivation.** TAG learning needs both text semantics and graph structure, but direct LM-GNN integration is expensive.
- **Problem analysis.** Fixed LM embeddings can miss graph context, while graph convolution is not naturally aligned with text representation learning.
- **Mechanism / method.** Decouples semantic aggregation and structural aggregation, integrating them into text representation learning.
- **Result / evidence.** Shows decoupled aggregation can improve TAG representation quality.
- **Boundary.** It is a representation-learning framework, not a graph rewrite method.
- **Migration to this project.** Use semantic/structural retrieval channels to build refined ego evidence; do not reproduce full decoupled LM training in v1.
- **Links.** Paper: [ACL Anthology](https://aclanthology.org/2025.acl-long.173/). Official repo: not verified.

### CTGL: Text-Attributed Graph Learning with Coupled Augmentations

- **Motivation.** Text and graph learning can help each other when augmentations are coupled rather than independent.
- **Problem analysis.** Text-only and graph-only augmentations can optimize different objectives and fail to exchange useful information.
- **Mechanism / method.** Introduces coupled text-graph augmentation and coupled contrastive learning.
- **Result / evidence.** Reports gains by coordinating text and graph augmentation.
- **Boundary.** It is a training-time TAG augmentation method, not post-hoc local graph repair.
- **Migration to this project.** Use text-graph disagreement/complementarity as a modifier signal; do not claim CTGL reproduction.
- **Links.** Paper: [ACL Anthology](https://aclanthology.org/2025.coling-main.722/), [OpenReview](https://openreview.net/forum?id=AFul0qgBef). Official repo: not verified.

## Social Bot Graph Reliability Boundary

### BotBR: Social Bot Detection with Balanced Feature Fusion and Reliability-Enhanced Graph Learning

- **Motivation.** Social bot detection suffers from imbalanced feature fusion and unreliable graph edges.
- **Problem analysis.** Existing graph structures contain noisy or unreliable connections that can harm propagation.
- **Mechanism / method.** Introduces reliability-enhanced graph learning, edge detection, and graph contrastive learning over modified graphs.
- **Result / evidence.** Establishes edge reliability as a strong direction in social bot detection.
- **Boundary.** It occupies much of the binary reliability plus contrastive graph-learning space.
- **Migration to this project.** Treat as a boundary-setting baseline: our modifier should avoid being just binary reliable/unreliable deletion.
- **Links.** Paper DOI: [10.1145/3726302.3729908](https://doi.org/10.1145/3726302.3729908). Official repo: not verified.

### BECE: Dispelling the Fake: Social Bot Detection Based on Edge Confidence Evaluation

- **Motivation.** Advanced bots camouflage through interactions with humans, creating unreliable edges.
- **Problem analysis.** Unreliable edges contaminate message passing and weaken bot-human representation separation.
- **Mechanism / method.** Models edge representations and noise with parameterized Gaussian distributions, estimates edge confidence, and filters unreliable edges.
- **Result / evidence.** Reports improvements across social bot datasets and GNN backbones.
- **Boundary.** It is reliability-centered and removes/filters low-confidence edges; it should not be duplicated as the main claim.
- **Migration to this project.** Use as evidence that edge confidence matters, while retaining suspicious edges as evidence rather than simply deleting them.
- **Links.** Paper: [IEEE Xplore](https://ieeexplore.ieee.org/document/10530431/). Author PDF: [TNNLS PDF](https://qqqqqqby.github.io/assets/publications/TNNLS2024.pdf). Official repo: not verified.

## Evidence Graph / LLM Enhancer

### Exploring the Potential of Large Language Models (LLMs) in Learning on Graphs

- **Motivation.** Early graph-LLM work lacked a clear separation between using LLMs to enhance graph learning and using LLMs to replace graph learners as direct predictors.
- **Problem analysis.** The paper explicitly studies two pipelines: `LLMs-as-Enhancers` and `LLMs-as-Predictors`. For enhancers, it shows that the bottleneck is often the quality of textual node representations rather than graph modeling alone. For predictors, it shows that raw graph prompting is constrained by context length and that global-graph prompting is infeasible, motivating a target-centered `ego-graph` view instead.
- **Mechanism / method.** Section 4 evaluates feature-level and text-level enhancement strategies, including cascading `encoder -> GNN` pipelines and text augmentation methods such as TAPE and KEA. Section 5 studies LLMs-as-Predictors with zero-shot and few-shot prompting, then adds summarized `2-hop` neighborhood information under an `ego-graph` perspective to test whether local graph structure helps direct LLM prediction.
- **Result / evidence.** The paper reports two findings that are especially relevant here. First, in the `LLMs-as-Enhancers` setting, deep sentence embedding models can outperform fine-tuned PLM embeddings in both performance and efficiency, especially in lower-label or constrained settings. Second, in the `LLMs-as-Predictors` setting, adding summarized `ego-graph` neighborhood information can improve performance on some datasets, but can also hurt under heterophily, showing that local graph evidence is useful but not uniformly safe.
- **Boundary.** The `ego-graph` result is obtained in an `LLM-as-Predictor` pipeline, not in an `LLM-as-Enhancer` evidence-graph pipeline. The paper studies textual graph benchmarks rather than social-bot-specific camouflage and propagation risk, and it uses simple neighborhood summarization rather than explicit edge-role or provenance modeling.
- **Migration to this project.** This is a strong conceptual bridge for our current design. We can migrate the `ego-graph` idea, the target-centered local-view framing, and the caution that heterophilic neighbors may harm naive structural prompting. We should not migrate the direct-predictor role. Instead, we should reinterpret `ego-graph` as a local evidence carrier for an enhancer-style LLM branch: `refined ego evidence graph -> evidence embedding -> downstream node discriminator`.
- **Links.** Paper: [arXiv](https://arxiv.org/abs/2307.03393), [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/2307.03393). Repo: [CurryTang/Graph-LLM](https://github.com/CurryTang/Graph-LLM).

### Decomposed Prompting: A Modular Approach for Solving Complex Tasks

- **Motivation.** A single prompt often performs poorly on tasks that actually contain multiple subproblems with different reasoning requirements.
- **Problem analysis.** Monolithic prompts force one context window and one instruction to simultaneously solve subtask decomposition, retrieval, reasoning, and synthesis. This creates optimization difficulty and makes it hard to swap or improve individual components.
- **Mechanism / method.** Decomposes a complex task into modular sub-prompts, each specialized for a subtask, and allows symbolic or learned modules to be inserted between them.
- **Result / evidence.** Shows that modular decomposition can outperform stronger monolithic few-shot prompting on multi-hop and long-context tasks.
- **Boundary.** The paper is not about graphs or social-bot detection. It does not tell us which social-bot modalities to split, and it does not justify injecting raw numeric feature tables into prompts.
- **Migration to this project.** This is the clearest high-level justification for using separate prompt experts for `tweet`, `graph_following`, `graph_follower`, and possibly `metadata`, followed by downstream fusion. It argues for decomposition first, but does not itself justify a final global-summary bottleneck.
- **Links.** Paper: [OpenReview](https://openreview.net/forum?id=_nGgzQjzaRy), [arXiv](https://arxiv.org/abs/2210.02406). Official repo: not verified.

### What Does the Bot Say? Opportunities and Risks of Large Language Models in Social Media Bot Detection

- **Motivation.** Social-bot detection mixes several evidence sources whose cues can point in different directions, so a single undifferentiated prompt is often too blunt.
- **Problem analysis.** The paper argues that user profile cues, relationship structure, and content behavior should not be collapsed into one homogeneous input channel because they reveal different bot strategies and failure modes.
- **Mechanism / method.** Proposes a mixture-of-heterogeneous-experts view for social-bot detection and studies specialized LLM-based bot detectors across user-information modalities.
- **Result / evidence.** Reports strong gains from modality-aware expert decomposition on social-bot benchmarks.
- **Boundary.** It is a supervised detector with LLM-based experts, not an explanation-first cache builder for a frozen downstream refiner.
- **Migration to this project.** Strong domain-specific support for keeping `tweet`, `graph_following`, and `graph_follower` as separate prompt experts rather than collapsing them into a single monolithic graph prompt. It also supports ranking and compressing neighbor evidence rather than dumping raw neighborhoods.
- **Links.** Paper: [ACL Anthology](https://aclanthology.org/2024.acl-long.196/), [PDF](https://aclanthology.org/2024.acl-long.196.pdf). Official repo: not verified.

### TAPE: Harnessing Explanations - LLM-to-LM Interpreter for Enhanced Text-Attributed Graph Representation Learning

- **Motivation.** Strong LLM reasoning is hard to inject directly into efficient downstream TAG pipelines.
- **Problem analysis.** Plain encoder features can miss semantic rationale, while end-to-end LLM usage is expensive and difficult to scale.
- **Mechanism / method.** Prompts an LLM to produce explanations, then feeds those explanations into a smaller encoder so the explanation itself becomes a reusable feature.
- **Result / evidence.** Reports strong TAG gains from explanation-derived features and shows that explanations can act as useful intermediate representations instead of only human-facing artifacts.
- **Boundary.** It is not a social-bot-specific prompting paper, and it does not prescribe directional neighbor-card design.
- **Migration to this project.** This is the cleanest justification for `explanation -> encoder embedding` as the v2 mainline. It directly supports generating expert-specific explanations with an instruct model and then embedding them with the current finetuned RoBERTa encoder for the GLANCE refiner.
- **Links.** Paper: [arXiv](https://arxiv.org/abs/2305.19523). Repo: [XiaoxinHe/TAPE](https://github.com/XiaoxinHe/TAPE).

### DGP: A Dual-Granularity Prompting Framework for Fraud Detection with Graph-Enhanced LLMs

- **Motivation.** Fraud-detection prompts become noisy and unmanageable when multi-hop neighbor content is serialized without compression.
- **Problem analysis.** Dense or text-heavy heterogeneous neighborhoods can drown out the target node, so prompt quality depends on keeping the target detailed while summarizing neighbor evidence.
- **Mechanism / method.** Uses dual-granularity prompting: preserve fine-grained target-node details, but compress neighbor information through coarse-grained summaries and modality-aware aggregation.
- **Result / evidence.** Reports better fraud-detection performance under manageable token budgets than stronger raw text-only prompting baselines.
- **Boundary.** It is a fraud-detection graph paper, not a social-bot-specific expert-cache pipeline, and it studies direct graph-enhanced LLM prompting rather than a frozen downstream refiner.
- **Migration to this project.** Strong support for the v2 prompt-content rule: keep the target relatively detailed, compress neighbors into short ranked cards, and keep exact statistics mostly outside the prompt body in a structured side channel. It also argues against asking the LLM to emit a task-conclusive label narrative inside the prompt cache; the safer transfer is `evidence summary -> encoder embedding`, especially when target-node detail and neighbor compression need to share a limited context budget.
- **Links.** Paper: [arXiv](https://arxiv.org/abs/2507.21653). Official repo: not verified.

### LMBot: Distilling Graph Knowledge into Language Model for Graph-less Deployment in Twitter Bot Detection

- **Motivation.** Graph-based Twitter bot detection is powerful but expensive at inference because it depends on fetching multi-hop graph context.
- **Problem analysis.** The paper shows that language models already become competitive after domain adaptation on Twitter bot detection, while graph structure remains a strong teacher signal.
- **Mechanism / method.** Represents each user as a textual sequence, uses the LM for domain adaptation, trains a graph model on top, then distills graph knowledge back into the LM for graph-less deployment.
- **Result / evidence.** Reports state-of-the-art Twitter-bot detection while preserving graph-less deployment capability.
- **Boundary.** It is not an LLM-as-explainer design and does not decompose prompts into explicit graph/tweet/profile experts. Its LM input is still closer to a unified user serialization.
- **Migration to this project.** Useful as evidence that `tweet + profile` textualization is a legitimate carrier for social-bot signals, but weaker evidence for a monolithic prompt than BotMoE is for decomposed modality-specific processing. It supports keeping a compact profile/text channel, not inflating prompts with many raw statistics.
- **Links.** Paper: [WSDM 2024 via DBLP](https://dblp.org/rec/conf/wsdm/CaiT0ZWZL24.html), [arXiv](https://arxiv.org/abs/2306.17408). Repo: [czjdsg/LMBot](https://github.com/czjdsg/LMBot).

### When Do LLMs Help With Node Classification? A Comprehensive Analysis

- **Motivation.** The literature reported many strong LLM-for-node-classification results, but comparisons were confounded by inconsistent splits, backbones, and implementation choices.
- **Problem analysis.** The paper argues that practitioners need setting-specific guidance: whether LLMs help depends on supervision regime, homophily, model role (`encoder / explainer / predictor`), and model strength. Without a benchmarked comparison, it is hard to know when LLM cost is justified.
- **Mechanism / method.** Builds `LLMNodeBed`, standardizes codebases and backbones, and compares multiple LLM-based node-classification paradigms across homophilic and heterophilic datasets, supervised and semi-supervised regimes, and multiple model sizes and prompt styles.
- **Result / evidence.** The paper finds that LLM-based methods can offer larger gains when graph structure is less informative, including lower-homophily and heterophilic settings, and that the margin between stronger LLM-based encoders/explainers and traditional LM baselines becomes more visible there. It also reinforces that the role of the LLM matters: encoder/explainer pipelines behave differently from predictor pipelines.
- **Boundary.** This is a benchmark and design-guideline paper, not an evidence-graph method. Its heterophilic datasets are not social-bot graphs, and it does not build target-centered refined evidence artifacts or distinguish `propagation utility` from `evidence utility`.
- **Migration to this project.** Use it as strong support that an LLM enhancer branch is most defensible exactly in regimes where graph structure is weak, conflicting, or locally misleading. This supports routing hard nodes with poor or noisy local propagation into an `ego evidence graph -> LLM enhancer` branch. It does not license replacing the node discriminator with an LLM predictor.
- **Links.** Paper: [PMLR ICML 2025](https://proceedings.mlr.press/v267/wu25y.html), [OpenReview PDF](https://openreview.net/pdf?id=O3WqAhxuc7). Project page: [LLMNodeBed](https://llmnodebed.github.io/).

### Actions Speak Louder Than Prompts: A Large-Scale Study of LLMs for Graph Inference

- **Motivation.** Prior graph-LLM work often defaults to static serialized graph prompts without a principled comparison against higher-agency interaction modes such as tool-use or code generation.
- **Problem analysis.** The paper shows that LLM graph inference quality depends jointly on interaction mode, graph domain, homophily/heterophily regime, text length, node degree, and which input source is most informative. Static k-hop prompting can work, but it becomes brittle when feature text is long, neighborhoods are large, or irrelevant context consumes the token budget.
- **Mechanism / method.** Runs a large-scale controlled study over Prompting (`self`, `1-hop`, `2-hop`), ReAct-style `GraphTool` / `GraphTool+`, and `Graph-as-Code`, covering citation, web-link, e-commerce, and social-network datasets including long-text settings such as Reddit. It also performs dependency ablations by truncating features, deleting edges, removing labels, and shuffling adjacency.
- **Result / evidence.** Reports three high-signal findings. First, `Graph-as-Code` is the strongest overall interaction mode, with especially large gains on long-text or high-degree graphs where prompting quickly hits token limits. Second, all LLM interaction modes remain effective on heterophilic graphs, which partially challenges stronger pessimistic readings that prompt-based graph use collapses under low homophily. Third, more adaptive interaction modes outperform fixed prompting because they can selectively retrieve and compose the most informative structure, features, or labels. The paper therefore does not overturn the TMLR 2024 warning that naive prompts are often shallow, but it does show that structured and agentic graph interaction is much more capable than a "prompts cannot use graph structure" reading would suggest.
- **Boundary.** The evaluated systems are direct LLM node classifiers. This conflicts with the current project's `LLM-as-enhancer` boundary. The paper also studies general graph inference rather than social-bot-specific evidence construction, and it does not define separate `propagation utility` and `evidence utility` for edges.
- **Migration to this project.** Treat this as a strong reference for the `evidence ego graph -> LLM enhancer` branch. Migrate its `0/1/2-hop`, `2-hop budget`, and `2-hop summary` prompt baselines as compulsory static baselines for hard nodes. Use `GraphTool` / `Graph-as-Code` as optional high-agency audit baselines over a refined local evidence graph, not as the main v1 method. The key lesson is to build a target-centered, budgeted, structured evidence object that lets the LLM selectively consume informative edges and node text, instead of dumping a raw ego graph into a fixed prompt. This indirectly supports the current evidence-graph direction and weakens any argument that refined local graph evidence is intrinsically unusable by LLMs.
- **Links.** Paper: [arXiv](https://arxiv.org/abs/2509.18487) (submitted `2025-09-23`, revised `2026-03-01`), [OpenReview PDF](https://openreview.net/pdf?id=sdSC1LPmyZ). Microsoft Research page: [Actions Speak Louder Than Prompts](https://www.microsoft.com/en-us/research/publication/actions-speak-louder-than-prompts-a-large-scale-study-of-llms-for-graph-inference/) (`ICLR 2026`; page dated `September 2025`).

### GLANCE: Glance for Context

- **Motivation.** Uniform LLM-GNN fusion hides that GNNs and LLMs excel on different node subgroups.
- **Problem analysis.** Aggregate metrics obscure nodes where LLM context provides benefit, especially structural subgroups such as heterophilous nodes.
- **Mechanism / method.** Trains a lightweight router to decide when to query an LLM, then uses LLM assistance for node-aware fusion.
- **Result / evidence.** Reports gains on hard subgroups while reducing unnecessary LLM calls.
- **Boundary.** It does not perform graph repair; it selects when to use LLM context.
- **Migration to this project.** Use selective LLM-as-enhancer framing for hard nodes after conformal routing and evidence graph construction.
- **Links.** Paper: [OpenReview](https://openreview.net/forum?id=oODFyykHF5), [arXiv](https://arxiv.org/abs/2510.10849). Official repo: not verified.

### LOGIN: A Large Language Model Consulted Graph Neural Network Training Framework

- **Motivation.** GNN training can benefit from LLM knowledge on uncertain or spotted nodes.
- **Problem analysis.** Pure GNNs may struggle when graph structure and text semantics disagree or when labels are sparse.
- **Mechanism / method.** Uses an LLM as a consultant in a GNN training framework, with prompts that include semantic and topology context.
- **Result / evidence.** Supports LLM-as-consultant/enhancer for selected nodes.
- **Boundary.** It is an interactive training framework; our current plan is post-hoc and local.
- **Migration to this project.** Use concise hard-node evidence prompts and LLM-as-enhancer, not the full training loop.
- **Links.** Paper: [arXiv](https://arxiv.org/abs/2405.13902). Repo listed by paper: [QiaoYRan/LOGIN](https://github.com/QiaoYRan/LOGIN). Local reproduction not verified.

### GraphText: Graph Reasoning in Text Space

- **Motivation.** LLMs operate on text, while graph tasks use relational structure.
- **Problem analysis.** Graph structure must be serialized into text for LLMs to consume it.
- **Mechanism / method.** Converts graph structures and node attributes into graph text sequences for LLM-based graph reasoning.
- **Result / evidence.** Demonstrates feasibility of graph-to-text task formulation.
- **Boundary.** Naive graph serialization can lose structure and overrun context budgets.
- **Migration to this project.** Serialize only evidence ego artifacts, not raw full ego dumps.
- **Links.** Paper: [arXiv](https://arxiv.org/abs/2310.01089), [Hugging Face paper page](https://huggingface.co/papers/2310.01089). Official repo: not verified.

### A Survey of Graph Meets Large Language Model

- **Motivation.** LLM-graph methods need a clear taxonomy.
- **Problem analysis.** LLMs can be used as enhancers, predictors, or alignment components, and these roles should not be conflated.
- **Mechanism / method.** Surveys and categorizes graph-LLM integration strategies.
- **Result / evidence.** Provides role taxonomy and limitations of graph-LLM systems.
- **Boundary.** It is a survey, not an implementation method.
- **Migration to this project.** Use the LLM-as-enhancer label; avoid direct LLM predictor framing.
- **Links.** Paper: [IJCAI](https://www.ijcai.org/proceedings/2024/898), [arXiv](https://arxiv.org/abs/2311.12399). Paper list repo: [yhLeeee/Awesome-LLMs-in-Graph-tasks](https://github.com/yhLeeee/Awesome-LLMs-in-Graph-tasks).

### Can LLMs Effectively Leverage Graph Structural Information through Prompts, and Why?

- **Motivation.** Many graph prompt methods assume LLMs understand serialized graph structure.
- **Problem analysis.** LLM performance may come from contextual label-relevant phrases rather than true graph-structural reasoning.
- **Mechanism / method.** Analyzes graph structural prompts, leakage-free data, and what local prompt elements drive performance.
- **Result / evidence.** Finds LLMs tend to process graph prompts as contextual paragraphs rather than faithful graph structures.
- **Boundary.** This is counter-evidence against claiming that LLMs reason over graph topology from prompts.
- **Migration to this project.** Treat evidence graph text as structured evidence context, not proof of LLM graph reasoning.
- **Links.** Paper: [OpenReview](https://openreview.net/forum?id=L2jRavXRxs). Repo: [TRAIS-Lab/LLM-Structured-Data](https://github.com/TRAIS-Lab/LLM-Structured-Data).

## Soft Graph Rewrite / Graph Structure Learning

### LDS: Learning Discrete Structures for Graph Neural Networks

- **Motivation.** GNNs assume a given graph, but real graph structures may be noisy, missing, or unknown.
- **Problem analysis.** Treating graph structure as fixed can prevent the model from learning useful dependencies.
- **Mechanism / method.** Learns a distribution over discrete graph edges via bilevel optimization.
- **Result / evidence.** Shows graph structure can be learned jointly with GNN parameters.
- **Boundary.** Full bilevel graph learning is too broad for our post-hoc local pipeline.
- **Migration to this project.** Use the principle that edge existence/weight can be probabilistic; implement only local soft weighting.
- **Links.** Paper: [arXiv](https://arxiv.org/abs/1903.11960). Repo: [lucfra/LDS-GNN](https://github.com/lucfra/LDS-GNN).

### Pro-GNN: Graph Structure Learning for Robust Graph Neural Networks

- **Motivation.** GNNs are vulnerable to noisy or adversarially perturbed graph structures.
- **Problem analysis.** Robust graph learning can exploit structural priors such as sparsity, low rank, and feature smoothness.
- **Mechanism / method.** Jointly learns a clean graph structure and robust GNN.
- **Result / evidence.** Improves robustness under graph perturbations.
- **Boundary.** It is full-graph robust structure learning and assumes priors that may not hold for bot-human camouflage.
- **Migration to this project.** Use budgeted repair/rollback discipline, not full-graph reconstruction.
- **Links.** Paper: [arXiv](https://arxiv.org/abs/2005.10203). Repo: [ChandlerBang/Pro-GNN](https://github.com/ChandlerBang/Pro-GNN).

### GNNGuard

- **Motivation.** Adversarial or irrelevant graph edges can corrupt GNN message passing.
- **Problem analysis.** Feature/representation similarity can indicate whether an edge is useful for propagation.
- **Mechanism / method.** Learns or assigns edge weights and prunes/attenuates unrelated edges.
- **Result / evidence.** Defends multiple GNNs against graph attacks.
- **Boundary.** It is adversarial-defense oriented and often treats low-similarity edges as harmful, which is unsafe for social bot evidence.
- **Migration to this project.** Use soft edge weighting but preserve suspicious edges as evidence.
- **Links.** Paper: [arXiv](https://arxiv.org/abs/2006.08149), [NeurIPS PDF](https://zitniklab.hms.harvard.edu/publications/papers/gnnguard-neurips20.pdf). Repo: [mims-harvard/GNNGuard](https://github.com/mims-harvard/GNNGuard).

### IDGL: Iterative Deep Graph Learning for Graph Neural Networks

- **Motivation.** Graph structure and node embeddings can improve each other iteratively.
- **Problem analysis.** Fixed observed graphs may be incomplete or noisy.
- **Mechanism / method.** Jointly and iteratively learns graph structure and GNN embeddings.
- **Result / evidence.** Reports better and more robust node embeddings.
- **Boundary.** It is an end-to-end full-graph learning framework.
- **Migration to this project.** Use only limited local accept/rollback iteration ideas.
- **Links.** Paper: [arXiv](https://arxiv.org/abs/2006.13009). Repo: [hugochan/IDGL](https://github.com/hugochan/IDGL).

### GLEM: Learning on Large-scale Text-attributed Graphs via Variational Inference

- **Motivation.** LMs and GNNs each capture complementary information on text-attributed graphs.
- **Problem analysis.** Separately training LM and GNN can underuse their mutual signal.
- **Mechanism / method.** Alternates LM and GNN learning in an EM-style framework.
- **Result / evidence.** Improves large-scale TAG learning.
- **Boundary.** It retrains models in an iterative EM loop, unlike our frozen/post-hoc design.
- **Migration to this project.** Use only the high-level idea that semantic and graph sides can iteratively correct each other.
- **Links.** Paper: [OpenReview](https://openreview.net/forum?id=q0nmYciuuZN). Repo: [AndyJZhao/GLEM](https://github.com/AndyJZhao/GLEM).

## Edge Importance as Modifier Inspiration

### GNNExplainer

- **Motivation.** GNN predictions are hard to interpret because they mix graph structure and node features.
- **Problem analysis.** Important subgraphs and features can explain a prediction.
- **Mechanism / method.** Optimizes a mask maximizing mutual information between the prediction and explanatory subgraph/features.
- **Result / evidence.** Produces compact explanations for GNN predictions and improves over baselines.
- **Boundary.** Explanation masks are not causal repair actions.
- **Migration to this project.** Use edge-importance thinking to justify modifier diagnostics, not final causal claims.
- **Links.** Paper: [NeurIPS](https://papers.nips.cc/paper/9123-gnnexplainer-generating-explanations-for-graph-neural-networks), [arXiv](https://arxiv.org/abs/1903.03894). Repo: [RexYing/gnn-model-explainer](https://github.com/RexYing/gnn-model-explainer).

### PGExplainer

- **Motivation.** Instance-wise GNN explanations can be expensive and hard to generalize.
- **Problem analysis.** A parameterized explainer can learn reusable edge explanation distributions.
- **Mechanism / method.** Learns an explanation network that parameterizes edge masks for multiple instances.
- **Result / evidence.** Provides multi-instance GNN explanations with strong explanation metrics.
- **Boundary.** It is an explanation method, not a graph rewriter.
- **Migration to this project.** Use as support for learned edge mask/utility ablations if deterministic modifiers underperform.
- **Links.** Paper: [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2020/hash/e37b08dd3015330dcbb5d6663667b8b8-Abstract.html), [arXiv](https://arxiv.org/abs/2011.04573). Repo: [flyingdoog/PGExplainer](https://github.com/flyingdoog/PGExplainer).

### GraphMask

- **Motivation.** GNNs in NLP need faithful interpretation of which edges matter.
- **Problem analysis.** Post-hoc alternatives can find small subgraphs that preserve output without reflecting the original computation.
- **Mechanism / method.** Uses differentiable edge masking to decide whether messages can be replaced without changing behavior.
- **Result / evidence.** Supports edge-level masking as an interpretable mechanism.
- **Boundary.** It explains existing decisions; it does not define social-bot edge reliability.
- **Migration to this project.** Use as support for differentiable or soft edge masking in modifier ablations.
- **Links.** Paper: [OpenReview](https://openreview.net/forum?id=WznmQa42ZAx), [arXiv](https://arxiv.org/abs/2010.00577). Repo: [MichSchli/GraphMask](https://github.com/MichSchli/GraphMask).
