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
- **Migration to this project.** Use the LLM-as-enhancer label; avoid LLM-as-final-predictor framing.
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

