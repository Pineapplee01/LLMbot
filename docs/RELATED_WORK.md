# Related Work — Social Bot Detection 2024-2025

> Comprehensive analysis of related methods in the current landscape.

## Overview

This document positions our RACE-Bot-D3F method against state-of-the-art social bot detection approaches from 2024-2025.

## Methods

### 1. LMbot (WSDM 2024)

**Paper**: "Distilling Graph Knowledge into Language Model for Graph-less Deployment in Twitter Bot Detection"  
**Authors**: Yang et al.  
**Venue**: WSDM 2024

**Core Approach**:
- Two-stage framework: GNN teacher → LM student distillation
- Graph-aware LM that captures structural information without graph at inference
- Relational Graph Transformer (RGT) for teacher model

**Key Innovation**:
- Enables deployment without graph infrastructure (graph-less inference)
- Distillation preserves graph knowledge in LM parameters

**Limitations**:
- Static architecture — doesn't adapt graph structure based on inputs
- No mechanism to handle low-confidence graph regions

**TwiBot-20 Performance**: Acc 0.8533, F1 0.8732

**Code**: `docs/published/lmbot-wsdm2024/`

---

### 2. SEBot (KDD 2024)

**Paper**: "SEBot: Structural Entropy Guided Multi-View Contrastive Learning for Social Bot Detection"  
**Authors**: Yang et al.  
**Venue**: KDD 2024

**Core Approach**:
- Uses structural entropy as uncertainty metric to optimize graph structure
- Encoding trees reveal hierarchical community structures
- Multi-view contrastive learning (node-level + subgraph-level)
- RGCN backbone with message passing beyond homophily

**Key Innovation**:
- **SEPN** (Structural Entropy-guided Node-level encoding): Bottom-up message passing on node-level encoding trees
- **SEPG** (Structural Entropy-guided Subgraph-level encoding): Subgraph-level encoding trees with multi-view contrastive learning
- Robust to adversarial bot behavior by enabling non-homophilic message passing

**Limitations**:
- Structural entropy is purely graph-based — doesn't leverage text/semantic information
- Global structure optimization — not input-adaptive
- Requires pre-computed encoding trees

**TwiBot-20 Performance**: ~88% F1 (reported in paper)

**Code**: `SEBot/` (cloned from GitHub)

**Citation**:
```bibtex
@inproceedings{yang2024sebot,
  title={SeBot: Structural Entropy Guided Multi-View Contrastive Learning for Social Bot Detection},
  author={Yang, Yingguang and Wu, Qi and He, Buyun and Peng, Hao and Yang, Renyu and Hao, Zhifeng and Liao, Yong},
  booktitle={KDD},
  pages={3841--3852},
  year={2024}
}
```

---

### 3. BotBR (SIGIR 2025)

**Paper**: "BotBR: Social Bot Detection with Balanced Feature Fusion and Reliability-Enhanced Graph Learning"  
**Authors**: Lin & Zhou  
**Venue**: SIGIR 2025

**Core Approach**:
- Balanced feature fusion between content and structural features
- Reliability-enhanced graph learning
- GNN-based with attention mechanisms

**Key Innovation**:
- Addresses feature imbalance between content and graph modalities
- Reliability weighting for noisy graph edges

**Limitations**:
- Feature-level fusion — doesn't modify graph topology
- Reliability is learned, not guided by external semantic signals

**TwiBot-20 Performance**: ~86.8% F1 (reported)

**Code**: `botbr/` (cloned from GitHub)

**Citation**:
```bibtex
@inproceedings{lin2025botbr,
  title={BotBR: Social Bot Detection with Balanced Feature Fusion and Reliability-Enhanced Graph Learning},
  author={Lin, Qilong and Zhou, Jingya},
  booktitle={SIGIR},
  pages={392--402},
  year={2025}
}
```

---

### 4. HyperScan (CIKM 2025)

**Paper**: "Higher-Order Information Matters: A Representation Learning Approach for Social Bot Detection"  
**Authors**: Gao et al.  
**Venue**: CIKM 2025

**Core Approach**:
- Hypergraph representation learning
- Captures higher-order interactions beyond pairwise edges
- Multi-scale information aggregation

**Key Innovation**:
- Hypergraph neural networks for social bot detection
- Higher-order structural patterns

**Limitations**:
- Computational complexity of hypergraph construction
- No semantic/text guidance for structure learning

**TwiBot-20 Performance**: 87.2% F1 (reported)

**Code**: `HyperScan/` (cloned from GitHub)

**Citation**:
```bibtex
@inproceedings{gao2025hi,
  title={Higher-Order Information Matters: A Representation Learning Approach for Social Bot Detection},
  author={Gao, Min and Duan, Qiang and Liu, Boen and Xiao, Yu and Wang, Xin and Chen, Yang},
  booktitle={CIKM},
  pages={675--685},
  year={2025}
}
```

---

## Our Method: RACE-Bot-D3F

**Full Name**: Risk-Aware Calibrated Evidence with Dependency-Discounted Directional Fusion

**Core Approach**:
- LLM-guided test-time graph surgery
- Conditional graph modification based on semantic-graph disagreement
- Reliability-aware routing between text and graph experts

**Key Innovations**:

1. **Disagreement Detection**: Identify nodes where LLM and GNN predictions differ
2. **Graph Surgery**: Prune unreliable edges, add helpful edges based on LLM confidence
3. **Dynamic Fusion**: Weight predictions by reliability, not fixed weights
4. **Calibration**: Post-hoc temperature scaling for well-calibrated uncertainties

**Advantages over Prior Work**:

| Aspect | LMbot | SEBot | BotBR | HyperScan | RACE-Bot-D3F (Ours) |
|--------|-------|-------|-------|-----------|---------------------|
| Uses LLM semantics | ✅ | ❌ | ❌ | ❌ | ✅ |
| Modifies graph structure | ❌ | ✅ (global) | ❌ | ✅ (hypergraph) | ✅ (local, input-adaptive) |
| Handles disagreement | ❌ | ❌ | ❌ | ❌ | ✅ |
| Test-time adaptation | ❌ | ❌ | ❌ | ❌ | ✅ |
| Graph-less deployment | ✅ | ❌ | ❌ | ❌ | ❌ (requires graph) |

**Current Status**:
- Mechanism validated (kill tests B2/B3 passed)
- Disagreement slice: +0.16 F1 improvement
- Awaiting baseline comparability audit for full comparison

**Code**: `LLMbot/`

---

## Technical Comparison

### Graph Modification Strategies

| Method | When to Modify | How to Modify | Guidance |
|--------|---------------|---------------|----------|
| SEBot | Pre-processing | Global encoding trees | Structural entropy |
| RACE-Bot-D3F | Test-time | Local edge prune/add | LLM confidence + disagreement |

### Fusion Strategies

| Method | Fusion Type | Weights | Calibration |
|--------|-------------|---------|-------------|
| LMbot | Late fusion | Learned | None |
| BotBR | Feature fusion | Reliability-based | Implicit |
| RACE-Bot-D3F | Dynamic routing | Uncertainty + reliability | Post-hoc temperature |

### Robustness Mechanisms

| Method | Adversarial Defense | Uncertainty Handling |
|--------|--------------------|---------------------|
| SEBot | Non-homophilic message passing | Structural entropy |
| RACE-Bot-D3F | Prune unreliable edges | EDL + temperature scaling |

---

## Experimental Setup Comparison

### Datasets

All methods evaluate on:
- **TwiBot-20**: Primary benchmark (11,826 nodes)
- **MGTAB**: Secondary benchmark (used by SEBot, BotBR, HyperScan)

### Metrics

| Method | Primary | Secondary |
|--------|---------|-----------|
| LMbot | Acc, F1 | — |
| SEBot | Acc, F1, Recall, Precision | — |
| BotBR | F1 | — |
| HyperScan | F1 | — |
| RACE-Bot-D3F | Acc, Macro-F1 | ECE, NLL, Brier, AURC |

### Split Protocol

**Critical**: All methods must use the same canonical splits for fair comparison.
- Train/Val/Test splits provided in `datasets/TwiBot-20/`
- No test-set hyperparameter tuning

---

## Research Opportunities

1. **Combine SEBot + RACE-Bot-D3F**: Use structural entropy for global optimization, LLM guidance for local surgery
2. **Extend to MGTAB**: Current focus on TwiBot-20; MGTAB evaluation needed
3. **Graph-less variant**: Can we distill the modified graph knowledge like LMbot?
4. **Theoretical analysis**: Why does disagreement indicate unreliable graph regions?

---

---

## Extended Literature Survey: Edge Trust & Graph Structure Learning (2024-2026)

> Papers organized by technical component relevant to our learned edge trust scoring framework.

### Component 1: Learned Edge Weighting / Edge Trust for GNNs

#### P1. TrustSGCN — Trustworthiness-Driven GCN for Signed Network Embedding
- **Authors**: Kim et al.
- **Venue**: arXiv 2023 (updated 2024)
- **URL**: https://ar5iv.labs.arxiv.org/html/2309.00816
- **Core Method**: Corrects incorrect embedding propagation in GCN by computing per-edge trustworthiness scores based on sign consistency. Uses trust to reweight message passing in signed networks.
- **Relevance**: Directly models edge-level trust for GNN propagation — closest conceptual ancestor to our edge trust scoring. However, limited to signed networks (trust/distrust), not general social graphs.
- **Differentiation**: Our framework generalizes trust beyond signed edges to arbitrary social relations, and jointly calibrates semantic + structural signals rather than relying on sign patterns alone.

#### P2. TrustGTN — Social Network Trust Evaluation via Heterogeneous GNN
- **Authors**: (MDPI Computers 2025)
- **URL**: https://www.mdpi.com/2073-431X/15/3/176
- **Core Method**: Heterogeneous Graph Transformer Network with soft selection mechanism that dynamically adjusts training matrices for trust evaluation. Learns trust propagation patterns across heterogeneous relation types.
- **Relevance**: Demonstrates learned trust scoring on social graphs with heterogeneous relations — directly applicable to our multi-relation social bot graphs.
- **Differentiation**: TrustGTN evaluates inter-user trust as a prediction target; we use trust as an intermediate mechanism to reweight edges for downstream bot classification.

#### P3. DGTEN — Deep Gaussian-Based Trust Evaluation Network
- **Authors**: (arXiv 2025)
- **URL**: https://arxiv.org/abs/2510.07620
- **Core Method**: Combines uncertainty-aware message passing with temporal modeling and Gaussian-based trust evaluation. Provides uncertainty quantification for trust predictions on dynamic graphs.
- **Relevance**: Directly combines uncertainty quantification with trust evaluation on graphs — validates our core thesis that uncertainty-aware trust scoring improves graph operations.
- **Differentiation**: DGTEN focuses on trust prediction as the end task; we use uncertainty-calibrated trust as an edge reweighting mechanism for bot detection.

---

### Component 2: Graph Structure Learning with Semantic Guidance

#### P4. LLM4RGNN — Can LLMs Improve Adversarial Robustness of GNNs?
- **Authors**: He et al.
- **Venue**: arXiv 2024 (KDD-adjacent)
- **URL**: https://arxiv.org/abs/2408.08685
- **Core Method**: Uses LLMs to purify adversarially perturbed graph structures. LLM evaluates whether edges are semantically plausible given node text attributes, then prunes implausible edges. Maintains GNN accuracy even at 40% perturbation ratio.
- **Relevance**: **Highly relevant** — demonstrates LLM-guided edge purification for graph robustness. Validates that semantic signals from LLMs can identify unreliable edges.
- **Differentiation**: LLM4RGNN performs binary edge purification (keep/remove); our framework learns continuous trust scores for soft reweighting. LLM4RGNN targets adversarial defense; we target camouflage bot detection. We also incorporate structural uncertainty, not just semantic plausibility.

#### P5. Unlocking Graph Structure Learning with Tree-Guided LLMs
- **Authors**: (arXiv 2025)
- **URL**: https://arxiv.org/html/2503.21223v5
- **Core Method**: Leverages LLMs to perform fine-grained graph description on text-attributed graphs (TAGs), using tree-structured guidance to learn graph topology from textual information.
- **Relevance**: Shows how LLM semantic understanding can guide graph structure discovery — supports our use of LLM confidence to inform edge trust.
- **Differentiation**: Focuses on graph construction from text; we modify existing social graph structure using semantic disagreement signals.

#### P6. LGB — Language Model and GNN-Driven Social Bot Detection
- **Authors**: (arXiv 2024)
- **URL**: https://arxiv.org/html/2406.08762v2
- **Core Method**: Fuses language model and GNN modalities to improve detection of sparsely linked nodes. LM branch provides semantic features; GNN branch provides structural features; fusion improves performance on isolated/low-degree nodes.
- **Relevance**: **Directly addresses our target problem** — improving bot detection on sparsely linked (isolated) nodes via LM+GNN fusion. Validates that semantic signals help where graph structure is weak.
- **Differentiation**: LGB uses late fusion of two modalities; we use semantic confidence to actively reweight graph edges, modifying the graph structure itself rather than just fusing predictions.

---

### Component 3: Soft Graph Rewiring / Differentiable Graph Editing

#### P7. Graph Rewiring with Gumbel-Softmax
- **Authors**: (arXiv 2025)
- **URL**: https://arxiv.org/html/2508.17531
- **Core Method**: Proposes Gumbel-Softmax-based differentiable rewiring that reduces deviations in neighborhood distributions. Enables end-to-end learnable edge addition/removal while maintaining differentiability.
- **Relevance**: **Key technical reference** — provides the differentiable relaxation mechanism (Gumbel-Softmax) that enables soft edge decisions during training. Directly applicable to our soft edge reweighting.
- **Differentiation**: Focuses on oversquashing/long-range dependencies; we apply soft rewiring specifically guided by semantic-structural disagreement for bot detection.

#### P8. Probabilistic Graph Rewiring via Virtual Nodes (IPR-MPNN)
- **Authors**: Karhadkar et al.
- **Venue**: arXiv 2024
- **URL**: https://arxiv.org/html/2405.17311v1
- **Core Method**: Introduces virtual nodes in a differentiable, end-to-end manner to enable long-distance message passing. Virtual nodes act as probabilistic bridges between distant graph regions.
- **Relevance**: Demonstrates differentiable graph augmentation (adding virtual connections) — conceptually similar to our approach of adding edges for isolated nodes.
- **Differentiation**: IPR-MPNN adds virtual nodes for information flow; we reweight existing edges and potentially add edges based on semantic similarity, guided by trust scores.

#### P9. Adaptive Edge Learning for Density-Aware Graph Generation
- **Authors**: (arXiv 2026)
- **URL**: https://arxiv.org/html/2601.23052
- **Core Method**: Differentiable edge predictor determines pairwise relationships from node embeddings; density-aware selection mechanism adaptively controls edge density per class.
- **Relevance**: Demonstrates learned, differentiable edge prediction with adaptive density control — directly relevant to our edge trust scoring that controls effective graph density.
- **Differentiation**: Targets graph generation; we target graph modification for classification. Our density control is implicit through trust thresholds rather than explicit density matching.

---

### Component 4: Uncertainty-Aware Graph Neural Networks

#### P10. Sparse Bayesian Message Passing under Structural Uncertainty
- **Authors**: (arXiv 2026)
- **URL**: https://arxiv.org/html/2601.01207v1
- **Core Method**: Proposes sparse signed message passing network that explicitly captures structural uncertainty from a Bayesian perspective. Naturally robust to edge noise and heterophily. Learns which edges to trust during message passing.
- **Relevance**: **Most directly relevant to our framework** — explicitly models structural uncertainty at the edge level and uses it to guide message passing. Validates our core hypothesis that uncertainty-aware edge operations improve robustness.
- **Differentiation**: Purely structural Bayesian approach; we additionally incorporate semantic confidence from LLM/text features. Our framework is directional (asymmetric trust), while this is symmetric.

#### P11. Uncertainty Modeling in GNNs via Stochastic Differential Equations (LGNSDE)
- **Authors**: (NeurIPS 2024)
- **URL**: https://openreview.net/forum?id=TYSQYx9vwd
- **Core Method**: SDE framework for learning uncertainty-aware representations on graphs. Graph Neural ODE extended with stochastic components to capture both aleatoric and epistemic uncertainty.
- **Relevance**: Provides theoretical grounding for uncertainty quantification in graph learning — supports our use of calibrated uncertainty to guide edge operations.
- **Differentiation**: Models node-level uncertainty via SDEs; we model edge-level trust/uncertainty and use it for graph structure modification.

#### P12. G-DeltaUQ — Accurate and Scalable Epistemic Uncertainty for GNNs
- **Authors**: (ICLR 2024)
- **URL**: https://openreview.net/forum?id=ZL6yd6N1S2
- **Core Method**: Training framework that improves intrinsic GNN uncertainty estimates via stochastic data centering. Provides calibrated confidence indicators under distribution shift.
- **Relevance**: Addresses GNN calibration under distribution shift — directly relevant since camouflage bots create distribution shift in local neighborhoods.
- **Differentiation**: Focuses on node-level uncertainty estimation; we extend to edge-level trust scoring and use uncertainty to drive graph modification.

#### P13. Enhance GNNs with Reliable Confidence via Adversarial Calibration Learning
- **Authors**: (arXiv 2025)
- **URL**: https://arxiv.org/html/2503.18235
- **Core Method**: Adversarial calibration learning that improves GNN confidence reliability across different graph regions. Addresses the problem that global calibration methods fail to generalize locally.
- **Relevance**: Validates that local (per-region) calibration matters for GNNs — supports our per-edge trust calibration approach rather than global calibration.
- **Differentiation**: Post-hoc calibration method; we integrate calibration into the edge trust learning process itself.

#### P14. Graph Structure Learning with Interpretable Bayesian Neural Networks
- **Authors**: (arXiv 2024, ICLR workshop)
- **URL**: https://arxiv.org/html/2406.14786v1
- **Core Method**: Uses interpretable Bayesian neural networks for graph structure learning with uncertainty quantification on edge predictions. MCMC-based posterior approximation provides edge-level confidence intervals.
- **Relevance**: **Highly relevant** — provides Bayesian uncertainty quantification specifically for learned graph edges. Demonstrates that edge uncertainty can be meaningfully quantified and used.
- **Differentiation**: General-purpose GSL method; we specialize for social bot detection with semantic guidance. Their Bayesian approach could complement our framework's uncertainty estimation.

---

### Component 5: Social Bot Detection with Graph Modification

#### P15. RABot — Reinforcement-Guided Graph Augmentation for Imbalanced and Noisy Bot Detection
- **Authors**: (arXiv 2026)
- **URL**: https://arxiv.org/abs/2602.21749
- **Core Method**: Multi-granularity graph augmentation framework using reinforcement learning. Neighborhood-aware oversampling interpolates minority-class embeddings within local subgraphs. RL agent learns which augmentations improve detection under noise and imbalance.
- **Relevance**: **Directly relevant** — addresses noisy graph structure in bot detection via learned graph augmentation. Validates that graph modification improves bot detection under noise.
- **Differentiation**: RABot uses RL for augmentation decisions; we use learned trust scores. RABot focuses on class imbalance; we focus on camouflage and isolated nodes. Our approach is differentiable end-to-end rather than RL-based.

#### P16. RMNP — Certainly Bot Or Not? Trustworthy Bot Detection via Robust Multi-Modal Neural Processes
- **Authors**: (arXiv 2025)
- **URL**: https://arxiv.org/html/2503.09626v1
- **Core Method**: Robust Multi-modal Neural Processes that handle modality inconsistencies caused by bot camouflage. Provides uncertainty-aware predictions that are robust to missing or corrupted modalities.
- **Relevance**: **Highly relevant** — directly addresses uncertainty-aware bot detection under camouflage. Validates that uncertainty quantification improves trustworthiness of bot detection.
- **Differentiation**: RMNP handles modality-level uncertainty; we handle edge-level structural uncertainty. RMNP doesn't modify graph structure; we actively reweight edges based on trust scores.

#### P17. BotDGT — Dynamicity-aware Social Bot Detection with Dynamic Graph Transformers
- **Authors**: (arXiv 2024)
- **URL**: https://arxiv.org/html/2404.15070v2
- **Core Method**: Dynamic graph transformer that models temporal evolution of social networks. Structural module captures topology from historical snapshots; temporal module integrates evolving behavior.
- **Relevance**: Addresses bot camouflage through temporal dynamics — complementary to our spatial edge trust approach.
- **Differentiation**: BotDGT uses temporal dynamics; we use semantic-structural disagreement. Could be combined: temporal trust evolution + semantic trust scoring.

#### P18. BotHP — Heterophily-Aware Bot Detection with Prototype-Guided Cluster Discovery
- **Authors**: (arXiv 2025)
- **URL**: https://arxiv.org/html/2506.00989v1
- **Core Method**: Dual-encoder architecture with graph-aware encoder (node commonality) and graph-agnostic encoder (node uniqueness). Prototype-guided cluster discovery handles heterophily in bot-human interactions.
- **Relevance**: Addresses heterophily in bot detection — the same problem our edge trust framework targets (bots connecting to humans create heterophilic edges).
- **Differentiation**: BotHP uses dual encoders to handle heterophily; we reweight heterophilic edges via trust scores. Our approach modifies the graph itself rather than learning separate representations.

#### P19. BotSCL — Heterophily-aware Social Bot Detection with Supervised Contrastive Learning
- **Authors**: (arXiv 2023, updated 2024)
- **URL**: https://arxiv.org/html/2306.07478v3
- **Core Method**: Contrastive learning framework that differentiates neighbor representations for heterophilic relations while assimilating homophilic ones. Graph augmentation generates different views.
- **Relevance**: Explicitly models heterophily in bot detection via contrastive learning — validates that heterophily-awareness improves bot detection.
- **Differentiation**: BotSCL uses contrastive learning to handle heterophily; we use edge trust reweighting. Our approach is more interpretable (explicit trust scores per edge).

---

### Component 6: Edge-Level Confidence / Reliability Scoring

#### P20. Pruning Graphs by Adversarial Robustness Evaluation (PARE)
- **Authors**: (arXiv 2024)
- **URL**: https://arxiv.org/html/2512.22128v1
- **Core Method**: Computes per-edge robustness scores and selectively prunes edges most likely to degrade model reliability. Uses adversarial robustness evaluation as edge quality metric.
- **Relevance**: **Directly relevant** — scores edges by reliability and prunes unreliable ones. Validates edge-level scoring for graph cleaning.
- **Differentiation**: PARE performs hard pruning based on robustness scores; we perform soft reweighting based on trust scores. PARE is defense-focused; we are detection-focused with semantic guidance.

#### P21. Structure-Adaptive GNN via Adversarial Synthesis and Self-Corrective Propagation
- **Authors**: (arXiv 2026)
- **URL**: https://arxiv.org/html/2602.17071v1
- **Core Method**: Addresses structural noise and non-homophilous topologies via adversarial synthesis of graph perturbations and self-corrective propagation that learns to undo harmful structural patterns.
- **Relevance**: Self-corrective propagation is conceptually similar to our trust-guided reweighting — both learn to correct unreliable graph structure during message passing.
- **Differentiation**: Uses adversarial training for correction; we use semantic-structural disagreement. Our approach provides explicit trust scores rather than implicit correction.

#### P22. Topologically-Stabilized Graph Neural Networks
- **Authors**: (arXiv 2024)
- **URL**: https://arxiv.org/html/2512.13852v1
- **Core Method**: Integrates persistent homology with GNNs to stabilize representations against structural perturbations. Topological features provide robustness certificates.
- **Relevance**: Provides structural robustness via topological analysis — complementary to our semantic-guided approach.
- **Differentiation**: Purely topological approach; we combine semantic and structural signals. Could be integrated: topological stability + semantic trust.

#### P23. Robust Graph Learning via Diffusion-Based Structure Purification
- **Authors**: (arXiv 2025)
- **URL**: https://arxiv.org/html/2502.05000v1
- **Core Method**: Prior-free diffusion model for graph structure purification. Uses graph transfer entropy to guide denoising, promoting semantic alignment between clean and purified graphs.
- **Relevance**: Diffusion-based graph purification with semantic alignment — validates that semantic signals can guide graph structure cleaning.
- **Differentiation**: Generative diffusion approach; we use discriminative trust scoring. Diffusion is computationally heavier; our approach is more efficient for inference.

#### P24. BetaExplainNN — Probabilistic Method to Explain GNNs
- **Authors**: (arXiv 2024)
- **URL**: https://arxiv.org/html/2412.11964v1
- **Core Method**: Beta distribution-based edge weight uncertainty quantification for GNN explanations. Provides probabilistic confidence intervals for edge importance scores.
- **Relevance**: Demonstrates that edge-level uncertainty can be meaningfully quantified using Beta distributions — provides a concrete parameterization option for our trust scores.
- **Differentiation**: Focuses on post-hoc explanation; we use edge uncertainty during training for graph modification.

---

### Summary Table: Positioning Our Edge Trust Framework

| Paper | Edge-Level Scoring | Semantic Guidance | Uncertainty-Aware | Soft Reweighting | Bot Detection |
|-------|-------------------|-------------------|-------------------|------------------|---------------|
| TrustSGCN (P1) | Yes (sign-based) | No | No | Yes | No |
| TrustGTN (P2) | Yes (heterogeneous) | No | No | Yes | No |
| DGTEN (P3) | Yes (Gaussian) | No | Yes | Yes | No |
| LLM4RGNN (P4) | Yes (binary) | Yes (LLM) | No | No (hard prune) | No |
| LGB (P6) | No | Yes (LM) | No | No | Yes |
| Gumbel-Rewire (P7) | Yes (Gumbel) | No | No | Yes | No |
| Bayesian MP (P10) | Yes (Bayesian) | No | Yes | Yes | No |
| Bayesian GSL (P14) | Yes (MCMC) | No | Yes | Yes | No |
| RABot (P15) | No | No | No | No (RL augment) | Yes |
| RMNP (P16) | No | Yes (multi-modal) | Yes | No | Yes |
| BotHP (P18) | No | No | No | No | Yes |
| PARE (P20) | Yes (robustness) | No | No | No (hard prune) | No |
| **Ours** | **Yes (learned trust)** | **Yes (LLM+text)** | **Yes (calibrated)** | **Yes (directional)** | **Yes** |

**Key gap our framework fills**: No existing work simultaneously provides (1) learned per-edge trust scores, (2) semantic guidance from LLM/text, (3) calibrated uncertainty, (4) soft directional reweighting, and (5) application to social bot detection. The closest works address 2-3 of these dimensions but not all five.

---

## References

1. Yang et al. "LMbot: Distilling Graph Knowledge into Language Model for Graph-less Deployment in Twitter Bot Detection." WSDM 2024.
2. Yang et al. "SEBot: Structural Entropy Guided Multi-View Contrastive Learning for Social Bot Detection." KDD 2024.
3. Lin & Zhou. "BotBR: Social Bot Detection with Balanced Feature Fusion and Reliability-Enhanced Graph Learning." SIGIR 2025.
4. Gao et al. "Higher-Order Information Matters: A Representation Learning Approach for Social Bot Detection." CIKM 2025.
5. Kim et al. "TrustSGCN: Trustworthiness-Driven Graph Convolutional Networks for Signed Network Embedding." arXiv 2023.
6. "TrustGTN: A Social Network Trust Evaluation Method Based on Heterogeneous Graph Neural Network." MDPI Computers 2025.
7. "DGTEN: A Robust Deep Gaussian based Graph Neural Network for Dynamic Trust Evaluation." arXiv 2025.
8. He et al. "Can Large Language Models Improve the Adversarial Robustness of Graph Neural Networks?" arXiv 2024.
9. "Unlocking Graph Structure Learning with Tree-Guided Large Language Models." arXiv 2025.
10. "LGB: Language Model and Graph Neural Network-Driven Social Bot Detection." arXiv 2024.
11. "Graph Rewiring with Gumbel-Softmax." arXiv 2025.
12. Karhadkar et al. "Probabilistic Graph Rewiring via Virtual Nodes." arXiv 2024.
13. "Adaptive Edge Learning for Density-Aware Graph Generation." arXiv 2026.
14. "Sparse Bayesian Message Passing under Structural Uncertainty." arXiv 2026.
15. "Uncertainty Modeling in Graph Neural Networks via Stochastic Differential Equations." NeurIPS 2024.
16. "G-DeltaUQ: Accurate and Scalable Estimation of Epistemic Uncertainty for GNNs." ICLR 2024.
17. "Enhance GNNs with Reliable Confidence Estimation via Adversarial Calibration Learning." arXiv 2025.
18. "Graph Structure Learning with Interpretable Bayesian Neural Networks." arXiv 2024.
19. "RABot: Reinforcement-Guided Graph Augmentation for Imbalanced and Noisy Social Bot Detection." arXiv 2026.
20. "RMNP: Certainly Bot Or Not? Trustworthy Social Bot Detection via Robust Multi-Modal Neural Processes." arXiv 2025.
21. "BotDGT: Dynamicity-aware Social Bot Detection with Dynamic Graph Transformers." arXiv 2024.
22. "BotHP: Boosting Bot Detection via Heterophily-Aware Representation Learning." arXiv 2025.
23. "BotSCL: Heterophily-aware Social Bot Detection with Supervised Contrastive Learning." arXiv 2023.
24. "Pruning Graphs by Adversarial Robustness Evaluation to Strengthen GNN Defenses." arXiv 2024.
25. "Structure-Adaptive Graph Neural Nets via Adversarial Synthesis and Self-Corrective Propagation." arXiv 2026.
26. "Topologically-Stabilized Graph Neural Networks." arXiv 2024.
27. "Robust Graph Learning via Prior-Free Diffusion-Based Structure Purification." arXiv 2025.
28. "BetaExplainNN: A Probabilistic Method to Explain Graph Neural Networks." arXiv 2024.
