# Literature Survey: Semantic Operator Alternatives & Edge Repair Mechanisms
**Date:** 2026-04-22 | **Scope:** 2024-2026 papers

---

## TIER 1 — Semantic Operator Alternatives (replacing Ridge regression for LM→GNN hidden state projection)

### T1-1. TTReFT — Test-Time Representation Refinement for Node Classification
- **Authors:** Jiaxin Zhang, Yiqi Wang, Siwei Wang, Xihong Yang, Yu Shi, Songlei Jian, Xinwang Liu, En Zhu
- **Venue/Year:** arXiv 2601.21615, 2026
- **Method:** Low-rank interventions on a sparse subset of node representations selected by uncertainty (high-entropy nodes). Uses an intervention-aware masked autoencoder that adjusts masking probability based on local intervention density. All pre-trained GNN parameters stay frozen.
- **Key result:** 88.77% on Cora (concept_degree split), 100% in-distribution retention vs. up to -25.32% degradation for parameter finetuning methods.
- **Relevance:** **VERY HIGH.** This is almost exactly our problem — post-hoc correction of frozen GNN representations. The low-rank intervention on high-uncertainty nodes is a direct upgrade path from Ridge regression. Instead of projecting LM embeddings to reproduce GNN states, we could inject low-rank residual corrections targeting nodes where the GNN is confused.
- **Adaptable to frozen backbone:** Yes, by design.

### T1-2. UAdapterGNN — Uncertainty-aware Adapter Learning for GNNs
- **Authors:** Bo Jiang, Weijun Zhao, Beibei Wang, Xiao Wang, Jin Tang
- **Venue/Year:** arXiv 2511.18859, 2025
- **Method:** Gaussian probabilistic adapters with mean+variance bottleneck layers on a frozen pre-trained GNN. Reparameterization trick for differentiable sampling. Captures epistemic uncertainty in the adapter itself.
- **Key result:** 72.46% avg ROC-AUC on molecular benchmarks (+1.3-3.5% over deterministic adapters); 4.90% robustness gain under 80% edge perturbation.
- **Relevance:** **HIGH.** The probabilistic adapter framework could replace Ridge regression with an uncertainty-aware projection that naturally downweights unreliable corrections. The variance output provides a built-in confidence signal.
- **Adaptable to frozen backbone:** Yes, explicitly designed for frozen GNN backbones.

### T1-3. HG-Adapter — Dual Adapters for Pre-Trained Heterogeneous GNNs
- **Authors:** Yujie Mo, Runpeng Yu, Xiaofeng Zhu, Xinchao Wang
- **Venue/Year:** arXiv 2411.01155, 2024
- **Method:** Two structure-aware adapters: a homogeneous adapter (same-type node similarity weighting + message passing) and a heterogeneous adapter (cross-type scoring). Low-rank decomposed mapping matrices. Label-propagated contrastive loss.
- **Key result:** 92.7% Macro-F1 on ACM, +0.81% avg over HetGPT across datasets.
- **Relevance:** **HIGH.** Our social graph is heterogeneous (user-tweet-list edges). The dual-adapter design with separate homogeneous/heterogeneous pathways could replace the single Ridge operator with relation-type-aware projections.
- **Adaptable to frozen backbone:** Yes, frozen pre-trained HGNN backbone.

### T1-4. TEA-GLM — Alignment of GNN Representations with LLM Token Embeddings
- **Authors:** Duo Wang, Yuan Zuo, Fengzhi Li, Junjie Wu
- **Venue/Year:** arXiv 2408.14512, 2024
- **Method:** Feature-wise contrastive pretraining aligns GNN node representations to LLM token embedding space via PCA-derived principal components. A linear projector then maps graph representations into K fixed graph token embeddings for LLM consumption.
- **Key result:** 0.848 zero-shot accuracy on Arxiv (vs. 0.793 for LLaGA).
- **Relevance:** **MEDIUM-HIGH.** The contrastive alignment between GNN and LM spaces is directly relevant — instead of Ridge regression (which minimizes MSE and reproduces errors), contrastive alignment preserves semantic structure while allowing the spaces to differ where they should.
- **Adaptable to frozen backbone:** Yes, LLM frozen; GNN pretrained then projector trained.

### T1-5. TAAM — Task-Aware Adaptive Modulation for Continual Graph Learning
- **Authors:** Jingtao Liu, Xinming Zhang
- **Venue/Year:** arXiv 2509.00735, 2025
- **Method:** Neural Synapse Modulators (NSMs) — lightweight modules that dynamically steer a frozen GNN's internal computational flow. Prototype-guided initialization and inference retrieval.
- **Key result:** 97.6% on Cora, 0.0% catastrophic forgetting across 6 benchmarks.
- **Relevance:** **MEDIUM.** The NSM concept of steering frozen GNN computation without modifying weights is architecturally interesting. Could inspire a "semantic modulator" that steers GNN hidden states using LM signals rather than replacing them.
- **Adaptable to frozen backbone:** Yes, by design.

### T1-6. LLMTTT — Test-Time Training on Graphs with LLMs
- **Authors:** Jiaxin Zhang, Yiqi Wang, Xihong Yang, Siwei Wang, et al.
- **Venue/Year:** arXiv 2404.13571, 2024
- **Method:** LLMs generate pseudo-labels for strategically selected test nodes (hybrid active selection: uncertainty + diversity). Two-stage: filtered pseudo-label training then self-training on unlabeled data.
- **Key result:** 88.53% on Cora, 73.82% on OGBN-Arxiv (concept_degree shift).
- **Relevance:** **MEDIUM.** The LLM-as-annotator paradigm is orthogonal to our semantic operator but could complement it — using LM confidence to generate soft targets that guide the correction operator, rather than just projecting embeddings.
- **Adaptable to frozen backbone:** Yes, model-agnostic test-time method.

### T1-7. LGB — Language Model and GNN-Driven Social Bot Detection
- **Authors:** Ming Zhou, Dan Zhang, Yuandong Wang, et al.
- **Venue/Year:** arXiv 2406.08762, 2024
- **Method:** Supervised fine-tuned LM handles isolated/sparse nodes; GNN handles dense nodes. Concatenation + MLP fusion. Addresses the ~55% of social network nodes lacking sufficient connections.
- **Key result:** TwiBot-22: 80.42% acc (+9.98% over MixHop); TwiBot-20: 85.09% acc (+10.95% over SIRAN).
- **Relevance:** **MEDIUM.** Directly in our domain (social bot detection with LM+GNN). Their concat+MLP fusion is a baseline; our semantic operator should outperform this naive approach. Useful as a comparison point.
- **Adaptable to frozen backbone:** Partially — requires LM fine-tuning.

---

## TIER 2 — Edge Repair Mechanisms (improving ego-subgraph edge reweighting)

### T2-1. RABot — Reinforcement-Guided Graph Augmentation for Bot Detection
- **Authors:** Longlong Zhang, Xi Wang, Haotong Du, Yangyi Xu, Zhuo Liu, Yang Liu
- **Venue/Year:** arXiv 2602.21749, 2026
- **Method:** RL-driven edge filtering that dynamically prunes spurious edges during GNN message passing, combined with neighborhood-aware minority-class oversampling via local subgraph interpolation.
- **Key result:** TwiBot-20: 87.92% acc, 88.40% F1 (+1.37% acc over RGT backbone).
- **Relevance:** **VERY HIGH.** Directly addresses edge noise in social bot detection. The RL edge filter is architecture-agnostic and works across GCN/GAT/RGT/RGCN backbones. Could replace our entropy-based edge reweighting with a learned policy.
- **Adaptable to frozen backbone:** Yes, architecture-agnostic augmentation module.

### T2-2. Shapley-Value Graph Sparsification for GNN Inference
- **Authors:** Selahattin Akkas, Ariful Azad
- **Venue/Year:** KDD 2025 Workshop (MLoG-GenAI), 2025
- **Method:** Computes per-node Shapley value explanations (GNNShap), aggregates edge importance scores, prunes negatively-contributing and low-importance edges. No retraining needed.
- **Key result:** Prunes 80% of edges on Cora/PubMed with <2% accuracy drop; 49-65% computation reduction.
- **Relevance:** **HIGH.** Post-hoc, no-retrain edge pruning based on contribution scores. Could replace our disagreement+entropy heuristic with principled Shapley-based edge importance. The "negatively-contributing edge" concept directly maps to our edge repair goal.
- **Adaptable to frozen backbone:** Yes, operates on pre-trained models at inference time.

### T2-3. LASER — Locality-Aware Sequential Rewiring
- **Authors:** Federico Barbero, Ameya Velingker, Amin Saberi, Michael Bronstein, Francesco Di Giovanni
- **Venue/Year:** arXiv 2310.01668 (likely NeurIPS/ICML)
- **Method:** Sequential trajectory of rewiring operations from original graph to increasingly connected versions. Selects edges to add based on per-node connectivity scores (walk counts) with locality constraints (shortest-path distance).
- **Key result:** Peptides-func: 0.6440 AP (vs. 0.5930 baseline); scales to 100k-node graphs.
- **Relevance:** **MEDIUM-HIGH.** The locality-aware, per-node rewiring is conceptually close to our ego-subgraph approach. The sequential trajectory idea (progressive rewiring) could improve our single-shot reweighting.
- **Adaptable to frozen backbone:** Rewiring is a preprocessing step, so yes.

### T2-4. HW-GNN — Homophily-Aware Gaussian-Window Graph Spectral Network
- **Authors:** Zida Liu, Jun Gao, Zhang Ji, Li Zhao
- **Venue/Year:** arXiv 2511.22493, 2025
- **Method:** Learnable Gaussian windows modulate polynomial basis functions to focus on bot-discriminative frequency bands. Homophily ratio guides window parameter learning.
- **Key result:** TwiBot-20: 91.51% F1 (+1.2%); TwiBot-22: 61.95% F1 (+3.2%).
- **Relevance:** **MEDIUM.** In our domain (bot detection) and achieves strong results. The spectral filtering approach is complementary — could inform which frequency bands our edge repair should preserve.
- **Adaptable to frozen backbone:** No, requires end-to-end training.

### T2-5. AdaRC — Mitigating Graph Structure Shifts at Test Time
- **Authors:** Wenxuan Bao, Zhichen Zeng, Zhining Liu, Hanghang Tong, Jingrui He
- **Venue/Year:** arXiv 2410.06976, 2024
- **Method:** Adapts hop-aggregation parameters via prediction-informed clustering (PIC) loss at test time. Controls how GNNs integrate features across different hops without modifying edge weights.
- **Key result:** Up to 31.95% improvement standalone; up to 40.61% combined with existing TTA methods.
- **Relevance:** **MEDIUM.** Addresses structure shift at test time but via aggregation parameters rather than edge weights. Could complement our edge reweighting by also adjusting hop-level mixing.
- **Adaptable to frozen backbone:** Yes, test-time adaptation.

### T2-6. FnRGNN — Distribution-aware Fairness via Edge Reweighting
- **Authors:** SoYoung Park, Sungsu Lim
- **Venue/Year:** CIKM 2025
- **Method:** Hybrid edge weighting: sim(x_i, x_j) * exp(-gamma * I[s_i != s_j]). Soft reweighting preserving connectivity rather than hard pruning.
- **Key result:** German dataset MSE 0.6994 vs. 1.0934 baseline GCN.
- **Relevance:** **LOW-MEDIUM.** The soft edge reweighting formula is elegant and could be adapted — replace the fairness demographic term with our disagreement signal. But the fairness framing is distant from our problem.
- **Adaptable to frozen backbone:** No, in-processing during training.

### T2-7. NCGCN — Clarify Confused Nodes via Separated Learning
- **Authors:** Jiajun Zhou, Shengbo Gong, et al.
- **Venue/Year:** arXiv 2306.02285, 2023
- **Method:** Neighborhood Confusion (NC) metric identifies nodes with heterogeneous neighborhoods. Separate GCN pathways for high-NC vs. low-NC groups with distinct transformation weights.
- **Key result:** 91.64% on Pubmed, 96.64% on Coauthor CS.
- **Relevance:** **MEDIUM.** The NC metric is directly useful — it quantifies exactly the "confused node" problem our edge repair targets. Could use NC as a better node selection criterion than entropy alone.
- **Adaptable to frozen backbone:** No, requires training-time separation.

---

## SYNTHESIS: Top Recommendations

### For Semantic Operator (replacing Ridge regression):
1. **TTReFT** (T1-1): Best fit. Low-rank interventions on high-uncertainty nodes, frozen backbone, post-hoc. Directly replaces Ridge with targeted residual corrections.
2. **UAdapterGNN** (T1-2): Second choice. Probabilistic adapter with built-in uncertainty quantification. Replaces deterministic projection with distribution-aware one.
3. **TEA-GLM** (T1-4): For the alignment loss. Replace MSE (Ridge) with contrastive alignment that preserves semantic structure rather than reproducing GNN errors.

### For Edge Repair (improving ego-subgraph reweighting):
1. **RABot** (T2-1): Best fit. RL-driven edge filtering, same domain (bot detection), architecture-agnostic, proven on TwiBot.
2. **Shapley Sparsification** (T2-2): Principled alternative. Post-hoc Shapley-based edge importance replaces our heuristic disagreement+entropy signal.
3. **LASER** (T2-3): For the progressive rewiring idea. Sequential trajectory instead of single-shot reweighting.
