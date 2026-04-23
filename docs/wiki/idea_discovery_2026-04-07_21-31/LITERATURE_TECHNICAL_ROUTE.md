# Literature-Informed Technical Route: Edge Trust Learning Framework

**Date**: 2026-04-10  
**Purpose**: Refine technical route based on high-quality literature search  
**Models**: Claude Opus 4.6 + GPT-5.4 (xhigh)

---

## Literature Landscape Summary

### Direct Competitors (Must Differentiate)

| Paper | Venue | Core Idea | Overlap | Our Gap |
|-------|-------|-----------|---------|---------|
| [UnGSL](https://openreview.net/forum?id=AB22PNdfwP) | WWW 2025 | Node uncertainty → adjust directional connection strengths | Very close: uncertainty-guided edge reweighting | No LLM/semantic signals; purely graph-based |
| [EWGSL](https://arxiv.org/html/2503.12157v2) | arXiv 2025 | Edge weight learning + attention denoising for noisy graphs | Edge weight learning for node classification | No semantic guidance; no bot detection |
| [BotBR](https://arxiv.org/html/) | SIGIR 2025 | Balanced feature fusion + reliability-enhanced graph learning | Reliability-aware graph learning for bots | Feature-level reliability, not edge-level trust |
| [RABot](https://arxiv.org/html/2602.21749v1) | arXiv 2026 | RL-based graph augmentation for imbalanced/noisy bot detection | Graph augmentation for bot detection | RL-based, not calibration-guided |

### Supporting Literature

| Paper | Venue | Relevance |
|-------|-------|-----------|
| [Rethinking GSL in Era of LLMs](https://arxiv.org/abs/2503.21223v2) | arXiv 2025 | Survey supporting "LLM-guided GSL" positioning |
| [ES-GNN](https://arxiv.org/html/2205.13700v5) | AAAI 2024 | Edge splitting into positive/negative channels |
| [Robust GSL under Heterophily](https://pubmed.ncbi.nlm.nih.gov/39893803/) | Neural Networks 2025 | High-pass filter + adaptive norm for heterophily |
| [Task-driven Heterophilic GSL](https://arxiv.org/html/2512.23406v1) | arXiv 2025 | Task-driven GSL for heterophilic graphs |
| [Probabilistic Graph Rewiring](https://arxiv.org/html/2405.17311v1) | NeurIPS 2024 | Differentiable end-to-end graph rewiring |
| [Sparse Bayesian Message Passing](https://arxiv.org/html/2601.01207v1) | arXiv 2026 | Bayesian MP under structural uncertainty |
| [Uncertainty-Aware Robust Learning on Noisy Graphs](https://arxiv.org/html/2306.08210v1) | arXiv 2023 | Uncertainty-aware learning on noisy graphs |
| [SEBot](https://arxiv.org/html/2405.11225v1) | KDD 2024 | Structural entropy for graph optimization |

---

## Novelty Analysis

### What Makes Us Different from "UnGSL + LLM Features"

The key insight from GPT-5.4 review: **our unique contribution is NOT the LLM itself, but explicit edge-level trust posterior learning from two independent views**.

| Aspect | UnGSL | Our Method |
|--------|-------|-----------|
| Trust signal | Node uncertainty only | Cross-view: semantic compatibility + structural uncertainty |
| Direction | Adjusts connection strengths | Asymmetric: source confidence → target uncertainty |
| Trigger | All edges uniformly | Disagreement-aware: only repair where views conflict |
| Semantic | None | LLM-derived semantic prior per directed edge |
| Heterophily | Not explicitly addressed | Trust ≠ same-label similarity; conflict-aware |

### Biggest Novelty Threats

1. "This is UnGSL with LLM features and an edge scorer" → Counter: edge-level cross-view trust posterior, not node-level uncertainty
2. "Support-edge addition is doing the work; trust model is cosmetic" → Counter: ablation showing soft reweight alone improves
3. Misuse of "calibration" → Rename to "cross-view edge trust estimation"

---

## Refined Technical Route (GPT-5.4 Recommended)

### Core Claim
> Directed edge trust posterior = semantic prior + uncertainty transfer + structural conflict, with minimal-edit graph repair.

### Method Architecture

```
Semantic Branch:
  h_i = LLM(text/profile of node i)  [frozen encoder + small projector]
  s_ij = sigmoid(MLP_s([h_i, h_j, h_i - h_j, h_i * h_j]))  [directional: s_ij ≠ s_ji]

Graph Branch:
  2-layer weighted RGT → class probs p_i, node uncertainty u_i
  Edge conflict: c_ij = JS(p_i, p_j) or 0.5*||p_i - p_j||_1
  Directional utility: b_ij = (1 - u_i) * u_j  ["confident source, uncertain target"]

Joint Trust Head:
  t_ij = sigmoid(MonoMLP([s_ij, b_ij, -c_ij, recip_ij, jaccard_ij]))
  Monotonic constraints: trust ↑ with semantic prior + utility, ↓ with conflict

Repair Strategy:
  Observed edges: w_ij = (1 - γ) + γ * t_ij  [soft reweight]
  Isolated/high-risk nodes: add top k=1-2 incoming support edges with high t_ij
  Edit budget: < 1-3% new edges
  Candidates: 2-hop neighbors or top semantic ANN neighbors

Loss:
  L_cls: node classification (CE)
  L_rank: rank high-consensus edges above high-conflict edges
  L_budget: penalize excessive reweighting/addition

Training:
  Stage 1: warm up semantic scorer + GNN separately
  Stage 2: joint optimization of trust + repaired graph classifier
```

### Key Design Choices

1. **Directional semantic prior** `s_ij ≠ s_ji` — captures asymmetric information flow
2. **Uncertainty transfer** `b_ij = (1-u_i)*u_j` — "confident source helps uncertain target"
3. **Monotonic trust MLP** — inductive bias: trust rises with semantic prior, falls with conflict
4. **Soft reweight** `w_ij = (1-γ) + γ*t_ij` — smooth, differentiable, no hard pruning
5. **Minimal support-edge addition** — only for isolated/high-risk, tiny budget

---

## Minimum Experiments for WWW/AAAI

1. **Main comparison** on TwiBot-20 with identical node features for all methods, including `UnGSL + our LLM features` as explicit baseline
2. **Ablation table**: -semantic prior, -uncertainty/conflict, -disagreement, -directionality, -support edges, hard prune vs soft reweight
3. **Robustness test** under synthetic edge corruption (random drop/add/rewire)
4. **Sparse-node analysis**: low-degree / isolated nodes (support-edge story lives or dies here)
5. **Trust analysis**: higher-trust edges have lower conflict and produce better downstream utility
6. **Second dataset** if possible (TwiBot-22 or MGTAB)

---

## Implementation Mapping to Current Codebase

| New Component | Reuses From |
|--------------|-------------|
| Semantic prior `s_ij` | `z_sem` from `semantic_ib_edl_head.py` |
| Node uncertainty `u_i` | `u_sem` from semantic head; `q_graph` from `graph_calibration.py` |
| Edge conflict `c_ij` | `_pair_conflict()` from `calibration_graph_rewrite.py` |
| Structural features | `compute_directed_structural_features()` from `utils.py` |
| Relation priors | `build_relation_priors()` from `calibration_graph_rewrite.py` |
| GNN backbone | `GraphRGTExpert` from `graph_rgt.py` (add edge_weight support) |
| Training loop | `DualRouterTrainer` from `dual_router_trainer.py` |

---

## Paper Framing Recommendation

**Title direction**: "Directed Edge Trust Learning for Social Bot Detection on Noisy Heterophilous Graphs"

**NOT**: "Calibration-guided graph rewrite" (too close to old framing)
**NOT**: "LLM-guided graph structure learning" (too generic)
**YES**: "Cross-view directed edge trust estimation with minimal-edit repair"

**Venue**: WWW 2026 (social network integrity track) or AAAI 2026 (broad AI)
