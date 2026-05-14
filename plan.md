# Conformal Estimator - Local Rewriter Research and Code Plan

Version: 2026-05-10

This document is the implementation-facing research plan for the current Stage 3 work. It does not redefine the already completed LM + GNN detector. The scope begins after frozen LM/RoBERTa embeddings and frozen GNN outputs are available, and ends at a trusted refined ego-graph artifact that later branches can consume.

The two downstream branches are intentionally left as candidates:

- GCL/GSL branch: use refined graph artifacts for graph contrastive learning or graph structure learning style evaluation.
- LLM branch: serialize refined ego evidence for hard-node selective LLM embedding/refiner.

The current code task is only the shared prefix:

```text
RoBERTa node embeddings + original graph + frozen GNN outputs
  -> interactive conformal post-hoc ego-quality estimator
  -> hard-node candidates
  -> local semantic/structural retrieval
  -> text-confirmed local graph rewriter
  -> trusted ego-evidence/refined-edge artifact
```

## 1. Core Research Thought

The main claim should be narrow:

> Social bot errors often concentrate in local graph contexts where the graph signal is low quality, text and structure disagree, or bot-human camouflage edges are diagnostically important. We therefore use conformal ego-quality signals to trigger local refinement, then use text-confirmed local rewriting to soft-reweight existing ego edges while preserving suspicious or virtual edges as evidence.

The claim is not:

- not a new conformal prediction theory;
- not a full-graph structure learning method;
- not binary edge reliability learning as the main novelty;
- not LLM-as-final-predictor;
- not proof that LLMs understand graph structure.

The contribution should be written as a single shared primitive:

```text
graph-quality-guided, evidence-preserving local ego refinement
```

The downstream GCL/GSL and LLM branches are consumers of the same refined ego artifact, not separate main mechanisms in the first implementation.

## 2. Data Flow and Module Contracts

### 2.1 Inputs

Required frozen inputs:

```text
X_R        : RoBERTa or Stage 1 LM node embeddings
A          : original graph edge_index and optional edge_type
P_GNN      : frozen GNN posterior probabilities
logits_GNN : frozen GNN logits
h_GNN      : optional frozen GNN node representation
y_train    : training labels
y_valid    : validation/calibration labels
```

Forbidden inputs for fitting estimator thresholds or modifier supervision:

```text
y_test
test-node correctness
future LLM outcomes
future repair outcomes
```

### 2.2 Outputs

The Stage 3 shared artifact must contain:

```text
estimator_manifest:
  prediction_sets
  set_size
  coverage_margin
  abstain_risk
  calibration_metadata

hard_node_manifest:
  selected_nodes
  selection_budget
  selection_source
  residual-risk provenance if used

retrieval_manifest:
  semantic_candidates
  structural_candidates
  propagation_candidates
  virtual_context_candidates

rewrite_artifact:
  refined_edge_index
  refined_edge_weight
  propagation_edges
  virtual_context_edges
  evidence_edges
  before_after_quality
  rollback_status
  provenance
```

Important contract:

```text
propagation_edges:
  existing ego edges only; eligible for soft reweight.

virtual_context_edges:
  semantic or structural candidates not written to the refined propagation graph.
  They are evidence only and may be consumed by GCL diagnostics or LLM prompts.
```

## 3. Module A - Conformal Ego-Quality Estimator

### Problem It Solves

The estimator decides which ego graphs are low confidence or low quality enough to enter local refinement. It must produce structured uncertainty signals, not just a scalar confidence score.

### Literature Basis

- [CF-GNN, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/54a1495b06c4ee2f07184afb9a37abda-Abstract-Conference.html): supports conformalized GNN prediction sets. Transfer the prediction-set interface and calibration split discipline.
- [DAPS/NAPS, ICML 2023](https://proceedings.mlr.press/v202/h-zargarbashi23a.html): supports graph-aware aggregation of conformity/nonconformity information. Transfer the idea that graph structure can inform conformal efficiency.
- [SNAPS, NeurIPS 2024](https://openreview.net/forum?id=iBZSOh027z): supports similarity- and neighborhood-aware conformal prediction. Transfer the idea that semantic similarity and structure can later refine the estimator.
- [GATS, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/5975754c7650dfee0682e06e1fec0522-Abstract-Conference.html) and [CaGCN, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/c7a9f13a6c0940277d46706c7ca32601-Abstract.html): use as calibration baselines, not the main interface.

### Boundary and Mismatch

CF-GNN/DAPS/NAPS/SNAPS target uncertainty and coverage for graph node classification. Our target is not only uncertainty reporting, but also a trigger for local ego-graph refinement in social bot detection. Therefore:

- do not claim new coverage theory;
- do not claim exchangeability if inductive message passing violates it;
- do not fit thresholds on test labels;
- do not require full DAPS/NAPS diffusion in v1.

### Migration

Implement a minimal conformal estimator:

```text
s(v, y) = 1 - P_GNN(y | v)
q_hat = calibration quantile of s(v, y_v) on valid/calibration split
prediction_set(v) = { y : 1 - P_GNN(y | v) <= q_hat }
```

Optional v1 graph smoothing:

```text
P_smooth(v) = (1 - beta) * P_GNN(v) + beta * mean_{u in N(v)} P_GNN(u)
```

The smoothing is an engineering heuristic inspired by DAPS/NAPS. It must be labelled as heuristic unless full conformal assumptions are implemented.

### Code Requirements

Add or complete estimator mode:

```text
graph_conformal_set_estimator
```

Expected API:

```python
fit(logits, probs, labels, train_idx, val_idx, edge_index=None, node_repr=None)
score(logits, probs, edge_index=None, node_repr=None)
build_manifest(logits, probs, labels, val_idx, test_idx, edge_index=None, node_repr=None)
```

Manifest must include:

```text
prediction_sets
set_size
coverage_margin
abstain_risk
conformal_nonconformity_threshold
calibration_metadata.fit_scope = "validation_split_labels_only"
calibration_metadata.test_labels_used_for_threshold = false
```

### Acceptance Criteria

- Unit test proves threshold changes when validation labels change.
- Unit test proves changing test labels does not change threshold or prediction sets.
- Parser accepts `--estimator_mode graph_conformal_set_estimator`.
- Manifest contract is JSON-serializable.
- No external repo vendoring or new dependency is introduced.

## 4. Module B - Hard-Node Candidate Router

### Problem It Solves

The router decides where refinement is allowed. It is a budgeted candidate selector, not a second bot detector.

### Literature Basis

- [GLANCE](https://openreview.net/forum?id=oODFyykHF5): selective use of LLM/GNN-LLM fusion for nodes where context helps.
- [LOGIN, WSDM 2025](https://doi.org/10.1145/3701551.3703488): LLM as consultant for uncertain/spotted nodes.
- [GATS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/5975754c7650dfee0682e06e1fec0522-Abstract-Conference.html): graph calibration requires more than MSP/entropy.

### Boundary and Mismatch

GLANCE and LOGIN focus on when to call or consult an LLM. We are first deciding when local graph refinement is warranted. Therefore:

- transfer selective routing, not their full LLM training loop;
- do not use LLM outcomes for router fitting in v1;
- do not let the router become a trainable second classifier unless explicitly ablated.

### Migration

Recommended v1:

```text
candidate_pool = existing residual-risk candidates if available
rank_score(v) = abstain_risk(v), with set_size and coverage_margin as tie-breakers
hard_nodes = top budget within candidate_pool
```

If residual-risk candidates are absent:

```text
hard_nodes = top budget by abstain_risk among validation-frozen eligible nodes
```

### Code Requirements

Create a `hard_node_manifest`:

```text
selected_nodes
budget
rank_score_name
source_manifest
tie_breakers
selection_split_policy
```

### Acceptance Criteria

- Selection is deterministic for the same inputs.
- Test labels are never used for selecting hard nodes.
- Manifest records whether residual-risk candidates were used.
- Budgeted selection size is bounded by configured ratio.

## 5. Module C - Local Ego Retrieval

### Problem It Solves

For each hard node, local retrieval produces candidates for rewriting and evidence. It prevents the method from blindly modifying all neighbors or dumping raw ego graphs into downstream modules.

### Literature Basis

- [SKETCH/Taming, ACL 2025](https://aclanthology.org/2025.acl-long.173/): decouples semantic aggregation and structural aggregation in text-attributed graphs.
- [GAugLLM, KDD 2024](https://arxiv.org/abs/2406.11945): argues text and graph are not naturally aligned; edge modification should combine structural candidates and textual commonality.
- [CTGL, COLING 2025](https://aclanthology.org/2025.coling-main.722/): coupled text-graph augmentation shows text and graph can be complementary.
- [SNAPS](https://openreview.net/forum?id=iBZSOh027z): similarity and structural neighborhood can jointly improve conformal efficiency.

### Boundary and Mismatch

SKETCH/Taming learns text representations with decoupled aggregation. GAugLLM uses LLM-based augmentation for graph contrastive learning. CTGL targets coupled augmentations for TAG learning. Our task is social bot local rewriting after a frozen detector. Therefore:

- do not implement full SKETCH decoupled training in v1;
- do not implement GAugLLM contrastive objective in the shared prefix;
- do not hard-add semantic candidates to the graph;
- do not call this a general TAG learner.

### Migration

Use GAugLLM-style logic:

```text
structure proposes candidates;
text confirms, weakens, or marks conflict;
estimator controls whether local rewrite is allowed;
only existing ego edges become propagation rewrite targets;
non-existing candidates remain virtual evidence.
```

Candidate sets:

```text
C_struct_existing(v):
  existing edges in L-hop ego graph.

C_struct_virtual(v):
  high proximity but non-existing structural candidates from k-hop/PPR/Jaccard-style local ranking.

C_sem_virtual(v):
  top-k RoBERTa embedding kNN not already connected to v.
```

Recommended v1:

```text
propagation_candidates = C_struct_existing(v)
virtual_context_candidates = C_struct_virtual(v) union C_sem_virtual(v)
```

### Code Requirements

Implement retrieval as a module that returns structured records:

```python
retrieve_local_ego_candidates(
    node_id,
    edge_index,
    node_repr,
    risk_score,
    hop=1,
    semantic_top_k=8,
    structural_top_k=8,
)
```

Each candidate record:

```text
src
dst
candidate_type: propagation | virtual_context
source: existing_edge | semantic_knn | structural_proximity
semantic_score
structural_score
writes_to_refined_graph
```

### Acceptance Criteria

- Existing edges are the only records with `writes_to_refined_graph=true`.
- Semantic kNN non-edges never appear in `refined_edge_index`.
- Retrieval output is stable under fixed inputs.
- Tests cover isolated nodes, duplicate edges, self-loops, and directed edge direction.

## 6. Module D - Text-Confirmed Local Rewriter

### Problem It Solves

The rewriter changes local propagation strength while preserving suspicious evidence. It is the core mechanism that bridges conformal quality and downstream consumers.

### Literature Basis

- [BotBR, SIGIR 2025](https://doi.org/10.1145/3726302.3729908): shows edge reliability and graph modification matter in social bot detection, but occupies binary reliability plus graph contrastive learning space.
- [BECE, TNNLS 2025](https://ieeexplore.ieee.org/document/10530431/): models edge confidence/noise in social bot detection, but is also reliability-centered.
- [GAugLLM](https://arxiv.org/abs/2406.11945): supports structural candidate plus textual commonality for edge modification.
- [GNNGuard, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/690d83983a63aa1818423fd6edd3bfdb-Abstract.html): supports learned edge weighting.
- [GNNExplainer, NeurIPS 2019](https://papers.nips.cc/paper_files/paper/2019/hash/d80b7040b773199015de6d3b4293c8ff-Abstract.html), [PGExplainer, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/e37b08dd3015330dcbb5d6663667b8b8-Abstract.html), and [GraphMask](https://arxiv.org/abs/2010.00577): support edge masks/importance as a valid edge-level signal.

### Boundary and Mismatch

BotBR/BECE already cover much of the social-bot edge reliability space. GNNGuard is adversarial-defense oriented. Explainers produce explanations, not graph repair. Therefore:

- do not frame the main method as binary reliable/unreliable edge detection;
- do not claim causal explanation from edge masks;
- do not delete suspicious bot-human edges by default;
- do not make four edge roles the fixed final method before ablation.

### Migration

Preferred v1 main output:

```text
continuous edge utility u_edge(e)
```

Role bins are artifact labels for evidence, not the primary training target:

```text
supportive / suspicious / neutral / uncertain
```

GAugLLM-style scoring should be implemented as a coupled score, not a hand-stacked feature dump:

```text
z_text(e)  = PairTextSignal(X_R[v], X_R[u])
z_graph(e) = PairGraphSignal(Ego_v, e, P_GNN, Q_phi)
u_edge(e)  = CoupledModifier(z_text(e), z_graph(e), Q_phi(v))
```

For v1 without a learnable head, use a transparent deterministic utility as a scaffold:

```text
u_edge(e) = normalize(
    a * semantic_agreement(e)
  + b * posterior_agreement(e)
  + c * structural_support(e)
)
```

The deterministic scaffold must be labelled as a baseline/scaffold. The paper-facing method should be decided by ablation:

```text
binary reliability vs continuous utility vs evidence role projection
```

### Counterfactual Pseudo Supervision

If a learnable modifier is implemented later, pseudo labels must be generated only on train/valid:

```text
Delta_down(e) = Loss_v(A_v with e downweighted) - Loss_v(A_v)
Delta_drop(e) = Loss_v(A_v with e removed)      - Loss_v(A_v)
Delta_up(e)   = Loss_v(A_v with e upweighted)   - Loss_v(A_v)
```

Possible targets:

```text
continuous utility:
  u_edge(e) = normalized counterfactual gain/loss

binary reliability:
  useful vs harmful or ambiguous

role bins:
  supportive / suspicious / neutral / uncertain for evidence only
```

Do not use:

```text
same-label edge = reliable
different-label edge = unreliable
```

This rule fails in bot-human camouflage and can delete important evidence.

### Code Requirements

Rewriter API:

```python
rewrite_local_ego(
    node_id,
    propagation_candidates,
    virtual_context_candidates,
    edge_index,
    edge_weight,
    node_repr,
    gnn_probs,
    conformal_quality,
)
```

Output:

```text
refined_edge_index
refined_edge_weight
edge_utility
evidence_role_bins
virtual_context_edges
rewrite_metadata
```

### Acceptance Criteria

- Only existing propagation edges can change edge weight.
- No virtual context candidate is written to `refined_edge_index`.
- Suspicious/low-utility edges remain in `evidence_edges`.
- Edge weights are clipped by configured budget.
- Rewriter records before/after weight and provenance.
- Deterministic scaffold and learnable modifier are separable modes.

## 7. Module E - Soft Rewrite and Rollback

### Problem It Solves

Soft rewrite lets the method adjust local propagation without irreversible deletion or full graph mutation.

### Literature Basis

- [LDS, ICML 2019](https://proceedings.mlr.press/v97/franceschi19a.html): graph structure can be represented as learnable edge probabilities.
- [Pro-GNN, KDD 2020](https://www.kdd.org/kdd2020/accepted-papers/view/graph-structure-learning-for-robust-graph-neural-networks.html): robust graph structure learning can repair noisy graphs.
- [GNNGuard](https://proceedings.neurips.cc/paper/2020/hash/690d83983a63aa1818423fd6edd3bfdb-Abstract.html): soft edge weighting can defend or stabilize GNN message passing.
- [IDGL, NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/e05c7ba4e087beea9410929698dc41a6-Abstract.html): iterative graph learning motivates limited accept/rollback loops.

### Boundary and Mismatch

LDS/Pro-GNN/IDGL are full graph learning methods, often end-to-end. Our method is post-hoc, local, budgeted, and artifact-producing. Therefore:

- do not implement full-graph bilevel structure learning in v1;
- do not claim adversarial robustness;
- do not rewrite the stored main graph;
- do not iterate beyond two rounds without explicit ablation.

### Migration

Soft rewrite formula:

```text
w_e' = w_e * clip(
    1 + lambda * budget(v) * u_edge(e) * confidence_gate(e),
    1 - lambda,
    1 + lambda
)
```

Where:

```text
budget(v) comes from conformal abstain_risk or selected hard-node budget.
u_edge(e) comes from modifier.
confidence_gate(e) reduces rewrite when modifier uncertainty is high.
```

Rollback:

```text
accept if:
  post_quality does not worsen
  set_size does not increase
  local uncertainty/risk does not increase beyond tolerance
else:
  keep original local weights but preserve failed rewrite as diagnostic artifact
```

### Code Requirements

Implement v1 with:

```text
max_iter = 1
rollback = true
main_graph_mutated = false
```

### Acceptance Criteria

- Tests prove original edge_index object is not modified.
- Tests prove output artifact contains refined edge weights.
- Tests prove rollback restores original local weights when quality worsens.
- Tests prove `max_iter > 2` is rejected or marked experimental.

## 8. Module F - Ego-Evidence Artifact for Future Consumers

### Problem It Solves

The refined graph must be usable by both future branches without changing upstream code:

- GCL/GSL consumer can use refined edge weights.
- LLM consumer can use evidence records.

### Literature Basis

- [BotBR](https://doi.org/10.1145/3726302.3729908): supports using modified graphs and contrastive objectives in social bot detection.
- [GAugLLM](https://arxiv.org/abs/2406.11945): supports graph augmentation as a contrastive learning component.
- [GraphText](https://arxiv.org/abs/2310.01089): supports graph-to-text serialization for text-attributed graphs.
- [GLANCE](https://openreview.net/forum?id=oODFyykHF5): supports selective LLM/GNN fusion.
- [LOGIN](https://doi.org/10.1145/3701551.3703488): supports concise evidence prompt/consultant use.
- [Graph Meets LLMs Survey, IJCAI 2024](https://www.ijcai.org/proceedings/2024/898): provides the LLM-as-enhancer framing.

### Boundary and Mismatch

BotBR/GAugLLM train contrastive graph objectives. GraphText/GLANCE/LOGIN use LLMs or graph text. Our shared artifact should not decide which branch is the final method. Therefore:

- do not implement GCL/GSL as part of the conformal-rewriter prefix;
- do not call LLM in the shared prefix;
- do not output LLM direct labels as main prediction;
- do not dump the full ego graph as raw prompt.

### Migration

Artifact format should support two consumers:

```text
for GCL/GSL:
  refined_edge_index
  refined_edge_weight
  original_edge_weight
  accepted_rewrite_mask

for LLM:
  top supportive/high-utility edges
  top suspicious/low-utility edges
  uncertain edges
  semantic virtual context
  structural virtual context
  quality summary before/after rewrite
```

### Acceptance Criteria

- Artifact has a stable schema version.
- LLM fields are evidence records, not labels.
- GCL/GSL fields contain enough graph tensors for later consumption.
- No LLM API call is made in this prefix.

## 9. Implementation Plan

### Step 0 - Documentation and Boundary Lock

Files:

```text
plan.md
.agents/PLANS.md
```

Requirements:

- Record that this task implements conformal estimator - local rewriter prefix only.
- Record that GCL/GSL and LLM are downstream consumers, not v1 prefix implementation.
- Record no new dependencies.

### Step 1 - Estimator Contract

Files:

```text
LLMbot/baseline/core/estimators.py
LLMbot/baseline/core/parser_args.py
LLMbot/baseline/core/model_building.py
LLMbot/baseline/core/tests/
```

Tasks:

- Complete `graph_conformal_set_estimator` if missing.
- Ensure parser and metadata expose the mode.
- Build manifest with prediction sets and calibration metadata.

Tests:

- threshold uses valid/calibration labels only;
- test label mutation does not affect threshold;
- manifest has required keys;
- no dependency addition.

### Step 2 - Stage-Level Estimator Integration

Files:

```text
LLMbot/baseline/core/trainer.py
LLMbot/baseline/core/tests/test_stage_scaffolding.py
```

Tasks:

- Add a narrow path for `--estimator_mode graph_conformal_set_estimator`.
- It should consume frozen G0/Stage 1 outputs and original graph.
- It should not require GATS gate unless the baseline suite explicitly requests GATS.

Tests:

- `estimator_matrix` with conformal mode writes `risk_manifest.json`.
- Dependency manifest records frozen G0 and graph inputs.
- GATS gate is not required for conformal mode.

### Step 3 - Local Ego Retrieval

Files:

```text
LLMbot/baseline/core/operators.py
LLMbot/baseline/core/tests/
```

Tasks:

- Implement candidate retrieval as a separate helper or class.
- Separate propagation and virtual context candidates.
- Support semantic kNN from RoBERTa/node_repr and structural candidates from ego graph.

Tests:

- propagation candidates are existing edges only;
- semantic non-edges are virtual only;
- virtual context edges are not written to refined graph;
- isolated-node case returns empty propagation and possible semantic virtual context.

### Step 4 - Local Rewriter

Files:

```text
LLMbot/baseline/core/operators.py
LLMbot/baseline/core/tests/
```

Tasks:

- Implement continuous utility scaffold.
- Produce refined local edge weights.
- Preserve evidence role bins as artifact fields.
- Do not change prediction probabilities in this prefix unless a separate consumer explicitly uses them.

Tests:

- supportive/high-utility edges increase weight within clip range;
- suspicious/low-utility edges decrease weight within clip range;
- neutral/uncertain edges remain close to original;
- edge weights are unchanged for non-hard nodes;
- virtual candidates never appear in refined edge index.

### Step 5 - Repair Stage Artifact

Files:

```text
LLMbot/baseline/core/trainer.py
LLMbot/baseline/core/tests/test_stage_scaffolding.py
```

Tasks:

- Allow `repair_matrix` with `--repair_mode ego_refinement` to load conformal risk manifest and frozen graph outputs.
- Write:

```text
ego_refinement_artifact.json
refined_edge_index.pt
refined_edge_weight.pt
survivor_manifest.json
dependency_manifest.json
```

Tests:

- stage writes all artifacts;
- dependency manifest includes estimator risk manifest and frozen G0 outputs;
- no LLM calls;
- main graph mutation flag is false.

### Step 6 - Optional Consumer Stubs Only

Files:

```text
LLMbot/baseline/core/model_building.py
LLMbot/baseline/core/parser_args.py
```

Tasks:

- Add downstream mode names only if needed for interface planning.
- Do not implement GCL/GSL or LLM branch in this prefix.

Acceptance:

- Any stub must fail loudly if invoked without implementation.
- Documentation must say downstream branch is not implemented in this step.

## 10. Experiment Plan After Implementation

The first engineering implementation does not prove the research claim. After code validation, run experiments in this order:

### E1 - Estimator Trigger

```text
MSP/entropy baseline
GATS/CaGCN-style calibration baseline
graph_conformal_set_estimator without graph smoothing
graph_conformal_set_estimator with graph smoothing
```

Goal: test whether conformal prediction-set signals are better hard-node triggers.

### E2 - Retrieval

```text
no retrieval
semantic-only
structural-only
semantic + structural uncoupled
GAugLLM-style structural propose + text confirm
```

Goal: test whether text-confirmed retrieval improves local refinement over single-source candidates.

### E3 - Rewriter

```text
no rewrite
binary reliability
continuous utility
role projection as evidence bins
hard delete
hard add
soft reweight existing edges
```

Goal: test whether soft local rewriting is more robust than hard graph mutation and sufficiently distinct from BotBR/BECE.

### E4 - Artifact Consumers

```text
artifact-only no LLM
GCL/GSL on refined graph
hard-node evidence LLM embedding + refiner
hard-node LLM direct prediction diagnostic only
```

Goal: decide whether downstream value comes from graph learning, LLM enhancement, or the shared rewriter.

## 11. Metrics

Main metrics:

```text
Accuracy
Macro-F1
bot-F1
human-F1
```

Calibration and uncertainty:

```text
ECE
Brier
AURC
average set size
singleton hit ratio
coverage by split/subgroup if valid
```

Stage 3 diagnostics:

```text
hard-node ratio
rewrite acceptance ratio
average edge weight delta
virtual context count
low-quality ego slice
text-graph disagreement slice
bot-human mixed-neighborhood slice
suspicious-edge-rich slice
LLM call ratio if LLM branch is enabled
```

## 12. Non-Negotiable Constraints

- No test labels for conformal threshold fitting.
- No test labels for edge modifier pseudo supervision.
- No full graph mutation in v1.
- No hard add of semantic candidates to main graph in v1.
- No binary reliability headline claim.
- No LLM call in conformal-rewriter prefix.
- No new dependencies in v1.
- No external repo vendoring.
- No experimental claim from unit tests or smoke tests.
- Any future GCL/GSL or LLM branch must consume the same artifact and be ablated against artifact-only.

## 13. Potential Improvements

### v1.1

- Add DAPS/NAPS-style graph diffusion for nonconformity scores.
- Add SNAPS-style semantic kNN aggregation using RoBERTa embeddings.
- Add rollback based on before/after conformal set size and coverage margin.

### v2

- Learn `u_edge(e)` from train/valid-only counterfactual pseudo supervision.
- Compare deterministic utility vs learned modifier.
- Add two-step iterative estimator loop with rollback.

### v3

- Add GCL/GSL consumer branch using refined graph.
- Add hard-node selective LLM embedding/refiner branch.
- Compare whether the refined artifact benefits GCL/GSL, LLM, or both.

## 14. Final Recommended Implementation Slice

Implement now:

```text
graph_conformal_set_estimator
hard-node manifest
local ego retrieval
continuous utility scaffold
soft reweight existing edges
trusted ego-evidence artifact
stage-level artifact writing
focused tests
```

Do not implement now:

```text
full GCL/GSL training
LLM API calls
LLM direct prediction
full graph structure learning
hard add/delete as main method
learned counterfactual modifier unless deterministic scaffold is validated
```

This slice is the right next step because it creates the shared object that both candidate directions need, while keeping the research boundary clean: the current novelty is conformal-quality-guided local ego rewriting, and the use of the modified graph is an experimentally decided downstream branch.
