Research Brief
Problem Statement

Current social bot detection methods are strong at the detector level, but they still leave a non-trivial residue of node-level errors. These errors are not uniformly distributed. They concentrate on structurally difficult or semantically conflicted nodes, such as sparse / isolated accounts, low-neighborhood-support accounts, and nodes whose propagated neighborhood evidence appears harmful or misleading. The key research problem is therefore not simply to build another stronger global detector, but to determine whether residual errors exhibit typed failure regimes and whether different regimes require different minimal corrective actions.

I want to formulate social bot detection as a failure-regime-conditioned minimal intervention problem. A strong graph detector remains the default predictor. Then a post-hoc estimator predicts node-wise failure risk and regime evidence, and only for nodes with justified risk do we apply minimal matched correction. The allowed action space is constrained to three actions: No-op, Sparse-Evidence local semantic rescue, and Propagation-Corruption local propagation repair. Final prediction must still come from the same GNN-centric head.

I explicitly do not want this project to drift into any of the following narratives: global semantic-graph fusion, graph rewrite / structure learning, LLM-calling / consultation routing, uncertainty-only intervention, or a second detector that replaces the base graph model. ARIS should treat this as a tightly constrained idea-refinement and experiment-prioritization problem, not as a free-form brainstorm.

Background
Field: Machine Learning / Graph ML / Social Media Integrity
Sub-area: Social bot detection, failure-aware post-hoc intervention, graph reliability under heterophily / sparse evidence
Current working thesis:
Not all residual GNN errors in social bot detection arise from the same cause, so they should not receive the same correction.
Pipeline I want to keep fixed:
raw account data → semantic representation → graph backbone detector → post-hoc failure/regime evidence → action selection → local semantic rescue / local propagation repair → one same-head refinement → final prediction and subgroup evaluation
Key papers / families already identified:
BotRGCN: strong bot-detection heterograph baseline, likely primary graph backbone reference
LGB: global semantic compensation for sparse / isolated nodes; used as a family-level sparse subgroup reference, not necessarily full reproduction
GNNGuard: edge weighting / suppression reference for local propagation repair
LA-GNN: local augmentation / local evidence completion reference for sparse-evidence semantic rescue
GLANCE: binary consult / router family; important as boundary/contrast, not target identity
BotGSL / BotBR / botsay: occupied claim-space around graph structure learning, reliability-enhanced graph learning, and LLM-heavy bot detection
What I already tried:
Built an FRMI-style prototype with:
post-hoc probe features: MSP, entropy, temperature-scaled confidence, in/out-degree, edge-type skew, neighbor prediction inconsistency, h(0) vs h(L) divergence, and neighborhood agreement proxy
FocalSemanticResidual
EgoEdgeReweight
D1–D8 intervention comparison
Observed that probe performance is strong, but operator results are weak and noisy
What didn’t work:
GATv2 underperformed relative to RGCN/BotRGCN-style backbone in my current setting
random frozen semantic projection was not meaningful
RGCN repair path degraded into node-level averaging rather than true per-edge local repair
current three-action implementation was not a true three-action policy; it behaved more like binary intervention + two-way routing
overall macro-F1 is too insensitive when intervention budget is small; targeted metrics are more informative
Constraints
Compute:
Remote GPU server available
Local RTX 4060 available for analysis / small-scale checks
I want reproduction and pilot design to be compute-conscious rather than brute-force
Timeline:
Need a strong, defensible direction quickly
Prefer a claim that can survive top-venue scrutiny over a larger but fuzzier pipeline
Target venue:
CCF-B+ / top-tier ML / Web / NLP-adjacent venue
The paper should read as a social bot detection paper, not as a graph causal inference paper or an LLM routing paper
What I'm Looking For
 New research direction from scratch
 Improvement on existing method: Failure-Regime-Conditioned Minimal Intervention for Social Bot Detection
 Diagnostic study / analysis paper
 Other: Constrained idea refinement under fixed claim boundary

Specifically, I want ARIS to do the following:

Stress-test whether the current frozen claim is still defensible:
social bot detection paper
object of study = failure regimes, not generic hard nodes
answer = regime-conditioned minimal intervention
same-head final prediction
no graph rewrite headline
no global semantic fusion headline
no LLM consultation headline
Refine the strongest defensible implementation within this boundary:
keep finetuned-RoBERTa as the main semantic backbone
use Qwen3-Embedding-8B only as a semantic source for Sparse-Evidence local evidence rescue, not as global backbone and not as graph rewrite guidance
keep graph-side intervention as local propagation repair, not graph rewrite
consider replacing rule-based action selection with post-hoc action-utility estimation or similarly constrained decision logic, but do not drift into full causal policy learning
Prioritize what must be reproduced first:
BotRGCN
GNNGuard-style local repair reference
LA-GNN-style local semantic rescue reference
LGB family-level sparse subgroup comparison
only then revisit FRMI main-table comparisons
Output should help me decide:
which parts of FRMI survive
which parts should be replaced
whether the final paper should remain a three-action intervention paper or degrade into a diagnostic / targeted post-hoc analysis paper
Domain Knowledge
This problem should stay in the TAG-like node classification setting for public benchmarks, but the interpretation is closer to risk-sensitive bot detection than generic text-attributed graph learning.
The most valuable empirical observation so far is not a total-score gain but that residual errors appear concentrated and structured, not uniform.
Sparse-Evidence and Propagation-Corruption should remain the only two active corrective regimes in v1; the third action is no-op.
Sparse-Evidence should mean something like:
low degree / isolated / graph-missing
weak neighborhood support
semantic evidence still useful
Propagation-Corruption should mean something like:
low local neighborhood consistency
semantic-structural conflict
camouflage-heavy or harmful-neighbor dominance
message passing may be net harmful
I do not want the method to become:
another semantic fusion detector
another reliability-enhanced graph learner
another graph structure learning method
another binary LLM consult router
Current preferred conceptual upgrade:
action layer may use utility-aware action selection
but utility estimation must remain post-hoc, constrained, non-label-predictive, and non-headline-causal
Non-Goals
Not a global graph rewrite / rewiring paper
Not a graph structure learning paper
Not an LLM-calling or LLM-consultation paper
Not a general TAG semantic-graph fusion paper
Not an uncertainty-only intervention paper
Not an end-to-end learned router
Not a second detector that predicts bot labels separately from the graph backbone
Not a full causal inference / SCM / graph treatment-effect paper
Not a benchmark-only paper
Not a prompt-engineering paper around Qwen or other LLMs
Existing Results (if any)
Current prototype observations:
probe looked promising and appeared stronger than entropy-only baselines
operator-level results were weak and noisy
likely causes:
semantic operator used a random frozen projection
repair operator was not true local per-edge repair
current three-action policy implementation did not yet match the conceptual design
Practical conclusion from current prototype:
P1-like diagnostic signal seems real
P2/S1-like intervention claim is not yet cleanly tested
Therefore, I do not want ARIS to blindly optimize the current implementation.
I want ARIS to:
refine the claim,
rank reproduction priorities,
identify the strongest defensible replacement for current semantic and repair operators,
tell me whether utility-aware action selection is worth integrating,
propose a clean experiment order under this fixed paper identity.