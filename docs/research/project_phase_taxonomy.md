# Project Phase Taxonomy

This file defines the canonical high-level research phase vocabulary for this project.

These labels are planning and communication labels, not CLI `--stage` names. Existing code-facing stage names such as `semantic_source_matrix`, `estimator_matrix`, `repair_matrix`, and `backbone_stress` remain unchanged.

The phase order is flexible. Future work may skip, repeat, merge, split, or rename phases as the research direction changes. When the user states the current phase, that explicit direction overrides any implied order below.

## Canonical Phase Names

| Phase | Canonical Name | Operational Meaning |
| --- | --- | --- |
| A | 阶段 A：semantic encoder → GNN | Build or evaluate the semantic encoder signal that feeds, conditions, or compares with the GNN path. |
| B | 阶段 B：GNN 输出 → post-hoc estimator | Convert GNN outputs, logits, embeddings, or diagnostics into post-hoc reliability, risk, or utility estimators. |
| C | 阶段 C：risk/regime → local semantic enhancement | Use risk or regime signals to improve local semantic representations, semantic logits, or semantic-side evidence. |
| D | 阶段 D：risk/regime → local propagation repair | Use risk or regime signals to locally repair graph propagation, including pruning, rewriting, or re-propagation studies. |
| E | 阶段 E：action selection | Select among available actions, experts, repair modes, abstention, fallback, or no-op choices. |
| F | 阶段 F：same-head refinement | Refine within the same prediction head or output family without changing the high-level action interface. |
| G | 阶段 G：positioning / stress test | Position the method against baselines, stress tests, robustness checks, comparison protocols, and paper-facing framing. |

## Mapping To Current Surfaces

| Research Phase | Current Code or Doc Surface | Notes |
| --- | --- | --- |
| 阶段 A：semantic encoder → GNN | `semantic_source_matrix`, semantic-source notes, Qwen/RoBERTa/IB-EDL source records | Mostly supporting-mainline semantic-source work. |
| 阶段 B：GNN 输出 → post-hoc estimator | `vertical_minimal`, `estimator_matrix`, frozen GNN outputs, risk manifests | Covers post-hoc estimator construction from GNN or dual-expert outputs. |
| 阶段 C：risk/regime → local semantic enhancement | `semantic_matrix`, semantic enhancement experiments | Evidence-dependent; do not claim completion without validated artifacts. |
| 阶段 D：risk/regime → local propagation repair | `repair_matrix`, graph rewrite and propagation repair records | Covers local graph repair, pruning, and re-propagation. |
| 阶段 E：action selection | `selector_matrix`, router/action tables | Covers choosing repair, semantic enhancement, fallback, or no-op. |
| 阶段 F：same-head refinement | Planned or evidence-dependent same-head refinement records | Treat as not complete unless a future manifest/report names concrete evidence. |
| 阶段 G：positioning / stress test | `positioning_matrix`, `backbone_stress`, baseline comparability docs, proposal/review docs | Covers stress testing, unified comparison, and research positioning. |

## Usage Rules

- Use the exact A-G names above in new planning notes, experiment logs, and handoffs.
- Do not rename existing CLI `--stage` values to match these labels.
- Do not rewrite historical records only to change old phase wording.
- If a task spans multiple phases, list every relevant phase explicitly.
- If a current task conflicts with the implied order, follow the user's stated current phase.
