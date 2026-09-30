# Local Graph Repair Diagnostics

Claim role: local graph conflict detection, structural prune diagnostics,
DIGNN-style conflict refinement, and graph repair-vs-deferral analysis.

Canonical vocabulary:

- `conflict_score`
- `pruned_edge_index`
- `repair_delta`
- `structural_view_logits`
- `deferral_decision`

Primary symbols:

- `trainer_dignn_conflict.py` now owns the DIGNN-style conflict-refinement
  implementation in this package.
- `trainer_graph.py` owns graph-stage orchestration and remains top-level in
  this slice.

Shared dependencies:

- `model_building.py`
- `operators.py`
- `utils/`
- `artifact_contracts.py`

Do not move yet:

- `trainer_graph.py`
- `operators.py`
