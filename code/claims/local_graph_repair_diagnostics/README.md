# Local Graph Repair Diagnostics

Claim role: local graph conflict detection, structural prune diagnostics,
DIGNN-style conflict refinement, and graph repair-vs-deferral analysis.

Current owner modules:

- `trainer_graph.py`
- `trainer_dignn_conflict.py`
- `operators.py`

Canonical vocabulary:

- `conflict_score`
- `pruned_edge_index`
- `repair_delta`
- `structural_view_logits`
- `deferral_decision`

Migration note: this folder is the best first candidate for a real module move
after import shims exist, because `trainer_graph.py` already owns much of this
surface.
