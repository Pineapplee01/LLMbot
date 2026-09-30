# High-Order Graph Consumption

Claim role: HyperScan-style second-view graph construction, low/high view
consumption, routed-only high-order consumers, and risk-gated high-order
residual diagnostics.

Current owner modules:

- `GNNs.py`
- `hypergnn.py`
- `trainer_preparation.py`
- `model_building.py`

Canonical vocabulary:

- `x_low`
- `x_new`
- `x_high`
- `second_view_edges`
- `routed_highpass_bundle`
- `center_node_ids`

Migration note: do not split this claim before the graph detector inputs,
`x_low`, `x_new`, `x_high`, and fusion contracts are stable across runner
scripts and manifests.
