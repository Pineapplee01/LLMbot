# High-Order Graph Consumption

Claim role: HyperScan-style second-view graph construction, low/high view
consumption, routed-only high-order consumers, and risk-gated high-order
residual diagnostics.

Canonical vocabulary:

- `x_low`
- `x_new`
- `x_high`
- `second_view_edges`
- `routed_highpass_bundle`
- `center_node_ids`

Primary symbols:

- HyperScan and routed high-pass classes currently live in `GNNs.py` and
  `hypergnn.py`.
- Second-view construction logic currently lives in `trainer_preparation.py`.

Shared dependencies:

- `GNNs.py`
- `hypergnn.py`
- `model_building.py`
- `utils/`

Do not move yet:

- `GNNs.py`
- `hypergnn.py`
- `trainer_preparation.py`
- `model_building.py`
