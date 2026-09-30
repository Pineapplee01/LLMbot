# Conformal Risk Routing

Claim role: local conformal risk, conformal KNN risk, router scoring,
selective prediction, and routed-node selection evidence.

Canonical vocabulary:

- `risk_score`
- `risk_scores`
- `router_score`
- `risk_budget`
- `selected_node_ids`
- `coverage_curve`

Primary symbols:

- `router.py` now owns the reliability-router implementation in this package.
- `estimators.py` still owns estimator families and remains top-level in this
  slice.

Shared dependencies:

- `estimators.py`
- `utils/`
- `artifact_contracts.py`
- `stage_runner.py`

Do not move yet:

- `estimators.py`
- `stage_runner.py`
- router-risk fallbacks in `trainer_legacy_impl.py`
