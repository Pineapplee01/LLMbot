# Conformal Risk Routing

Claim role: local conformal risk, conformal KNN risk, router scoring,
selective prediction, and routed-node selection evidence.

Current owner modules:

- `estimators.py`
- `router.py`
- `stage_runner.py`
- `trainer_legacy_impl.py`

Canonical vocabulary:

- `risk_score`
- `risk_scores`
- `router_score`
- `risk_budget`
- `selected_node_ids`
- `coverage_curve`

Migration note: `router.py` should remain reliability/selective-router focused.
Estimator and matrix fallback code in `estimators.py` and
`trainer_legacy_impl.py` should be separated in a later slice.
