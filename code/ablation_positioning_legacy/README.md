# Ablation Positioning Legacy

Claim role: estimator, semantic, repair, selector, positioning, stress, and
appendix ablation compatibility while the migration body is still shrinking.

Canonical vocabulary:

- `replay_compatibility`
- `legacy_alias`
- `historical_command`

Primary symbols:

- Historical replay and matrix fallback logic remains in `trainer_legacy_impl.py`,
  `estimators.py`, and `stage_registry.py`.
- This claim must not define new public method names.

Shared dependencies:

- `stage_registry.py`
- `estimators.py`
- `trainer_legacy_impl.py`

Do not move yet:

- `trainer_legacy_impl.py`
- legacy matrix fallback code in `estimators.py`
