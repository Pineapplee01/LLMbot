# Phase A Foundations

Claim role: semantic encoder preparation, semantic embedding classifiers,
frozen graph detector preparation, graph calibration, and legacy distillation
baselines.

Canonical vocabulary:

- `semantic_embeddings`
- `graph_detector_outputs`
- `fused_x`
- `train_idx`
- `valid_idx`
- `test_idx`
- `calibration_logits`

Primary symbols:

- `trainer_preparation.py` and `trainer_semantic.py` own most current runtime
  behavior but remain top-level in this slice.
- `trainer_distillation.py`, `model_building.py`, and `GNNs.py` contain shared
  foundation primitives used by multiple claims.

Shared dependencies:

- `utils/`
- `runtime_env.py`
- `artifact_contracts.py`
- `model_building.py`
- `GNNs.py`

Do not move yet:

- `trainer_preparation.py`
- `trainer_semantic.py`
- `trainer_distillation.py`
- `model_building.py`
- `GNNs.py`
