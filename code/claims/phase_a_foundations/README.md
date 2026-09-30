# Phase A Foundations

Claim role: semantic encoder preparation, semantic embedding classifiers,
frozen graph detector preparation, graph calibration, and legacy distillation
baselines.

Current owner modules:

- `trainer_preparation.py`
- `trainer_semantic.py`
- `trainer_distillation.py`
- `model_building.py`
- `GNNs.py`

Canonical vocabulary:

- `semantic_embeddings`
- `graph_detector_outputs`
- `fused_x`
- `train_idx`
- `valid_idx`
- `test_idx`
- `calibration_logits`

Migration note: keep implementation in the flat `LLMbot/code/` modules until
`trainer_legacy_impl.py` no longer owns fallback behavior for this surface.
