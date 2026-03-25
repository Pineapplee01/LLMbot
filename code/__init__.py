"""
LEGACY / INACTIVE MODULE

This file used to expose a staged training framework (`base_trainer`,
`losses`, `Trainer`) that no longer matches the active `code/` mainline.

The supported execution path today is:
`precompute.py` -> `main.py` -> `model.py` -> `train.py`

This module is intentionally reduced to an inert summary so it is not mistaken
for an up-to-date package surface.
"""

LEGACY_SUMMARY = r'''
Historical export surface (no longer active):

- from .base_trainer import TrainerConfig, MiniBatch, NeighborSampler, BaseTrainer
- from .losses import EdlLoss, AvUCLoss, UncertaintyMarginLoss,
  SupervisedContrastiveLoss, LossComputer
- from .Trainer import VIBTrainer, GNNTrainer, GateTrainer, FusionTrainer

Historical role:
- package-level entrypoint for an older staged SeGA/llmbot training framework
- long-form educational commentary on the staged pipeline
- public `__all__` surface that does not match the current repository layout

Reason kept:
- to document that this package file is legacy rather than silently deleting it
- to prevent reviewers from treating the old staged framework as the supported
  import surface for the active `code/` directory
'''

__all__ = ["LEGACY_SUMMARY"]
