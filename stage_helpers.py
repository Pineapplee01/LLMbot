"""Shared helpers for active-mainline stage execution.

This module is the new home for cross-stage helper APIs. During the
compatibility window it re-exports the existing implementations from
``trainer.py`` so new code can stop importing the monolithic file
directly while behavior stays unchanged.
"""

from trainer_legacy_impl import (  # noqa: F401
    MissingFrozenArtifactError,
    PHASE_A_CONTRACT,
    PHASE_A_DISABLED_COMPONENTS,
    _resolve_device,
)
