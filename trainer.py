"""Compatibility facade for the active mainline trainer surface.

Active mainline code should prefer the narrower modules:
- stage_helpers.py
- trainer_preparation.py
- trainer_semantic.py
- trainer_graph.py
- trainer_glance.py
- stage_runner.py

During the current compatibility window, the legacy implementation still
resides in ``trainer_legacy_impl.py`` and is re-exported here so existing
imports continue to work.
"""

from trainer_legacy_impl import (  # noqa: F401
    MissingFrozenArtifactError,
    PHASE_A_CONTRACT,
    PHASE_A_DISABLED_COMPONENTS,
    StageRunner,
    _resolve_device,
    build_or_load_faithful_gats,
    build_or_load_frozen_g0,
    load_frozen_g0,
    run_legacy_graph_seed,
    run_phase_a_matrix,
    run_semantic_finetune_seed,
)
