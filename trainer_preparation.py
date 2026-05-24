"""Preparation-stage API surface for active mainline.

Canonical preparation tasks:
- graph_detector_prepare
- graph_calibration_prepare
"""

from trainer_legacy_impl import (  # noqa: F401
    build_or_load_faithful_gats,
    build_or_load_frozen_g0,
    load_frozen_g0,
)
