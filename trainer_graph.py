"""Graph provenance and graph-stage compatibility surface.

Graph-aware logic still lives in ``trainer.py`` during the compatibility
window. This module establishes the canonical import location for future
active-mainline work.
"""

from trainer_legacy_impl import MissingFrozenArtifactError  # noqa: F401
