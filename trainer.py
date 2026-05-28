"""Compatibility facade for the active mainline trainer surface.

Active mainline code should prefer the narrower owner modules:
- artifact_contracts.py
- runtime_env.py
- trainer_preparation.py
- trainer_semantic.py
- trainer_distillation.py
- trainer_graph.py
- trainer_glance.py
- stage_runner.py

During the current compatibility window, StageRunner and some transitional
execution paths still live in ``trainer_legacy_impl.py``. This facade keeps
legacy imports stable while redirecting active owners to their extracted
modules.
"""

from artifact_contracts import MissingFrozenArtifactError, PHASE_A_CONTRACT, PHASE_A_DISABLED_COMPONENTS  # noqa: F401
from runtime_env import _resolve_device  # noqa: F401
from stage_runner import StageRunner  # noqa: F401
from trainer_distillation import run_legacy_graph_seed  # noqa: F401
from trainer_preparation import build_or_load_faithful_gats, build_or_load_frozen_g0, load_frozen_g0  # noqa: F401
from trainer_semantic import run_semantic_finetune_seed  # noqa: F401


def run_phase_a_matrix(args, seed, data, stage_dir, base_bundle):
    raise NotImplementedError(
        "Phase A matrix via StageRunner is not yet implemented. "
        "Use main.py module-controls entry: "
        "python main.py --embedding_path /path/to/emb.pt "
        "--use_GNN --graph_backbone rgcn --seeds 1"
    )
