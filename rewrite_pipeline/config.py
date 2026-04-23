"""Experiment configuration for the rewrite pipeline."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass
class RewriteConfig:
    # ── Paths ──
    seed: int = 42
    dataset_path: str = "datasets/TwiBot-20"
    q_final_path: str = "datasets/TwiBot-20/qwen3_emb_last.pt"
    semantic_artifact_dir: str = "LLMbot/saved_artifacts/m5_sem_ibedl_ats"
    semantic_ckpt_dir: str = "LLMbot/checkpoints/dual_router/m5_sem_ibedl_ats"
    results_dir: str = "results/rewrite_pipeline"

    # ── Feature source ──
    feature_source: str = "z_sem"  # "z_sem" | "q_final"

    # ── Graph expert training ──
    graph_epochs: int = 30
    graph_lr: float = 1e-3
    hidden_dim: int = 256
    graph_num_layers: int = 2
    graph_heads: int = 4
    dropout: float = 0.1
    weight_decay: float = 1e-4
    patience: int = 5
    num_classes: int = 2
    calibrate_graph: bool = True

    # ── Rewrite ──
    rewrite_mode: str = "prune_add"  # "none"|"prune_only"|"add_only"|"prune_add"|"random_add"
    trigger_mode: str = "disagreement"  # "disagreement"|"trust_only"|"full"|"random"
    trigger_threshold: float = 0.20
    keep_threshold: float = 0.55
    add_threshold: float = 0.60
    candidate_topk: int = 5
    max_candidates: int = 128
    drop_budget_in: int = 2
    drop_budget_out: int = 2
    add_budget_in: int = 2
    add_budget_out: int = 2
    relation_aware: bool = True
    direction_aware: bool = True

    @property
    def run_tag(self) -> str:
        return f"seed_{self.seed}/feature_{self.feature_source}/graph_{self.rewrite_mode}"

    @property
    def run_dir(self) -> Path:
        return Path(self.results_dir) / self.run_tag

    def save(self, path: Path | None = None):
        p = path or (self.run_dir / "config.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "RewriteConfig":
        return cls(**json.loads(Path(path).read_text()))
