"""Stage-6 of the v8 EQC pipeline: LLM embedding refiner with linear probe.

This module implements the sixth (and final) stage of the v8 EQC pipeline:
conversion of the Stage-5 evidence-graph artifact into a fixed-size
representation via a frozen external encoder, and combination with the
post-rewrite GNN state and frozen RoBERTa text embedding through a
linear probe trained on the ``train_cal`` split. The Block-E suite
instantiates seven encoder variants that share Stages 1-5 verbatim:

* ``e_null``        : no Stage-6; linear probe on ``(h_gnn_post, x_roberta)``.
* ``e_roberta``     : same-serialization frozen RoBERTa with structured,
                      field-aware mean-pooling over the evidence text.
* ``e_llm_collapse``: Qwen embedding of the ``collapse`` schema (flat
                      neighbour list, no role labels).
* ``e_llm_qwen``    : Qwen embedding of the canonical v1.0 schema.
* ``e_llm_mistral`` : Mistral embedding of the canonical v1.0 schema.
* ``e_llm_shuffle_role`` : Qwen embedding of the ``shuffle_role`` schema.
* ``e_llm_shuffle_order``: Qwen embedding of the ``shuffle_order`` schema.

The class is deliberately separate from :mod:`operators` because it
integrates an external asset (the LLM embedding model) and because the
single-file convention of the core directory reserves ``operators.py``
for in-pipeline, zero-external-dependency operators. The linear probe is
the *only* trainable inference component in the entire v8 pipeline
(鈮?9.2k parameters), by construction.

References
----------
* LLM-as-enhancer, not predictor: GLANCE (ICLR 2026) and the IJCAI 2024
  Graph Meets LLM Survey. Motivates the selective, embedding-only use of
  external encoders at Stage-6.
* Linear-probe sufficiency: SimTeG (ICLR 2024) established that frozen
  text encoders plus a linear head are an extremely strong baseline on
  text-attributed graphs; we use this result to justify the v1 default
  over a 2-layer MLP (which lives in the Block-L ablation only).
* Evidence-prompt serialization: GraphText (2023) and the TMLR 2024
  analysis of graph-prompt behaviour. We only serialize the Stage-5
  canonical artifact; we do not attempt to make the LLM reason about
  graph structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


_ENCODER_DIM_DEFAULTS = {
    "e_null": 0,
    "e_roberta": 768,
    "e_llm_collapse": 3584,
    "e_llm_qwen": 3584,
    "e_llm_mistral": 4096,
    "e_llm_shuffle_role": 3584,
    "e_llm_shuffle_order": 3584,
}


@dataclass
class EvidenceEncoderSpec:
    """Configuration for a Block-E encoder variant.

    Attributes
    ----------
    name : str
        Variant identifier (see module docstring for the seven canonical
        names and their role in the Block-E falsification grid).
    schema_mode : str
        Stage-5 scramble mode consumed: ``canonical``, ``shuffle_role``,
        ``shuffle_order``, ``collapse``, or ``minimal``.
    backbone : str
        External encoder family: ``none`` (E-null), ``roberta``
        (E-RoBERTa same-serialization control), ``qwen``, or ``mistral``.
    embedding_dim : int
        Expected embedding dimensionality produced by ``encode_fn``.
    token_budget : int
        Token budget enforced at serialization time (deterministic
        drop-tail truncation; see
        :meth:`EvidenceGraphRewriter.build_evidence_graph`).
    """

    name: str
    schema_mode: str
    backbone: str
    embedding_dim: int
    token_budget: int = 512
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMEvidenceRefiner:
    """Stage-6 embedding + linear probe for the v8 EQC pipeline.

    The refiner is deliberately minimal: a single linear layer operates
    on the concatenation of the post-rewrite GNN hidden state, the frozen
    RoBERTa text embedding, and the Stage-5 evidence-graph embedding
    produced by an external encoder. The probe is trained on the
    ``train_cal`` split only; the external encoder is always frozen.

    The refiner does not perform any text generation, prompt optimization,
    or model fine-tuning. It is plumbing for the Stage-5 evidence-graph
    contribution: the Block-E suite uses :meth:`fit_linear_probe` plus
    :meth:`predict_proba` to compare seven encoder variants sharing the
    identical Stage-1..5 pipeline.
    """

    metadata = {
        "claim_role": "eqc_v8_stage6_plumbing_readout",
        "scientific_gate": "linear_probe_on_train_cal_labels_only",
        "paper_identity_risk": "low_plumbing_not_novelty",
        "promotion_rule": "pre_committed_plumbing_never_primary_claim",
    }

    def __init__(self, spec: EvidenceEncoderSpec,
                 h_gnn_dim: int = 256, x_roberta_dim: int = 768,
                 l2_regularization: float = 1.0, seed: int = 0):
        self.spec = spec
        self.h_gnn_dim = int(h_gnn_dim)
        self.x_roberta_dim = int(x_roberta_dim)
        self.l2_regularization = float(l2_regularization)
        self.seed = int(seed)
        self._encode_fn: Optional[Callable[[Sequence[str]], np.ndarray]] = None
        self._classifier = None
        self._fit_summary: Dict[str, Any] = {}

    @property
    def is_fitted(self) -> bool:
        return self._classifier is not None

    def register_encoder(self, encode_fn: Callable[[Sequence[str]], np.ndarray]) -> "LLMEvidenceRefiner":
        """Attach the external-encoder callable.

        ``encode_fn`` must accept a list of serialized evidence-graph
        strings (length ``N``) and return a ``numpy.ndarray`` of shape
        ``(N, spec.embedding_dim)``. The callable is treated as frozen
        and stateless; ``LLMEvidenceRefiner`` performs no LLM training
        and will raise a ``ValueError`` if the returned shape disagrees
        with :attr:`spec.embedding_dim`.

        For ``spec.backbone == "none"`` (E-null) the caller must pass
        ``encode_fn = None`` or skip this step; Stage-6 will concatenate
        only ``h_gnn_post`` and ``x_roberta`` at fit and inference time.
        """
        if self.spec.backbone == "none":
            self._encode_fn = None
            return self
        if encode_fn is None:
            raise ValueError(
                f"Encoder backbone '{self.spec.backbone}' requires a non-null encode_fn."
            )
        self._encode_fn = encode_fn
        return self

    def _encode_prompts(self, prompts: Sequence[str]) -> np.ndarray:
        if self.spec.backbone == "none":
            return np.zeros((len(prompts), 0), dtype=np.float64)
        if self._encode_fn is None:
            raise RuntimeError(
                f"LLMEvidenceRefiner[{self.spec.name}] requires register_encoder() before encoding prompts."
            )
        embeddings = np.asarray(self._encode_fn(list(prompts)), dtype=np.float64)
        if embeddings.ndim != 2:
            raise ValueError(
                f"encode_fn must return a 2D array; got shape {embeddings.shape}."
            )
        if embeddings.shape[0] != len(prompts):
            raise ValueError(
                f"encode_fn returned {embeddings.shape[0]} rows for {len(prompts)} prompts."
            )
        if embeddings.shape[1] != int(self.spec.embedding_dim):
            raise ValueError(
                f"encode_fn returned embedding dim {embeddings.shape[1]}; "
                f"spec expects {self.spec.embedding_dim}."
            )
        return embeddings

    @staticmethod
    def _assemble_features(h_gnn_post: np.ndarray, x_roberta: np.ndarray,
                           z_evidence: np.ndarray) -> np.ndarray:
        """Concatenate ``[h_gnn_post; x_roberta; z_evidence]`` row-wise
        while tolerating empty ``z_evidence`` for the E-null variant.
        """
        h = np.asarray(h_gnn_post, dtype=np.float64)
        x = np.asarray(x_roberta, dtype=np.float64)
        z = np.asarray(z_evidence, dtype=np.float64)
        if h.shape[0] != x.shape[0]:
            raise ValueError(f"h_gnn_post rows ({h.shape[0]}) != x_roberta rows ({x.shape[0]}).")
        # Distinguish E-null (embedding_dim=0, shape (N,0)) from zero-row arrays.
        no_evidence = z.ndim < 2 or z.shape[1] == 0
        if not no_evidence and z.shape[0] != h.shape[0]:
            raise ValueError(f"z_evidence rows ({z.shape[0]}) != h_gnn_post rows ({h.shape[0]}).")
        if no_evidence:
            return np.concatenate([h, x], axis=1)
        return np.concatenate([h, x, z], axis=1)

    def fit_linear_probe(self, h_gnn_post: np.ndarray, x_roberta: np.ndarray,
                         evidence_prompts: Sequence[str], labels: np.ndarray,
                         split_tag: str = "train_cal") -> "LLMEvidenceRefiner":
        """Fit the single-linear-layer probe on the ``train_cal`` split.

        Parameters
        ----------
        h_gnn_post : numpy.ndarray
            Post-rewrite GNN hidden state, shape ``(N, h_gnn_dim)``.
        x_roberta : numpy.ndarray
            Frozen RoBERTa pooled embedding, shape ``(N, x_roberta_dim)``.
        evidence_prompts : sequence of str
            Serialized v1.0 evidence-graph artifacts (one per hard node).
        labels : numpy.ndarray
            Binary labels for the ``train_cal`` hard-node subset.
        split_tag : str
            Caller-supplied split identifier recorded in fit_summary for
            audit purposes. Must be ``"train_cal"`` for claim-grade runs.

        Returns
        -------
        LLMEvidenceRefiner
            ``self``, for fluent chaining.
        """
        if split_tag != "train_cal":
            import warnings
            warnings.warn(
                f"LLMEvidenceRefiner.fit_linear_probe called with split_tag={split_tag!r}; "
                "claim-grade runs must use split_tag='train_cal'.",
                stacklevel=2,
            )
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "LLMEvidenceRefiner.fit_linear_probe requires scikit-learn."
            ) from exc

        z_evidence = self._encode_prompts(evidence_prompts)
        features = self._assemble_features(h_gnn_post, x_roberta, z_evidence)
        targets = np.asarray(labels, dtype=np.int64).reshape(-1)
        if features.shape[0] != targets.shape[0]:
            raise ValueError("Feature rows and label rows disagree at fit_linear_probe.")

        classifier = LogisticRegression(
            penalty="l2",
            C=1.0 / max(self.l2_regularization, 1e-8),
            solver="lbfgs",
            max_iter=1000,
            random_state=self.seed,
        )
        classifier.fit(features, targets)
        self._classifier = classifier
        self._fit_summary = {
            "source": "llm_evidence_refiner_linear_probe",
            "spec_name": str(self.spec.name),
            "schema_mode": str(self.spec.schema_mode),
            "backbone": str(self.spec.backbone),
            "input_dim_total": int(features.shape[1]),
            "h_gnn_dim": int(self.h_gnn_dim),
            "x_roberta_dim": int(self.x_roberta_dim),
            "z_evidence_dim": int(z_evidence.shape[1] if not (z_evidence.ndim < 2 or z_evidence.shape[1] == 0) else 0),
            "n_fit_nodes": int(features.shape[0]),
            "l2_regularization": float(self.l2_regularization),
            "fit_scope": str(split_tag),
            "framing": "single_linear_layer_readout_sim_teg_inspired",
        }
        return self

    def predict_proba(self, h_gnn_post: np.ndarray, x_roberta: np.ndarray,
                      evidence_prompts: Sequence[str]) -> np.ndarray:
        """Return the two-class posterior probabilities ``p_final(v)``.

        Must be called after :meth:`fit_linear_probe`. The ``E-null``
        variant discards ``evidence_prompts`` and operates on
        ``[h_gnn_post; x_roberta]`` alone; other variants encode the
        prompts through the registered external encoder.
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"LLMEvidenceRefiner[{self.spec.name}].predict_proba called before fit_linear_probe."
            )
        z_evidence = self._encode_prompts(evidence_prompts)
        features = self._assemble_features(h_gnn_post, x_roberta, z_evidence)
        probs = self._classifier.predict_proba(features)
        return np.asarray(probs, dtype=np.float64)

    def state_dict_payload(self) -> Dict[str, Any]:
        return {
            "spec_name": str(self.spec.name),
            "schema_mode": str(self.spec.schema_mode),
            "backbone": str(self.spec.backbone),
            "embedding_dim": int(self.spec.embedding_dim),
            "token_budget": int(self.spec.token_budget),
            "h_gnn_dim": int(self.h_gnn_dim),
            "x_roberta_dim": int(self.x_roberta_dim),
            "l2_regularization": float(self.l2_regularization),
            "is_fitted": bool(self.is_fitted),
            "fit_summary": dict(self._fit_summary),
        }


def build_block_e_encoder_specs(include_shuffle_controls: bool = True,
                                include_mistral: bool = True,
                                token_budget: int = 512) -> List[EvidenceEncoderSpec]:
    """Return the canonical Block-E encoder variant list.

    The Block-E falsification grid requires seven variants that share
    the identical Stages 1-5 output and differ only in the Stage-6
    encoder and (for the scramble controls) the serialization mode.
    Disabling ``include_shuffle_controls`` yields only the three
    load-bearingness variants (E-null, E-RoBERTa, E-LLM-Qwen);
    disabling ``include_mistral`` drops the LLM-family robustness
    control. The default configuration matches the v8 Round-5
    FINAL_PROPOSAL acceptance criterion.
    """
    specs: List[EvidenceEncoderSpec] = [
        EvidenceEncoderSpec(
            name="e_null",
            schema_mode="canonical",
            backbone="none",
            embedding_dim=_ENCODER_DIM_DEFAULTS["e_null"],
            token_budget=token_budget,
            metadata={"role": "load_bearingness_baseline"},
        ),
        EvidenceEncoderSpec(
            name="e_roberta",
            schema_mode="canonical",
            backbone="roberta",
            embedding_dim=_ENCODER_DIM_DEFAULTS["e_roberta"],
            token_budget=token_budget,
            metadata={"role": "same_serialization_frozen_encoder_control"},
        ),
        EvidenceEncoderSpec(
            name="e_llm_qwen",
            schema_mode="canonical",
            backbone="qwen",
            embedding_dim=_ENCODER_DIM_DEFAULTS["e_llm_qwen"],
            token_budget=token_budget,
            metadata={"role": "canonical_llm_embedding"},
        ),
    ]
    if include_mistral:
        specs.append(
            EvidenceEncoderSpec(
                name="e_llm_mistral",
                schema_mode="canonical",
                backbone="mistral",
                embedding_dim=_ENCODER_DIM_DEFAULTS["e_llm_mistral"],
                token_budget=token_budget,
                metadata={"role": "llm_family_robustness_control"},
            )
        )
    if include_shuffle_controls:
        specs.extend([
            EvidenceEncoderSpec(
                name="e_llm_collapse",
                schema_mode="collapse",
                backbone="qwen",
                embedding_dim=_ENCODER_DIM_DEFAULTS["e_llm_collapse"],
                token_budget=token_budget,
                metadata={"role": "minimal_artifact_baseline"},
            ),
            EvidenceEncoderSpec(
                name="e_llm_shuffle_role",
                schema_mode="shuffle_role",
                backbone="qwen",
                embedding_dim=_ENCODER_DIM_DEFAULTS["e_llm_shuffle_role"],
                token_budget=token_budget,
                metadata={"role": "role_label_scramble_control"},
            ),
            EvidenceEncoderSpec(
                name="e_llm_shuffle_order",
                schema_mode="shuffle_order",
                backbone="qwen",
                embedding_dim=_ENCODER_DIM_DEFAULTS["e_llm_shuffle_order"],
                token_budget=token_budget,
                metadata={"role": "intra_bucket_order_scramble_control"},
            ),
        ])
    return specs


def assemble_block_e_acceptance_report(results: Dict[str, Dict[str, float]],
                                       seed_std_per_variant: Optional[Dict[str, float]] = None,
                                       ) -> Dict[str, Any]:
    """Evaluate the v8 Round-5 four-part Claim-C acceptance criterion.

    ``results`` must be a nested mapping of the form
    ``results[dataset][variant] = slice_f1_mean``. When provided,
    ``seed_std_per_variant`` supplies per-variant seed standard
    deviations used for the magnitude threshold. The returned report
    contains booleans for each of the four sub-parts plus the
    aggregated verdict; downstream drivers decide how to contract the
    dominant contribution when a part fails, per the pre-committed
    fallback table in ``FINAL_PROPOSAL.md``.
    """
    import warnings

    datasets = sorted(results.keys())
    if len(datasets) < 2:
        raise ValueError("Block-E acceptance requires at least two datasets.")
    seed_std = seed_std_per_variant or {}

    # Warn when seed_std is absent 鈥?magnitude gate becomes trivially vacuous.
    if not seed_std or all(float(seed_std.get(v, 0.0)) == 0.0 for v in seed_std):
        warnings.warn(
            "assemble_block_e_acceptance_report: seed_std_per_variant is absent or all-zero. "
            "Part-II magnitude gate will pass on any positive gap (vacuous criterion).",
            stacklevel=2,
        )

    variants_any_dataset = set(next(iter(results.values())).keys())

    # Require all three scramble controls for a valid Part-III check.
    _REQUIRED_SCRAMBLE = ("e_llm_shuffle_role", "e_llm_shuffle_order", "e_llm_collapse")
    missing_scramble = [v for v in _REQUIRED_SCRAMBLE if v not in variants_any_dataset]
    if missing_scramble:
        warnings.warn(
            f"assemble_block_e_acceptance_report: scramble variants {missing_scramble} are absent. "
            "Part-III will be False (contracted). All three scramble controls are required for READY.",
            stacklevel=2,
        )

    # Require Mistral for a valid Part-IV check.
    mistral_present = "e_llm_mistral" in variants_any_dataset
    if not mistral_present:
        warnings.warn(
            "assemble_block_e_acceptance_report: 'e_llm_mistral' is absent. "
            "Part-IV LLM-family robustness cannot be evaluated; verdict will be 'contracted'.",
            stacklevel=2,
        )

    def _delta(variant_a: str, variant_b: str, dataset: str) -> float:
        return float(results[dataset][variant_a]) - float(results[dataset][variant_b])

    def _direction_consistent(variant_a: str, variant_b: str) -> bool:
        signs = {np.sign(_delta(variant_a, variant_b, ds)) for ds in datasets}
        return signs == {1.0}

    def _magnitude_one_seed_std(variant_a: str, variant_b: str) -> bool:
        for ds in datasets:
            std_a = float(seed_std.get(variant_a, 0.0))
            std_b = float(seed_std.get(variant_b, 0.0))
            combined_std = float(np.sqrt(std_a ** 2 + std_b ** 2))
            if _delta(variant_a, variant_b, ds) >= combined_std - 1e-12:
                return True
        return False

    part_i = _direction_consistent("e_llm_qwen", "e_roberta") and _direction_consistent("e_roberta", "e_null")
    part_ii = any(
        _magnitude_one_seed_std(a, b)
        for a, b in [("e_llm_qwen", "e_roberta"), ("e_roberta", "e_null")]
    )
    scramble_variants = [v for v in _REQUIRED_SCRAMBLE if v in variants_any_dataset]
    part_iii = (
        len(missing_scramble) == 0
        and all(_direction_consistent("e_llm_qwen", v) for v in scramble_variants)
    )
    # Part-IV: Mistral must be present; absent Mistral 鈫?contracted.
    if mistral_present:
        part_iv = True
        for ds in datasets:
            std_qwen = float(seed_std.get("e_llm_qwen", 0.0))
            std_mistral = float(seed_std.get("e_llm_mistral", 0.0))
            combined_std = float(np.sqrt(std_qwen ** 2 + std_mistral ** 2))
            if abs(_delta("e_llm_qwen", "e_llm_mistral", ds)) > max(combined_std, 1e-12):
                part_iv = False
                break
    else:
        part_iv = False

    verdict = "ready" if (part_i and part_ii and part_iii and part_iv) else "contracted"
    return {
        "contract": "block_e_claim_c_acceptance_v1",
        "datasets": list(datasets),
        "variants_required": [
            "e_null", "e_roberta", "e_llm_qwen", "e_llm_mistral",
            "e_llm_shuffle_role", "e_llm_shuffle_order", "e_llm_collapse",
        ],
        "part_i_direction_null_roberta_qwen": bool(part_i),
        "part_ii_magnitude_at_least_one_seed_std": bool(part_ii),
        "part_iii_scramble_degradations": bool(part_iii),
        "part_iv_llm_family_robustness": bool(part_iv),
        "verdict": verdict,
    }
