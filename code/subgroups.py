import numpy as np
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


class SubgroupContract(dict):
    def __init__(self, *args, aliases=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._aliases = dict(aliases or {})

    def __getitem__(self, key):
        if key in self._aliases:
            return self._aliases[key]()
        if key not in self and "groups" in self and key in self["groups"]:
            return self["groups"][key]
        return super().__getitem__(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _safe_quantile(values, quantile, fallback=0.0):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return float(fallback)
    return float(np.quantile(values, quantile))


def _normalize_unit_interval(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    lower = float(values.min())
    upper = float(values.max())
    if not np.isfinite(lower) or not np.isfinite(upper) or abs(upper - lower) < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lower) / (upper - lower)).astype(np.float32)


def _build_group(mask, **metadata):
    mask = np.asarray(mask, dtype=bool)
    payload = {
        "mask": mask.tolist(),
        "count": int(mask.sum()),
        "ratio": float(mask.mean()) if mask.size else 0.0,
    }
    payload.update(metadata)
    return payload


def _candidate_signal(payload, *names):
    for name in names:
        if isinstance(payload, dict) and name in payload:
            return _to_numpy(payload[name]).astype(np.float32), name
    return None, None


def _candidate_mask(payload, length, *names):
    for name in names:
        if not isinstance(payload, dict) or name not in payload:
            continue
        raw = _to_numpy(payload[name])
        if raw.dtype == np.bool_ and raw.shape[0] == length:
            return raw.astype(bool), name
        idx = raw.astype(np.int64).reshape(-1)
        mask = np.zeros(length, dtype=bool)
        valid_idx = idx[(idx >= 0) & (idx < length)]
        mask[valid_idx] = True
        return mask, name
    return None, None


def structural_features(edge_index, edge_type, num_nodes):
    edge_index = _to_numpy(edge_index)
    edge_type = _to_numpy(edge_type)
    src, dst = edge_index

    in_deg = np.zeros(num_nodes, dtype=np.float32)
    out_deg = np.zeros(num_nodes, dtype=np.float32)
    np.add.at(in_deg, dst, 1)
    np.add.at(out_deg, src, 1)
    total_deg = in_deg + out_deg

    relation_hist = np.zeros((num_nodes, int(edge_type.max()) + 1 if edge_type.size > 0 else 1), dtype=np.float32)
    for relation_id in range(relation_hist.shape[1]):
        mask = edge_type == relation_id
        np.add.at(relation_hist[:, relation_id], dst[mask], 1)

    relation_sum = relation_hist.sum(axis=1, keepdims=True)
    relation_sum = np.clip(relation_sum, a_min=1.0, a_max=None)
    relation_dist = relation_hist / relation_sum
    relation_skew = np.abs(relation_dist - (1.0 / relation_dist.shape[1])).sum(axis=1)

    return {
        "in_degree": in_deg,
        "out_degree": out_deg,
        "total_degree": total_deg,
        "relation_skew": relation_skew.astype(np.float32),
    }


def build_structural_sparse_mask(edge_index, edge_type, num_nodes, degree_quantile=0.25):
    features = structural_features(edge_index, edge_type, num_nodes)
    total_degree = features["total_degree"]
    cutoff = _safe_quantile(total_degree, degree_quantile)
    sparse_mask = total_degree <= cutoff
    manifest = {
        "definition": "structural_sparse_degree",
        "degree_quantile": degree_quantile,
        "degree_cutoff": cutoff,
        "count": int(sparse_mask.sum()),
        "ratio": float(sparse_mask.mean()) if sparse_mask.size else 0.0,
    }
    return sparse_mask.astype(bool), features, manifest


def _neighbor_consensus(edge_index, node_values, num_nodes):
    edge_index = _to_numpy(edge_index)
    node_values = np.asarray(node_values)
    src, dst = edge_index
    counts = np.zeros(num_nodes, dtype=np.int32)
    same = np.zeros(num_nodes, dtype=np.float32)
    mismatch = np.zeros(num_nodes, dtype=np.float32)
    for src_idx, dst_idx in zip(src, dst):
        counts[dst_idx] += 1
        if node_values[src_idx] == node_values[dst_idx]:
            same[dst_idx] += 1.0
        else:
            mismatch[dst_idx] += 1.0

    agreement = np.full(num_nodes, 0.5, dtype=np.float32)
    inconsistency = np.zeros(num_nodes, dtype=np.float32)
    nonzero = counts > 0
    agreement[nonzero] = same[nonzero] / counts[nonzero]
    inconsistency[nonzero] = mismatch[nonzero] / counts[nonzero]
    return agreement, inconsistency, counts


def _build_risk_strata(risk_score, validation_threshold):
    lower_cut = max(validation_threshold * 0.5, 1e-8)
    return {
        "low": _build_group(risk_score < lower_cut, lower=0.0, upper=lower_cut),
        "moderate": _build_group(
            (risk_score >= lower_cut) & (risk_score < validation_threshold),
            lower=lower_cut,
            upper=validation_threshold,
        ),
        "high": _build_group(risk_score >= validation_threshold, lower=validation_threshold, upper=None),
    }


def _resolve_train_mask(num_nodes, train_idx=None, risk_manifest=None, probe_features=None):
    if train_idx is not None:
        raw = _to_numpy(train_idx)
        if raw.dtype == np.bool_ and raw.shape[0] == num_nodes:
            return raw.astype(bool), "train_mask"
        idx = raw.astype(np.int64).reshape(-1)
        mask = np.zeros(num_nodes, dtype=bool)
        valid_idx = idx[(idx >= 0) & (idx < num_nodes)]
        mask[valid_idx] = True
        return mask, "train_idx"

    for payload in (risk_manifest, probe_features):
        mask, name = _candidate_mask(payload, num_nodes, "train_mask", "fit_train_mask", "train_idx", "fit_train_idx")
        if mask is not None:
            return mask, name
    return None, None


def _learn_harmful_propagation(
    proxy_risk,
    neigh_inconsistency,
    relation_skew,
    structural_sparsity,
    labels,
    predictions,
    train_mask,
):
    labels = _to_numpy(labels)
    predictions = _to_numpy(predictions)
    if labels.ndim > 1:
        labels = labels.argmax(axis=1)
    if predictions.ndim > 1:
        predictions = predictions.argmax(axis=1)

    feature_table = np.column_stack(
        [
            _normalize_unit_interval(proxy_risk),
            _normalize_unit_interval(neigh_inconsistency),
            _normalize_unit_interval(relation_skew),
            _normalize_unit_interval(structural_sparsity),
        ]
    ).astype(np.float32)
    y_train = (labels[train_mask] != predictions[train_mask]).astype(np.int32)
    x_train = feature_table[train_mask]
    if x_train.shape[0] < 8 or len(np.unique(y_train)) < 2:
        return None, {
            "available": False,
            "reason": "insufficient_train_diversity",
            "feature_names": [
                "best_P_proxy_risk",
                "neigh_inconsistency",
                "relation_skew",
                "structural_sparsity",
            ],
        }

    learner = LogisticRegression(
        C=0.05,
        penalty="l2",
        solver="lbfgs",
        max_iter=200,
        class_weight="balanced",
    )
    learner.fit(x_train, y_train)
    signal = learner.predict_proba(feature_table)[:, 1].astype(np.float32)
    return signal, {
        "available": True,
        "model": "logistic_regression",
        "training_scope": "train_only",
        "feature_names": [
            "best_P_proxy_risk",
            "neigh_inconsistency",
            "relation_skew",
            "structural_sparsity",
        ],
        "regularization_C": 0.05,
        "train_size": int(train_mask.sum()),
        "train_positive_rate": float(y_train.mean()),
        "coefficients": learner.coef_[0].astype(np.float32).tolist(),
        "intercept": float(learner.intercept_[0]),
    }


def build_operational_subgroups(
    structural_manifest,
    risk_manifest,
    probe_features=None,
    propagation_quantile=0.85,
    edge_index=None,
    predictions=None,
):
    total_degree = np.asarray(structural_manifest["features"]["total_degree"], dtype=np.float32)
    relation_skew = np.asarray(structural_manifest["features"]["relation_skew"], dtype=np.float32)
    risk_score = _to_numpy(risk_manifest["risk_score"]).astype(np.float32)
    num_nodes = total_degree.shape[0]

    agreement_signal, agreement_source = _candidate_signal(probe_features, "neigh_pred_agreement")
    inconsistency_signal, inconsistency_source = _candidate_signal(probe_features, "neigh_inconsistency")
    if agreement_signal is None or inconsistency_signal is None:
        prediction_signal, _ = _candidate_signal(risk_manifest, "predictions", "preds", "pred")
        if prediction_signal is None:
            prediction_signal = _to_numpy(predictions).astype(np.int64) if predictions is not None else np.zeros(num_nodes, dtype=np.int64)
        agreement_signal, inconsistency_signal, _ = _neighbor_consensus(
            edge_index,
            prediction_signal.astype(np.int64),
            num_nodes,
        )
        agreement_source = agreement_source or "derived_neighbor_predictions"
        inconsistency_source = inconsistency_source or "derived_neighbor_predictions"

    low_cutoff = float(structural_manifest["meta"]["degree_cutoff"])
    high_cutoff = _safe_quantile(
        total_degree,
        1.0 - structural_manifest["meta"]["degree_quantile"],
        fallback=low_cutoff,
    )
    skew_cutoff = _safe_quantile(relation_skew, propagation_quantile)
    inconsistency_cutoff = _safe_quantile(inconsistency_signal, propagation_quantile)
    validation_threshold = float(
        risk_manifest.get("thresholds", {}).get("validation_risk_threshold", _safe_quantile(risk_score, 0.85))
    )

    groups = {
        "degree_buckets": {
            "low": _build_group(total_degree <= low_cutoff, lower=None, upper=low_cutoff),
            "mid": _build_group(
                (total_degree > low_cutoff) & (total_degree < high_cutoff),
                lower=low_cutoff,
                upper=high_cutoff,
            ),
            "high": _build_group(total_degree >= high_cutoff, lower=high_cutoff, upper=None),
        },
        "neigh_pred_agreement": _build_group(
            agreement_signal >= 0.5,
            signal_source=agreement_source,
            threshold=0.5,
            direction="higher_is_more_agreement",
        ),
        "neigh_inconsistency": _build_group(
            inconsistency_signal >= inconsistency_cutoff,
            signal_source=inconsistency_source,
            threshold=inconsistency_cutoff,
            direction="higher_is_more_inconsistent",
        ),
        "relation_skew_high": _build_group(
            relation_skew >= skew_cutoff,
            signal_source="atomic_relation_skew",
            threshold=skew_cutoff,
            direction="higher_is_more_skewed",
        ),
        "validation_frozen_risk_strata": _build_risk_strata(risk_score, validation_threshold),
    }

    return SubgroupContract(
        {
            "contract": "operational_selector_safe_v2",
            "selector_safe": True,
            "group_names": [
                "degree_buckets",
                "neigh_pred_agreement",
                "neigh_inconsistency",
                "relation_skew_high",
                "validation_frozen_risk_strata",
            ],
            "groups": groups,
            "selector_safe_views": {
                "sparse_mask": groups["degree_buckets"]["low"]["mask"],
                "prop_mask": groups["neigh_inconsistency"]["mask"],
            },
        },
        aliases={
            "sparse": lambda: groups["degree_buckets"]["low"],
            "propagation_corruption": lambda: groups["neigh_inconsistency"],
        },
    )


def build_analysis_subgroups(
    structural_manifest,
    risk_manifest,
    labels=None,
    predictions=None,
    probe_features=None,
    propagation_quantile=0.85,
    edge_index=None,
    train_idx=None,
):
    num_nodes = len(structural_manifest["sparse_mask"])
    groups = {}

    if labels is not None:
        labels_np = _to_numpy(labels)
        if labels_np.ndim > 1:
            labels_np = labels_np.argmax(axis=1)

        oracle_homophily, oracle_camouflage, neighbor_counts = _neighbor_consensus(
            edge_index,
            labels_np.astype(np.int64),
            num_nodes,
        )
        homophily_cutoff = _safe_quantile(oracle_homophily[neighbor_counts > 0], 0.75, fallback=0.5)
        camouflage_cutoff = _safe_quantile(oracle_camouflage[neighbor_counts > 0], 0.75, fallback=0.5)

        groups["oracle_homophily"] = _build_group(
            (neighbor_counts > 0) & (oracle_homophily >= homophily_cutoff),
            signal_source="oracle_neighbor_label_agreement",
            threshold=homophily_cutoff,
            direction="higher_is_more_homophilous",
        )
        groups["oracle_camouflage_heavy"] = _build_group(
            (neighbor_counts > 0) & (oracle_camouflage >= camouflage_cutoff),
            signal_source="oracle_neighbor_label_inconsistency",
            threshold=camouflage_cutoff,
            direction="higher_is_more_camouflaged",
        )

        if predictions is not None:
            proxy_risk, proxy_source = _candidate_signal(
                risk_manifest,
                "best_P_proxy_risk",
                "best_proxy_risk",
                "proxy_risk",
                "risk_score",
            )
            if proxy_risk is None:
                proxy_risk, proxy_source = _candidate_signal(
                    probe_features,
                    "best_P_proxy_risk",
                    "best_proxy_risk",
                    "proxy_risk",
                )
            if proxy_risk is None:
                proxy_risk = _to_numpy(risk_manifest["risk_score"]).astype(np.float32)
                proxy_source = "risk_score"

            inconsistency_signal, _ = _candidate_signal(probe_features, "neigh_inconsistency")
            if inconsistency_signal is None:
                inconsistency_signal = oracle_camouflage

            relation_skew = np.asarray(structural_manifest["features"]["relation_skew"], dtype=np.float32)
            total_degree = np.asarray(structural_manifest["features"]["total_degree"], dtype=np.float32)
            if np.any(total_degree > 0):
                reference_degree = max(np.percentile(total_degree[total_degree > 0], 50), 1.0)
                structural_sparsity = 1.0 - np.clip(total_degree / reference_degree, 0.0, 1.0)
            else:
                structural_sparsity = np.ones_like(total_degree, dtype=np.float32)

            train_mask, train_source = _resolve_train_mask(
                num_nodes,
                train_idx=train_idx,
                risk_manifest=risk_manifest,
                probe_features=probe_features,
            )
            if train_mask is not None:
                harmful_signal, learner_manifest = _learn_harmful_propagation(
                    proxy_risk=proxy_risk,
                    neigh_inconsistency=inconsistency_signal,
                    relation_skew=relation_skew,
                    structural_sparsity=structural_sparsity,
                    labels=labels_np,
                    predictions=predictions,
                    train_mask=train_mask,
                )
                if harmful_signal is not None:
                    harmful_cutoff = _safe_quantile(harmful_signal, propagation_quantile)
                    groups["harmful_propagation"] = _build_group(
                        harmful_signal >= harmful_cutoff,
                        signal_source="learned_harmful_propagation",
                        threshold=harmful_cutoff,
                        proxy_risk_source=proxy_source,
                        train_source=train_source,
                        learner=learner_manifest,
                    )

    return SubgroupContract(
        {
            "contract": "analysis_only_v2",
            "selector_safe": False,
            "group_names": list(groups.keys()),
            "groups": groups,
        }
    )


def build_subgroup_manifests(
    edge_index,
    edge_type,
    num_nodes,
    risk_manifest,
    labels=None,
    predictions=None,
    probe_features=None,
    degree_quantile=0.25,
    propagation_quantile=0.85,
    train_idx=None,
):
    sparse_mask, features, sparse_meta = build_structural_sparse_mask(
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=num_nodes,
        degree_quantile=degree_quantile,
    )
    structural_manifest = {
        "features": features,
        "sparse_mask": sparse_mask,
        "meta": sparse_meta,
        "schema": {
            "operational_contract": "operational_selector_safe_v2",
            "analysis_contract": "analysis_only_v2",
            "selector_safe_boundary": {
                "operational_selector_safe": True,
                "analysis_selector_safe": False,
            },
        },
    }
    operational = build_operational_subgroups(
        structural_manifest=structural_manifest,
        risk_manifest=risk_manifest,
        probe_features=probe_features,
        propagation_quantile=propagation_quantile,
        edge_index=edge_index,
        predictions=predictions,
    )
    analysis = build_analysis_subgroups(
        structural_manifest=structural_manifest,
        risk_manifest=risk_manifest,
        labels=labels,
        predictions=predictions,
        probe_features=probe_features,
        propagation_quantile=propagation_quantile,
        edge_index=edge_index,
        train_idx=train_idx,
    )
    return structural_manifest, operational, analysis
