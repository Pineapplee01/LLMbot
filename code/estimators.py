import hashlib
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


RESIDUAL_RISK_PAPER_BUDGETS = (0.10, 0.20, 0.30, 0.40)

RISK_CANDIDATE_MSP = "msp"
RISK_CANDIDATE_ENTROPY = "entropy"
RISK_CANDIDATE_MARGIN = "margin"
RISK_CANDIDATE_MSP_TS = "msp_ts"
GLANCE_ROUTER_LOGISTIC = "glance_router_logistic"
GLANCE_SOFT_HOMOPHILY_PRIOR = "glance_soft_homophily_prior"
GLANCE_DEGREE_HOMOPHILY_PRIOR = "glance_degree_homophily_prior"
GLANCE_UNCERTAINTY_HOMOPHILY_BLEND = "glance_uncertainty_homophily_blend"

GLANCE_ROUTING_STRUCTURAL_FEATURE_NAMES = (
    "glance_log_total_degree",
    "glance_sparse_degree_score",
    "glance_relative_degree",
    "glance_low_relative_degree_score",
    "glance_soft_estimated_homophily",
    "glance_estimated_heterophily",
    "glance_local_prediction_support",
    "glance_local_prediction_inconsistency",
    "glance_neighbor_context_gap",
    "glance_relation_skew",
    "glance_uncertainty_score",
)

GLANCE_FORMULA_REFERENCE = {
    "relative_degree": "bar_d_v = mean_{u in N(v)} sqrt((d_v + 1) / (d_u + 1))",
    "soft_estimated_homophily": "hat_h_v = p_Q,v dot mean_{u in N_1(v)} p_Q,u",
    "router_probability": "a_v = sigmoid(w^T f_v), with fixed-budget top-k/top-B selection",
    "stage2_adaptation": "learn P(Stage1 prediction is wrong | f_v); no LLM query outcome or repair action is used",
}

STAGE2_MINIMUM_THRESHOLDS = {
    "ErrRecall@20": 0.35,
    "ErrRecall@30": 0.50,
    "ErrRecall@40": 0.65,
    "Lift@20": 1.75,
    "Lift@30": 1.60,
    "Lift@40": 1.50,
    "auroc_error": 0.70,
    "auprc_error_ratio": 1.80,
    "aurc_relative_reduction": 0.05,
}

STAGE2_STRONG_THRESHOLDS = {
    "ErrRecall@20": 0.45,
    "ErrRecall@30": 0.60,
    "ErrRecall@40": 0.75,
    "Lift@20": 2.20,
    "Lift@30": 2.00,
    "Lift@40": 1.80,
    "auroc_error": 0.78,
    "auprc_error_ratio": 2.50,
    "aurc_relative_reduction": 0.10,
}

def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_tensor(value, device=None):
    if torch.is_tensor(value):
        return value.to(device) if device is not None else value
    return torch.tensor(value, device=device)


def _labels_to_numpy(labels):
    labels_np = _to_numpy(labels)
    if labels_np.ndim > 1:
        labels_np = labels_np.argmax(axis=1)
    return labels_np.reshape(-1)


def _valid_index_array(indices, upper_bound):
    index_np = _to_numpy(indices).astype(np.int64).reshape(-1)
    return index_np[(index_np >= 0) & (index_np < int(upper_bound))]


def _stage1_residual_errors(labels, probs):
    labels_np = _labels_to_numpy(labels)
    probs_np = _normalize_probs(probs).numpy()
    if labels_np.shape[0] != probs_np.shape[0]:
        raise ValueError("Residual-risk labels and Stage 1 probabilities must have the same number of nodes.")
    predictions = probs_np.argmax(axis=1)
    residual_errors = (predictions != labels_np).astype(np.int32)
    return labels_np, predictions, residual_errors, probs_np


@dataclass(frozen=True)
class GlanceTrainingTarget:
    objective: str
    labels: np.ndarray
    fit_idx: np.ndarray
    target_semantics: str
    metadata: dict
    advantage: Optional[np.ndarray] = None


def build_residual_error_training_target(labels, probs, fit_idx):
    _, _, residual_errors, _ = _stage1_residual_errors(labels, probs)
    if fit_idx is None:
        raise ValueError("Residual-risk GLANCE training requires OOF fit_idx.")
    fit_idx_np = _valid_index_array(fit_idx, residual_errors.shape[0])
    fit_labels = residual_errors[fit_idx_np].astype(np.int32)
    return GlanceTrainingTarget(
        objective="residual_error",
        labels=fit_labels,
        fit_idx=fit_idx_np,
        target_semantics="1[Stage1 prediction != y]",
        metadata={
            "target_source": "oof_stage1_residual_error",
            "stage3_action_or_repair_outcome_used": False,
            "llm_counterfactual_outcome_used": False,
            "fit_count": int(fit_idx_np.size),
            "positive_count": int(fit_labels.sum()) if fit_idx_np.size else 0,
        },
        advantage=None,
    )


def _node_aligned_vector(value, name):
    if value is None:
        return None
    vector = _to_numpy(value).reshape(-1)
    if vector.size == 0:
        raise ValueError(f"{name} must contain one value per node.")
    return vector


def _paired_node_vectors(left, right, left_name, right_name):
    left_vec = _node_aligned_vector(left, left_name)
    right_vec = _node_aligned_vector(right, right_name)
    if left_vec is None or right_vec is None:
        return None, None
    if left_vec.shape[0] != right_vec.shape[0]:
        raise ValueError(f"{left_name} and {right_name} must have the same number of nodes.")
    return left_vec, right_vec


def build_glance_advantage_training_target(
    *,
    fit_idx,
    llm_query_cost=0.2,
    gnn_loss=None,
    llm_loss=None,
    gnn_correct=None,
    llm_correct=None,
):
    """Build Loveland et al. GLANCE-style route-benefit labels.

    This is an explicit ablation target, not the default Stage 2 residual-risk
    target. It requires node-aligned GNN-only and LLM/refiner counterfactual
    supervision and marks that LLM counterfactual outcomes were used.
    """
    gnn_loss_vec, llm_loss_vec = _paired_node_vectors(gnn_loss, llm_loss, "gnn_loss", "llm_loss")
    if gnn_loss_vec is not None:
        node_count = int(gnn_loss_vec.shape[0])
        fit_idx_np = _valid_index_array(fit_idx, node_count)
        beta = float(llm_query_cost)
        advantage = gnn_loss_vec.astype(np.float64) - llm_loss_vec.astype(np.float64) - beta
        target_source = "loss_advantage"
        target_semantics = "1[loss_gnn - loss_llm - beta > 0]"
    else:
        gnn_correct_vec, llm_correct_vec = _paired_node_vectors(
            gnn_correct,
            llm_correct,
            "gnn_correct",
            "llm_correct",
        )
        if gnn_correct_vec is None:
            raise ValueError(
                "GLANCE advantage training requires explicit GNN-vs-LLM counterfactual supervision "
                "via paired gnn_loss/llm_loss or gnn_correct/llm_correct arrays."
            )
        node_count = int(gnn_correct_vec.shape[0])
        fit_idx_np = _valid_index_array(fit_idx, node_count)
        beta = float(llm_query_cost)
        gnn_utility = gnn_correct_vec.astype(np.float64)
        llm_utility = llm_correct_vec.astype(np.float64)
        advantage = llm_utility - gnn_utility - beta
        target_source = "correctness_advantage"
        target_semantics = "1[utility_llm - utility_gnn - beta > 0]"

    fit_advantage = advantage[fit_idx_np].astype(np.float32)
    fit_labels = (fit_advantage > 0.0).astype(np.int32)
    return GlanceTrainingTarget(
        objective="glance_advantage",
        labels=fit_labels,
        fit_idx=fit_idx_np,
        target_semantics=target_semantics,
        metadata={
            "target_source": target_source,
            "target_semantics": target_semantics,
            "llm_query_cost": beta,
            "stage3_action_or_repair_outcome_used": True,
            "llm_counterfactual_outcome_used": True,
            "counterfactual_outcome_used": True,
            "paper_faithful_glance_inspired": True,
            "official_code_verified": False,
            "not_a_new_bot_classifier": True,
            "fit_count": int(fit_idx_np.size),
            "positive_count": int(fit_labels.sum()) if fit_idx_np.size else 0,
        },
        advantage=fit_advantage,
    )


def fit_temperature_scaling(logits_val, y_val):
    logits_val = _to_tensor(logits_val)
    y_val = _to_tensor(y_val).long()
    best_nll = float("inf")
    best_temperature = 1.0
    for temperature in np.arange(0.5, 5.1, 0.1):
        probs = F.softmax(logits_val / temperature, dim=1)
        nll = -torch.log(probs[torch.arange(len(y_val)), y_val] + 1e-8).mean().item()
        if nll < best_nll:
            best_nll = nll
            best_temperature = float(temperature)
    return best_temperature


def _risk_inputs(y_wrong, risk_scores, context="Residual-risk metrics"):
    y_wrong = _to_numpy(y_wrong).astype(np.int32).reshape(-1)
    risk_scores = _to_numpy(risk_scores).astype(np.float64).reshape(-1)
    if y_wrong.shape[0] != risk_scores.shape[0]:
        raise ValueError(f"{context} require y_wrong and risk_scores to have the same length.")
    y_wrong = np.clip(y_wrong, 0, 1)
    risk_scores = np.nan_to_num(risk_scores, nan=0.0, posinf=1.0, neginf=0.0)
    return y_wrong, risk_scores


def parse_budget_list(value, default=RESIDUAL_RISK_PAPER_BUDGETS):
    if value is None:
        budgets = tuple(default)
    elif isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
        budgets = tuple(float(piece) for piece in pieces) if pieces else tuple(default)
    else:
        budgets = tuple(float(item) for item in value)

    if not budgets:
        raise ValueError("Residual-risk budgets must include at least one value.")
    for budget in budgets:
        if not math.isfinite(budget) or budget <= 0.0 or budget > 1.0:
            raise ValueError(f"Residual-risk budget must be in (0, 1], got {budget}.")
    return budgets


def residual_risk_budget_curve(y_wrong, risk_scores, budgets=RESIDUAL_RISK_PAPER_BUDGETS):
    y_wrong, risk_scores = _risk_inputs(y_wrong, risk_scores, context="Residual-risk budget metrics")
    budgets = parse_budget_list(budgets)
    if y_wrong.size == 0:
        return []
    order = np.argsort(-risk_scores)
    total_errors = int(y_wrong.sum())
    base_error_rate = float(y_wrong.mean())
    rows = []
    for budget in budgets:
        k = min(max(int(len(y_wrong) * float(budget)), 1), len(y_wrong))
        selected = order[:k]
        hits = int(y_wrong[selected].sum())
        remaining = order[k:]
        precision = float(hits / max(k, 1))
        remaining_risk = float(y_wrong[remaining].mean()) if remaining.size else 0.0
        rows.append(
            {
                "budget": float(budget),
                "selected_count": int(k),
                "error_recall": float(hits / max(total_errors, 1)),
                "precision": precision,
                "lift": float(precision / base_error_rate) if base_error_rate > 0 else 0.0,
                "remaining_risk": remaining_risk,
            }
        )
    return rows


def router_budget_curve(y_wrong, risk_scores, budgets=RESIDUAL_RISK_PAPER_BUDGETS):
    """Legacy alias for residual_risk_budget_curve."""
    return residual_risk_budget_curve(y_wrong, risk_scores, budgets=budgets)


def random_expected_residual_risk_budget_curve(y_wrong, budgets=RESIDUAL_RISK_PAPER_BUDGETS):
    y_wrong = _to_numpy(y_wrong).astype(np.int32).reshape(-1)
    budgets = parse_budget_list(budgets)
    if y_wrong.size == 0:
        return []
    total_errors = int(y_wrong.sum())
    base_error_rate = float(y_wrong.mean())
    rows = []
    for budget in budgets:
        k = min(max(int(len(y_wrong) * float(budget)), 1), len(y_wrong))
        precision = base_error_rate
        rows.append(
            {
                "budget": float(budget),
                "selected_count": int(k),
                "error_recall": float((k / len(y_wrong)) if total_errors else 0.0),
                "precision": precision,
                "lift": 1.0 if base_error_rate > 0 else 0.0,
                "remaining_risk": base_error_rate,
            }
        )
    return rows


def _risk_coverage_auc(y_wrong, risk_scores):
    y_wrong, risk_scores = _risk_inputs(y_wrong, risk_scores, context="Risk-coverage metrics")
    y_wrong = y_wrong.astype(np.float64)
    if y_wrong.size == 0:
        return 0.0
    order = np.argsort(risk_scores)
    cumulative_errors = np.cumsum(y_wrong[order])
    counts = np.arange(1, len(order) + 1, dtype=np.float64)
    coverages = counts / float(len(order))
    risks = cumulative_errors / counts
    coverages = np.concatenate([[0.0], coverages])
    risks = np.concatenate([[0.0], risks])
    return float(np.trapz(risks, coverages))


def oracle_risk_coverage_auc(y_wrong):
    y_wrong = _to_numpy(y_wrong).astype(np.int32).reshape(-1)
    if y_wrong.size == 0:
        return 0.0
    oracle_scores = y_wrong.astype(np.float64)
    return _risk_coverage_auc(y_wrong, oracle_scores)


def _reliability_bins(y_wrong, risk_scores, n_bins=10, adaptive=False):
    y_wrong, risk_scores = _risk_inputs(y_wrong, risk_scores, context="Reliability diagram")
    risk_prob = np.clip(risk_scores, 1e-8, 1.0 - 1e-8)
    if y_wrong.size == 0:
        return [], 0.0

    rows = []
    if adaptive:
        order = np.argsort(risk_prob)
        bin_indices = [chunk for chunk in np.array_split(order, int(n_bins)) if chunk.size]
    else:
        edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
        bin_indices = []
        for bin_idx in range(int(n_bins)):
            if bin_idx == int(n_bins) - 1:
                mask = (risk_prob >= edges[bin_idx]) & (risk_prob <= edges[bin_idx + 1])
            else:
                mask = (risk_prob >= edges[bin_idx]) & (risk_prob < edges[bin_idx + 1])
            bin_indices.append(np.flatnonzero(mask))

    ece = 0.0
    for bin_idx, idx in enumerate(bin_indices):
        if adaptive:
            lower = float(risk_prob[idx].min()) if idx.size else 0.0
            upper = float(risk_prob[idx].max()) if idx.size else 0.0
        else:
            lower = float(bin_idx / int(n_bins))
            upper = float((bin_idx + 1) / int(n_bins))
        count = int(idx.size)
        if count:
            mean_risk = float(risk_prob[idx].mean())
            empirical_error = float(y_wrong[idx].mean())
            gap = float(abs(mean_risk - empirical_error))
            fraction = float(count / y_wrong.size)
            ece += fraction * gap
        else:
            mean_risk = 0.0
            empirical_error = 0.0
            gap = 0.0
            fraction = 0.0
        rows.append(
            {
                "bin": int(bin_idx),
                "lower": lower,
                "upper": upper,
                "count": count,
                "fraction": fraction,
                "mean_predicted_risk": mean_risk,
                "empirical_error_rate": empirical_error,
                "empirical_accuracy": float(1.0 - empirical_error),
                "calibration_gap": gap,
            }
        )
    return rows, float(ece)


def _binary_nll(y_wrong, risk_scores):
    y_wrong, risk_scores = _risk_inputs(y_wrong, risk_scores, context="Binary NLL")
    if y_wrong.size == 0:
        return 0.0
    p = np.clip(risk_scores, 1e-8, 1.0 - 1e-8)
    return float(-(y_wrong * np.log(p) + (1 - y_wrong) * np.log(1.0 - p)).mean())


def _calibration_slope_intercept(y_wrong, risk_scores):
    y_wrong, risk_scores = _risk_inputs(y_wrong, risk_scores, context="Calibration slope")
    if y_wrong.size < 3 or np.unique(y_wrong).size < 2 or float(np.std(risk_scores)) <= 1e-12:
        return float("nan"), float("nan")
    p = np.clip(risk_scores, 1e-6, 1.0 - 1e-6)
    logit_p = np.log(p / (1.0 - p)).reshape(-1, 1)
    try:
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        model.fit(logit_p, y_wrong)
        return float(model.coef_[0][0]), float(model.intercept_[0])
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return float("nan"), float("nan")


def residual_risk_metrics(y_wrong, risk_scores, budgets=RESIDUAL_RISK_PAPER_BUDGETS, n_bins=10):
    y_wrong, risk_scores = _risk_inputs(y_wrong, risk_scores, context="Residual-risk metrics")
    budgets = parse_budget_list(budgets, default=RESIDUAL_RISK_PAPER_BUDGETS)
    if y_wrong.size == 0:
        empty = {
            "count": 0,
            "error_count": 0,
            "base_error_rate": 0.0,
            "auroc_error": float("nan"),
            "auprc_error": float("nan"),
            "auprc_error_ratio": float("nan"),
            "aurc": 0.0,
            "oracle_aurc": 0.0,
            "e_aurc": 0.0,
            "ece": 0.0,
            "adaptive_ece": 0.0,
            "brier": 0.0,
            "nll": 0.0,
            "calibration_slope": float("nan"),
            "calibration_intercept": float("nan"),
            "reliability_diagram": {"fixed_bins": [], "adaptive_bins": []},
        }
        for budget in budgets:
            key = int(round(float(budget) * 100))
            empty[f"utility_at_{key}"] = 0.0
            empty[f"ErrRecall@{key}"] = 0.0
            empty[f"Precision@{key}"] = 0.0
            empty[f"Lift@{key}"] = 0.0
        empty["auroc"] = empty["auroc_error"]
        empty["auprc"] = empty["auprc_error"]
        return empty

    risk_prob = np.clip(risk_scores, 1e-8, 1.0 - 1e-8)
    error_count = int(y_wrong.sum())
    base_error_rate = float(y_wrong.mean())
    if 0 < error_count < len(y_wrong):
        auroc_error = float(roc_auc_score(y_wrong, risk_scores))
        auprc_error = float(average_precision_score(y_wrong, risk_scores))
    else:
        auroc_error = float("nan")
        auprc_error = float("nan")

    aurc = _risk_coverage_auc(y_wrong, risk_scores)
    oracle_aurc = oracle_risk_coverage_auc(y_wrong)
    fixed_bins, ece = _reliability_bins(y_wrong, risk_prob, n_bins=n_bins, adaptive=False)
    adaptive_bins, adaptive_ece = _reliability_bins(y_wrong, risk_prob, n_bins=n_bins, adaptive=True)
    slope, intercept = _calibration_slope_intercept(y_wrong, risk_prob)
    metrics = {
        "count": int(y_wrong.size),
        "error_count": error_count,
        "base_error_rate": base_error_rate,
        "auroc_error": auroc_error,
        "auprc_error": auprc_error,
        "auprc_error_ratio": float(auprc_error / base_error_rate) if base_error_rate > 0 and math.isfinite(auprc_error) else float("nan"),
        "aurc": aurc,
        "oracle_aurc": oracle_aurc,
        "e_aurc": float(max(aurc - oracle_aurc, 0.0)),
        "ece": ece,
        "adaptive_ece": adaptive_ece,
        "brier": float(np.mean((risk_prob - y_wrong) ** 2)),
        "nll": _binary_nll(y_wrong, risk_prob),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "reliability_diagram": {
            "fixed_bins": fixed_bins,
            "adaptive_bins": adaptive_bins,
        },
    }
    metrics["auroc"] = metrics["auroc_error"]
    metrics["auprc"] = metrics["auprc_error"]
    for row in residual_risk_budget_curve(y_wrong, risk_scores, budgets=budgets):
        key = int(round(row["budget"] * 100))
        metrics[f"utility_at_{key}"] = row["precision"]
        metrics[f"ErrRecall@{key}"] = row["error_recall"]
        metrics[f"Precision@{key}"] = row["precision"]
        metrics[f"Lift@{key}"] = row["lift"]
    return metrics


def risk_metrics(y_wrong, risk_scores, budgets=RESIDUAL_RISK_PAPER_BUDGETS):
    """Legacy-compatible residual-risk metric surface."""
    return residual_risk_metrics(y_wrong, risk_scores, budgets=budgets)


def _safe_router_metrics(y_wrong, risk_scores, budgets=RESIDUAL_RISK_PAPER_BUDGETS):
    budgets = parse_budget_list(budgets)
    metrics = residual_risk_metrics(y_wrong, risk_scores, budgets=budgets)
    for row in residual_risk_budget_curve(y_wrong, risk_scores, budgets=budgets):
        key = int(round(row["budget"] * 100))
        metrics[f"ErrRecall@{key}"] = row["error_recall"]
        metrics[f"Precision@{key}"] = row["precision"]
        metrics[f"Lift@{key}"] = row["lift"]
    return metrics


def paired_bootstrap_residual_risk_delta(
    y_wrong,
    candidate_scores,
    baseline_scores,
    budgets=RESIDUAL_RISK_PAPER_BUDGETS,
    metric_names=("ErrRecall@30", "ErrRecall@40", "aurc"),
    n_samples=512,
    seed=0,
):
    y_wrong, candidate_scores = _risk_inputs(y_wrong, candidate_scores, context="Paired bootstrap")
    _, baseline_scores = _risk_inputs(y_wrong, baseline_scores, context="Paired bootstrap")
    if y_wrong.size == 0:
        return {}
    budgets = parse_budget_list(budgets)
    rng = np.random.default_rng(int(seed))
    deltas = {name: [] for name in metric_names}
    for _ in range(int(n_samples)):
        sample_idx = rng.choice(np.arange(y_wrong.size), size=y_wrong.size, replace=True)
        y_sample = y_wrong[sample_idx]
        cand = residual_risk_metrics(y_sample, candidate_scores[sample_idx], budgets=budgets)
        base = residual_risk_metrics(y_sample, baseline_scores[sample_idx], budgets=budgets)
        for name in metric_names:
            if name.lower() == "aurc":
                deltas[name].append(float(base.get("aurc", 0.0) - cand.get("aurc", 0.0)))
            else:
                deltas[name].append(float(cand.get(name, 0.0) - base.get(name, 0.0)))

    def _ci(values):
        values = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(values.mean()),
            "p025": float(np.quantile(values, 0.025)),
            "p50": float(np.quantile(values, 0.50)),
            "p975": float(np.quantile(values, 0.975)),
            "significant_positive_95ci": bool(np.quantile(values, 0.025) > 0.0),
        }

    return {
        ("AURC_reduction" if name.lower() == "aurc" else f"{name}_delta"): _ci(values)
        for name, values in deltas.items()
    }


def _threshold_checks(metrics, baselines, thresholds):
    checks = []
    base_error_rate = float(metrics.get("base_error_rate", 0.0))
    auprc_ratio = metrics.get("auprc_error_ratio")
    if auprc_ratio is None or not math.isfinite(float(auprc_ratio)):
        auprc = float(metrics.get("auprc_error", float("nan")))
        auprc_ratio = float(auprc / base_error_rate) if base_error_rate > 0 and math.isfinite(auprc) else float("nan")

    def _add(name, value, threshold, relation=">="):
        value = float(value) if value is not None else float("nan")
        passed = bool(math.isfinite(value) and value >= float(threshold))
        checks.append({"metric": name, "value": value, "threshold": float(threshold), "relation": relation, "passed": passed})

    for name in ("ErrRecall@20", "ErrRecall@30", "ErrRecall@40", "Lift@20", "Lift@30", "Lift@40", "auroc_error"):
        _add(name, metrics.get(name, float("nan")), thresholds[name])
    _add("auprc_error_ratio", auprc_ratio, thresholds["auprc_error_ratio"])

    aurc = float(metrics.get("aurc", float("nan")))
    reductions = {}
    for baseline_name in ("msp", "entropy", "msp_ts"):
        baseline = (baselines or {}).get(baseline_name)
        if not baseline:
            continue
        baseline_aurc = float(baseline.get("aurc", float("nan")))
        if math.isfinite(aurc) and math.isfinite(baseline_aurc) and baseline_aurc > 0:
            reductions[baseline_name] = float((baseline_aurc - aurc) / baseline_aurc)
    if reductions:
        value = min(reductions.values())
        _add("aurc_relative_reduction_vs_uncertainty", value, thresholds["aurc_relative_reduction"])
    else:
        checks.append(
            {
                "metric": "aurc_relative_reduction_vs_uncertainty",
                "value": float("nan"),
                "threshold": float(thresholds["aurc_relative_reduction"]),
                "relation": ">=",
                "passed": False,
                "reason": "msp/entropy baseline metrics not available",
            }
        )
    return checks, reductions


def stage2_acceptance_gate(metrics, baselines=None, calibration_reference=None):
    baselines = baselines or {}
    min_checks, min_reductions = _threshold_checks(metrics, baselines, STAGE2_MINIMUM_THRESHOLDS)
    strong_checks, strong_reductions = _threshold_checks(metrics, baselines, STAGE2_STRONG_THRESHOLDS)
    minimum_passed = all(item["passed"] for item in min_checks)
    strong_passed = all(item["passed"] for item in strong_checks)

    ece = float(metrics.get("ece", float("nan")))
    adaptive_ece = float(metrics.get("adaptive_ece", float("nan")))
    brier = float(metrics.get("brier", float("nan")))
    nll = float(metrics.get("nll", float("nan")))
    slope = float(metrics.get("calibration_slope", float("nan")))
    intercept = float(metrics.get("calibration_intercept", float("nan")))
    raw_ece = None
    if calibration_reference:
        raw_ece = calibration_reference.get("ece")
    ece_relative_reduction = (
        float((float(raw_ece) - ece) / float(raw_ece))
        if raw_ece is not None and math.isfinite(float(raw_ece)) and float(raw_ece) > 0 and math.isfinite(ece)
        else float("nan")
    )
    calibration_checks = [
        {
            "metric": "ece",
            "value": ece,
            "threshold": 0.05,
            "relation": "<= or >=20% relative reduction",
            "passed": bool((math.isfinite(ece) and ece <= 0.05) or (math.isfinite(ece_relative_reduction) and ece_relative_reduction >= 0.20)),
            "relative_reduction_vs_reference": ece_relative_reduction,
        },
        {"metric": "adaptive_ece", "value": adaptive_ece, "threshold": 0.05, "relation": "<=", "passed": bool(math.isfinite(adaptive_ece) and adaptive_ece <= 0.05)},
        {"metric": "calibration_slope", "value": slope, "lower": 0.8, "upper": 1.2, "relation": "within", "passed": bool((not math.isfinite(slope)) or (0.8 <= slope <= 1.2))},
        {"metric": "calibration_intercept", "value": intercept, "lower": -0.5, "upper": 0.5, "relation": "within", "passed": bool((not math.isfinite(intercept)) or (-0.5 <= intercept <= 0.5))},
    ]
    if calibration_reference:
        ref_brier = calibration_reference.get("brier")
        ref_nll = calibration_reference.get("nll")
        if ref_brier is not None:
            calibration_checks.append(
                {"metric": "brier_vs_reference", "value": brier, "threshold": float(ref_brier), "relation": "<=", "passed": bool(math.isfinite(brier) and brier <= float(ref_brier) + 1e-12)}
            )
        if ref_nll is not None:
            calibration_checks.append(
                {"metric": "nll_vs_reference", "value": nll, "threshold": float(ref_nll), "relation": "<=", "passed": bool(math.isfinite(nll) and nll <= float(ref_nll) + 1e-12)}
            )
    calibration_passed = all(item["passed"] for item in calibration_checks)
    if strong_passed and calibration_passed:
        claim_status = "strong_calibrated_residual_risk_estimator"
    elif minimum_passed and calibration_passed:
        claim_status = "minimum_calibrated_residual_risk_estimator"
    elif strong_passed or minimum_passed:
        claim_status = "residual_ranker_only_calibration_not_estimator"
    else:
        claim_status = "not_ready_for_paper_claim"
    return {
        "contract": "stage2_acceptance_gate_v1",
        "paper_name": "Calibrated Residual-Risk Hard-Node Selector",
        "task": "estimate P(Stage1 prediction is wrong | Stage1 outputs/features)",
        "screening_only": True,
        "diagnosis_or_action": False,
        "minimum": {"passed": minimum_passed, "checks": min_checks, "aurc_relative_reductions": min_reductions},
        "strong": {"passed": strong_passed, "checks": strong_checks, "aurc_relative_reductions": strong_reductions},
        "calibration": {"passed": calibration_passed, "checks": calibration_checks},
        "claim_status": claim_status,
    }


def _normalize_probs(probs):
    probs = _to_tensor(probs).float().cpu()
    probs = probs.clamp_min(1e-8)
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)


def _js_divergence(p, q):
    p = _normalize_probs(p)
    q = _normalize_probs(q)
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p) - torch.log(m))).sum(dim=1)
    kl_qm = (q * (torch.log(q) - torch.log(m))).sum(dim=1)
    return 0.5 * (kl_pm + kl_qm)


def _neighbor_probability_context(probs, edge_index, edge_type=None):
    probs = _normalize_probs(probs)
    edge_index = _to_tensor(edge_index).long().cpu()
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must be shaped [2, num_edges] for residual-risk graph context features.")
    if edge_index.numel() == 0:
        num_nodes = int(probs.size(0))
        return probs.clone(), torch.zeros(num_nodes), torch.zeros(num_nodes)

    src = edge_index[0].long()
    dst = edge_index[1].long()
    num_nodes = int(probs.size(0))
    if int(src.min().item()) < 0 or int(dst.min().item()) < 0 or int(src.max().item()) >= num_nodes or int(dst.max().item()) >= num_nodes:
        raise ValueError("edge_index contains node ids outside the probability tensor range.")
    ctx = torch.zeros_like(probs)
    deg = torch.zeros(num_nodes, dtype=torch.float32)
    ctx.index_add_(0, dst, probs[src])
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
    ctx = ctx / deg.clamp_min(1.0).unsqueeze(1)
    isolated = deg <= 0
    if isolated.any():
        ctx[isolated] = probs[isolated]

    relation_skew = torch.zeros(num_nodes, dtype=torch.float32)
    if edge_type is not None:
        edge_type = _to_tensor(edge_type).long().cpu().view(-1)
        if edge_type.numel():
            if edge_type.numel() != src.numel():
                raise ValueError("edge_type length must match edge_index num_edges for residual-risk relation summaries.")
            if int(edge_type.min().item()) < 0:
                raise ValueError("edge_type must contain non-negative relation ids.")
            n_rel = int(edge_type.max().item()) + 1
            hist = torch.zeros(num_nodes, n_rel, dtype=torch.float32)
            for rel_id in range(n_rel):
                rel_mask = edge_type == rel_id
                if rel_mask.any():
                    rel_nodes = dst[rel_mask]
                    hist.index_put_(
                        (rel_nodes, torch.full_like(rel_nodes, rel_id)),
                        torch.ones(int(rel_mask.sum()), dtype=torch.float32),
                        accumulate=True,
                    )
            total = hist.sum(dim=1, keepdim=True).clamp_min(1.0)
            relation_skew = torch.abs(hist / total - (1.0 / max(n_rel, 1))).sum(dim=1)
    return ctx, deg, relation_skew


def _minmax01(values):
    values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if values.size == 0:
        return values.astype(np.float32)
    low = float(values.min())
    high = float(values.max())
    if high - low <= 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - low) / (high - low)).astype(np.float32)


def _optional_node_matrix(value, num_nodes, name):
    if value is None:
        return None
    matrix = _to_numpy(value).astype(np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D node-level array.")
    if int(matrix.shape[0]) != int(num_nodes):
        raise ValueError(f"{name} rows must match num_nodes={num_nodes}, got {matrix.shape[0]}.")
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def _degree_statistics(edge_index, num_nodes):
    """Degree features matching GLANCE's relative-degree prior.

    For directed edge_index we treat incoming sources as N(v), which matches the
    existing message-passing context used by Stage 2 graph features:

        bar_d_v = mean_{u in N(v)} sqrt((d_v + 1) / (d_u + 1)).
    """
    in_degree = np.zeros(num_nodes, dtype=np.float32)
    out_degree = np.zeros(num_nodes, dtype=np.float32)
    relative_degree = np.ones(num_nodes, dtype=np.float32)
    if edge_index is None:
        return {
            "in_degree": in_degree,
            "out_degree": out_degree,
            "total_degree": in_degree + out_degree,
            "relative_degree": relative_degree,
        }

    edge_index_t = _to_tensor(edge_index).long().cpu()
    if edge_index_t.numel() == 0:
        return {
            "in_degree": in_degree,
            "out_degree": out_degree,
            "total_degree": in_degree + out_degree,
            "relative_degree": relative_degree,
        }
    if edge_index_t.dim() != 2 or edge_index_t.size(0) != 2:
        raise ValueError("edge_index must be shaped [2, num_edges] for GLANCE routing features.")
    src = edge_index_t[0].numpy()
    dst = edge_index_t[1].numpy()
    if src.size:
        if int(src.min()) < 0 or int(dst.min()) < 0 or int(src.max()) >= num_nodes or int(dst.max()) >= num_nodes:
            raise ValueError("edge_index contains node ids outside the GLANCE feature range.")
    np.add.at(out_degree, src, 1.0)
    np.add.at(in_degree, dst, 1.0)
    total_degree = in_degree + out_degree

    rel_sum = np.zeros(num_nodes, dtype=np.float32)
    rel_count = np.zeros(num_nodes, dtype=np.float32)
    contribution = np.sqrt((total_degree[dst] + 1.0) / np.maximum(total_degree[src] + 1.0, 1e-8))
    np.add.at(rel_sum, dst, contribution.astype(np.float32))
    np.add.at(rel_count, dst, 1.0)
    has_neighbors = rel_count > 0
    relative_degree[has_neighbors] = rel_sum[has_neighbors] / rel_count[has_neighbors]
    return {
        "in_degree": in_degree,
        "out_degree": out_degree,
        "total_degree": total_degree,
        "relative_degree": relative_degree,
    }


def _soft_estimated_homophily(q_probs, edge_index):
    """Label-free GLANCE soft local homophily estimate.

    Loveland et al. replace label homophily with a posterior-only proxy:

        hat_h_v = p_Q,v dot mean_{u in N_1(v)} p_Q,u.

    Isolated nodes have no observed neighborhood alignment and receive 0.0.
    """
    q_probs_t = _normalize_probs(q_probs)
    num_nodes = int(q_probs_t.size(0))
    if edge_index is None:
        return np.zeros(num_nodes, dtype=np.float32), "not_available_no_graph"
    context_probs, in_degree, _ = _neighbor_probability_context(q_probs_t, edge_index)
    soft_homophily = (q_probs_t * context_probs).sum(dim=1).numpy().astype(np.float32)
    soft_homophily[in_degree.numpy() <= 0] = 0.0
    return np.clip(soft_homophily, 0.0, 1.0), "available"


def _feature_matrix_from_map(feature_map, feature_names, num_nodes):
    columns = []
    for name in feature_names:
        values = feature_map.get(name)
        if values is None:
            values = np.zeros(num_nodes, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.shape[0] != num_nodes:
            raise ValueError(f"Feature {name} has {values.shape[0]} rows, expected {num_nodes}.")
        columns.append(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0))
    if not columns:
        return np.zeros((num_nodes, 0), dtype=np.float32)
    return np.column_stack(columns).astype(np.float32)


def _add_node_matrix_summaries(feature_map, feature_names, matrix, prefix, max_direct_dims=0):
    if matrix is None:
        return [], []
    centroid = matrix.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(matrix, axis=1)
    centroid_distance = np.linalg.norm(matrix - centroid, axis=1)
    added = [f"{prefix}_norm", f"{prefix}_centroid_distance"]
    feature_map[added[0]] = norm.astype(np.float32)
    feature_map[added[1]] = centroid_distance.astype(np.float32)
    feature_names.extend(added)

    direct_names = []
    direct_dims = min(max(int(max_direct_dims), 0), int(matrix.shape[1]))
    for dim in range(direct_dims):
        name = f"{prefix}_dim_{dim:03d}"
        feature_map[name] = matrix[:, dim].astype(np.float32)
        feature_names.append(name)
        direct_names.append(name)
    return added, direct_names


def build_glance_for_context_router_features(
    logits,
    probs,
    edge_index=None,
    edge_type=None,
    lm_probs=None,
    q_probs=None,
    node_repr=None,
    node_features=None,
    mc_dropout_uncertainty=None,
    max_embedding_dims=32,
    max_node_feature_dims=0,
):
    """GLANCE-inspired cheap node-aware routing features for residual-risk ranking.

    Loveland et al. use GNN embeddings, dropout uncertainty, degree, node features,
    and a label-free soft local homophily estimate to decide which nodes should
    query an LLM. This helper adapts those signals into a post-hoc Stage 2
    residual-risk feature matrix; it does not invoke an LLM or choose repair actions.
    """
    posterior_block = _posterior_residual_feature_block(logits=logits, probs=probs)
    probs_t = posterior_block["probs_t"]
    probs_np = posterior_block["probs_np"]
    predictions = posterior_block["predictions"]
    num_nodes = int(probs_np.shape[0])

    feature_map = {}
    feature_names = []
    for name, column in zip(posterior_block["feature_names"], posterior_block["columns"]):
        feature_map[f"glance_{name}"] = np.asarray(column, dtype=np.float32)
        feature_names.append(f"glance_{name}")

    degree_stats = _degree_statistics(edge_index, num_nodes)
    total_degree = degree_stats["total_degree"]
    log_total_degree = np.log1p(total_degree)
    sparse_degree_score = 1.0 - _minmax01(log_total_degree)
    relative_degree = np.nan_to_num(degree_stats["relative_degree"], nan=1.0, posinf=1.0, neginf=1.0)
    low_relative_degree_score = np.clip(1.0 - relative_degree, 0.0, 1.0)

    if edge_index is not None:
        context_probs, _, relation_skew_t = _neighbor_probability_context(probs_t, edge_index, edge_type=edge_type)
        context_np = context_probs.numpy()
        local_support = context_np[np.arange(num_nodes), predictions]
        local_inconsistency = 1.0 - local_support
        neighbor_context_gap = np.abs(probs_np - context_np).sum(axis=1) / 2.0
        relation_skew = relation_skew_t.numpy()
    else:
        local_support = np.zeros(num_nodes, dtype=np.float32)
        local_inconsistency = np.ones(num_nodes, dtype=np.float32)
        neighbor_context_gap = np.zeros(num_nodes, dtype=np.float32)
        relation_skew = np.zeros(num_nodes, dtype=np.float32)

    if q_probs is not None:
        q_source = "q_probs"
        q_probs_for_homophily = q_probs
    elif lm_probs is not None:
        q_source = "lm_probs"
        q_probs_for_homophily = lm_probs
    else:
        q_source = "stage1_gnn_probs"
        q_probs_for_homophily = probs_t
    soft_homophily, homophily_status = _soft_estimated_homophily(q_probs_for_homophily, edge_index)
    estimated_heterophily = 1.0 - soft_homophily

    entropy_score = np.asarray(posterior_block["base_scores"][RISK_CANDIDATE_ENTROPY], dtype=np.float32)
    uncertainty_score = entropy_score
    if mc_dropout_uncertainty is not None:
        uncertainty_score = _to_numpy(mc_dropout_uncertainty).astype(np.float32).reshape(-1)
        if uncertainty_score.shape[0] != num_nodes:
            raise ValueError("mc_dropout_uncertainty must provide one uncertainty value per node.")
        uncertainty_score = np.clip(_minmax01(uncertainty_score), 0.0, 1.0)

    structural_values = {
        "glance_log_total_degree": log_total_degree,
        "glance_sparse_degree_score": sparse_degree_score,
        "glance_relative_degree": relative_degree,
        "glance_low_relative_degree_score": low_relative_degree_score,
        "glance_soft_estimated_homophily": soft_homophily,
        "glance_estimated_heterophily": estimated_heterophily,
        "glance_local_prediction_support": local_support,
        "glance_local_prediction_inconsistency": local_inconsistency,
        "glance_neighbor_context_gap": neighbor_context_gap,
        "glance_relation_skew": relation_skew,
        "glance_uncertainty_score": uncertainty_score,
    }
    for name, values in structural_values.items():
        feature_map[name] = np.asarray(values, dtype=np.float32)
        feature_names.append(name)

    dual_block = _dual_posterior_residual_feature_block(
        lm_probs=lm_probs,
        gnn_probs_t=probs_t,
        gnn_probs_np=probs_np,
        predictions=predictions,
    )
    for name, column in zip(dual_block["feature_names"], dual_block["columns"]):
        feature_name = f"glance_{name}"
        feature_map[feature_name] = np.asarray(column, dtype=np.float32)
        feature_names.append(feature_name)

    embedding_matrix = _optional_node_matrix(node_repr, num_nodes, "node_repr")
    embedding_summary_names, embedding_direct_names = _add_node_matrix_summaries(
        feature_map,
        feature_names,
        embedding_matrix,
        prefix="glance_gnn_embedding",
        max_direct_dims=max_embedding_dims,
    )
    raw_feature_matrix = _optional_node_matrix(node_features, num_nodes, "node_features")
    node_feature_summary_names, node_feature_direct_names = _add_node_matrix_summaries(
        feature_map,
        feature_names,
        raw_feature_matrix,
        prefix="glance_node_feature",
        max_direct_dims=max_node_feature_dims,
    )

    features = _feature_matrix_from_map(feature_map, feature_names, num_nodes)
    msp_score = np.asarray(posterior_block["base_scores"][RISK_CANDIDATE_MSP], dtype=np.float32)
    base_scores = dict(posterior_block["base_scores"])
    base_scores[GLANCE_SOFT_HOMOPHILY_PRIOR] = np.clip(estimated_heterophily, 0.0, 1.0).astype(np.float32)
    base_scores[GLANCE_DEGREE_HOMOPHILY_PRIOR] = np.clip(
        0.50 * estimated_heterophily + 0.30 * sparse_degree_score + 0.20 * low_relative_degree_score,
        0.0,
        1.0,
    ).astype(np.float32)
    base_scores[GLANCE_UNCERTAINTY_HOMOPHILY_BLEND] = np.clip(
        0.35 * msp_score + 0.25 * uncertainty_score + 0.25 * estimated_heterophily + 0.15 * local_inconsistency,
        0.0,
        1.0,
    ).astype(np.float32)

    return {
        "features": features,
        "feature_map": feature_map,
        "feature_names": feature_names,
        "structural_feature_names": list(structural_values.keys()),
        "embedding_feature_names": list(embedding_summary_names) + list(embedding_direct_names),
        "node_feature_names": list(node_feature_summary_names) + list(node_feature_direct_names),
        "base_scores": base_scores,
        "num_nodes": num_nodes,
        "homophily_estimator_source": q_source,
        "homophily_status": homophily_status,
        "formula_reference": dict(GLANCE_FORMULA_REFERENCE),
        "used_node_repr": embedding_matrix is not None,
        "used_node_features": raw_feature_matrix is not None,
        "used_mc_dropout_uncertainty": mc_dropout_uncertainty is not None,
    }


def _residual_candidate_selection_key(candidate_name, metrics):
    aurc = float(metrics.get("aurc", float("inf")))
    if not math.isfinite(aurc):
        aurc = float("inf")
    auroc_error = float(metrics.get("auroc_error", float("nan")))
    auroc_tiebreak = -auroc_error if math.isfinite(auroc_error) else 0.0
    return (
        aurc,
        -float(metrics.get("ErrRecall@30", 0.0)),
        -float(metrics.get("ErrRecall@40", 0.0)),
        auroc_tiebreak,
        str(candidate_name),
    )


def _select_residual_risk_candidate(y_wrong, candidate_scores, selection_idx, budgets=RESIDUAL_RISK_PAPER_BUDGETS):
    y_wrong = _to_numpy(y_wrong).astype(np.int32).reshape(-1)
    selection_idx = _valid_index_array(selection_idx, y_wrong.size)
    if selection_idx.size == 0:
        first_name = next(iter(candidate_scores))
        return {
            "selected_candidate": first_name,
            "status": "fallback_no_validation_nodes",
            "validation_metrics": {},
            "candidate_order": [first_name],
        }

    validation_metrics = {}
    sortable = []
    for name, score in candidate_scores.items():
        if score is None:
            continue
        score_np = np.asarray(score, dtype=np.float32).reshape(-1)
        if score_np.shape[0] != y_wrong.shape[0]:
            continue
        metrics = residual_risk_metrics(y_wrong[selection_idx], score_np[selection_idx], budgets=budgets)
        validation_metrics[name] = metrics
        sortable.append(_residual_candidate_selection_key(name, metrics))
    if not sortable:
        first_name = next(iter(candidate_scores))
        return {
            "selected_candidate": first_name,
            "status": "fallback_no_scored_candidates",
            "validation_metrics": validation_metrics,
            "candidate_order": [first_name],
        }
    sortable.sort()
    ordered = [item[-1] for item in sortable]
    return {
        "selected_candidate": ordered[0],
        "status": "selected_by_validation_aurc_then_errrecall",
        "validation_metrics": validation_metrics,
        "candidate_order": ordered,
        "selection_metric": "min_validation_AURC_tiebreak_ErrRecall@30_ErrRecall@40_AUROC",
    }


def _base_residual_candidate_metadata():
    return {
        RISK_CANDIDATE_MSP: {"status": "available", "family": "posterior", "baseline": "MSP risk = 1 - max p_i"},
        RISK_CANDIDATE_ENTROPY: {"status": "available", "family": "posterior", "baseline": "normalized predictive entropy"},
        RISK_CANDIDATE_MARGIN: {"status": "available", "family": "posterior", "baseline": "1 - top1/top2 probability margin"},
    }


def _one_hot_candidate_view_tensors(candidate_scores, selected_candidate):
    candidate_names = list(candidate_scores)
    view_matrix = np.column_stack([np.asarray(candidate_scores[name], dtype=np.float32) for name in candidate_names])
    selected_column = candidate_names.index(selected_candidate)
    weights = np.zeros((view_matrix.shape[0], len(candidate_names)), dtype=np.float32)
    weights[:, selected_column] = 1.0
    return candidate_names, torch.tensor(view_matrix, dtype=torch.float32), torch.tensor(weights, dtype=torch.float32)


def build_probe_features(logits, probs, edge_index, edge_type, node_repr):
    logits = _to_tensor(logits).cpu()
    probs = _to_tensor(probs).cpu()
    edge_index = _to_tensor(edge_index).cpu()
    edge_type = _to_tensor(edge_type).cpu()
    node_repr = _to_tensor(node_repr).cpu()

    num_nodes = probs.shape[0]
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    et = edge_type.numpy()
    preds = probs.argmax(dim=1).numpy()

    entropy = (-probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1).numpy()
    msp = probs.max(dim=1)[0].numpy()

    in_deg = np.zeros(num_nodes, dtype=np.float32)
    out_deg = np.zeros(num_nodes, dtype=np.float32)
    np.add.at(in_deg, dst, 1)
    np.add.at(out_deg, src, 1)

    if et.size > 0:
        relation_buckets = int(et.max()) + 1
    else:
        relation_buckets = 1
    relation_hist = np.zeros((num_nodes, relation_buckets), dtype=np.float32)
    for relation_id in range(relation_buckets):
        mask = et == relation_id
        np.add.at(relation_hist[:, relation_id], dst[mask], 1)
    relation_total = np.clip(relation_hist.sum(axis=1, keepdims=True), a_min=1.0, a_max=None)
    relation_skew = np.abs(relation_hist / relation_total - (1.0 / relation_buckets)).sum(axis=1)

    in_neighbors = [[] for _ in range(num_nodes)]
    for src_idx, dst_idx in zip(src, dst):
        in_neighbors[dst_idx].append(src_idx)

    neigh_pred_agreement = np.full(num_nodes, 0.5, dtype=np.float32)
    neigh_inconsistency = np.zeros(num_nodes, dtype=np.float32)
    for node_idx in range(num_nodes):
        neighbors = in_neighbors[node_idx]
        if not neighbors:
            continue
        nb_preds = preds[neighbors]
        neigh_pred_agreement[node_idx] = np.mean(nb_preds == preds[node_idx])
        neigh_inconsistency[node_idx] = np.mean(nb_preds != preds[node_idx])

    base_repr = F.normalize(node_repr, dim=1)
    cosine_div = (1.0 - (base_repr * base_repr.mean(dim=0, keepdim=True)).sum(dim=1)).numpy()

    features = np.column_stack(
        [
            entropy,
            1.0 - msp,
            in_deg,
            out_deg,
            relation_skew,
            neigh_pred_agreement,
            cosine_div,
            neigh_inconsistency,
        ]
    )

    sparse_evidence_score = (1.0 - np.clip(in_deg / max(np.percentile(in_deg[in_deg > 0], 50), 1), 0, 1)) * 0.6 + relation_skew * 0.4
    prop_corruption_score = neigh_inconsistency * 0.4 + cosine_div * 0.3 + (1.0 - neigh_pred_agreement) * 0.3

    return {
        "features": features,
        "entropy": entropy,
        "msp": msp,
        "in_degree": in_deg,
        "out_degree": out_deg,
        "relation_skew": relation_skew,
        "neigh_pred_agreement": neigh_pred_agreement,
        "neigh_inconsistency": neigh_inconsistency,
        "sparse_evidence_score": sparse_evidence_score,
        "prop_corruption_score": prop_corruption_score,
    }


class BaseRiskEstimator:
    metadata = {
        "claim_role": "supporting_baseline",
        "scientific_gate": "proxy_first",
        "paper_identity_risk": "low",
        "promotion_rule": "never",
    }

    def __init__(self):
        self.val_threshold = None
        self.calibration_metadata = {}

    def fit(self, logits, probs, labels, train_idx, val_idx, edge_index=None, edge_type=None, node_repr=None, **kwargs):
        raise NotImplementedError

    def score(self, logits, probs, edge_index=None, edge_type=None, node_repr=None, **kwargs):
        raise NotImplementedError

    def build_manifest(self, logits, probs, labels, val_idx, test_idx, edge_index=None, edge_type=None, node_repr=None, **kwargs):
        risk_score = self.score(logits, probs, edge_index=edge_index, edge_type=edge_type, node_repr=node_repr, **kwargs)
        labels_np = _to_numpy(labels)
        if labels_np.ndim > 1:
            labels_np = labels_np.argmax(axis=1)
        preds = _to_numpy(probs).argmax(axis=1)
        wrong = (preds != labels_np).astype(np.int32)

        val_idx_np = _to_numpy(val_idx).astype(np.int64)
        test_idx_np = _to_numpy(test_idx).astype(np.int64)
        budgets = kwargs.get("budgets", (0.05, 0.10, 0.15, 0.20))
        val_metrics = risk_metrics(wrong[val_idx_np], risk_score[val_idx_np], budgets=budgets)
        test_metrics = risk_metrics(wrong[test_idx_np], risk_score[test_idx_np], budgets=budgets)

        if self.val_threshold is None:
            budget = max(int(len(val_idx_np) * 0.15), 1)
            self.val_threshold = float(np.sort(risk_score[val_idx_np])[-budget])

        hcw_mask = ((1.0 - _to_numpy(probs).max(axis=1)) < 0.1) & (wrong == 1)
        return {
            "risk_score": risk_score.astype(np.float32),
            "thresholds": {"validation_risk_threshold": float(self.val_threshold)},
            "calibration_metadata": dict(self.calibration_metadata),
            "hcw_mask": hcw_mask.tolist(),
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
        }


class MSPTemperatureEstimator(BaseRiskEstimator):
    metadata = {
        "claim_role": "anchor_control",
        "scientific_gate": "phase0_required",
        "paper_identity_risk": "low",
        "promotion_rule": "never",
    }

    def __init__(self):
        super().__init__()
        self.temperature = 1.0

    def fit(self, logits, probs, labels, train_idx, val_idx, edge_index=None, edge_type=None, node_repr=None, **kwargs):
        labels_np = _to_numpy(labels)
        if labels_np.ndim > 1:
            labels_np = labels_np.argmax(axis=1)
        val_idx_np = _to_numpy(val_idx).astype(np.int64)
        self.temperature = fit_temperature_scaling(_to_tensor(logits)[val_idx_np], labels_np[val_idx_np])
        self.calibration_metadata = {"temperature": float(self.temperature)}
        return self

    def score(self, logits, probs, edge_index=None, edge_type=None, node_repr=None, **kwargs):
        logits = _to_tensor(logits)
        scaled_probs = F.softmax(logits / self.temperature, dim=1).cpu().numpy()
        return 1.0 - scaled_probs.max(axis=1)


class LOGINUncertaintyRouter(BaseRiskEstimator):
    metadata = {
        "claim_role": "stage_b_login_official_uncertainty_router",
        "stage2_role": "hard_node_selector",
        "canonical_stage2": False,
        "scientific_gate": "phase_a_frozen_gnn_uncertainty_only",
        "paper_identity": "LOGIN_init Node-Selection Uncertainty Router",
        "paper_identity_risk": "uncertainty_submodule_only",
        "promotion_rule": "ablation_only_never_canonical",
        "screening_only": True,
        "diagnosis_or_action": False,
        "llm_call": False,
        "login_scope": "node_selection_uncertainty_only",
    }

    OFFICIAL_REPO_PATH = r"G:\Research\BotDetection\LOGIN_init"
    OFFICIAL_FORMULA = "var(stack(logits_list), dim=0).sum(dim=1)"
    OFFICIAL_SELECTION_CONTRACT = "global_topk_by_pl_rate"
    OFFICIAL_DROPOUT_LIST = [0.5, 0.5, 0.5, 0.5, 0.5]
    OFFICIAL_PL_RATE = 0.1

    def __init__(self, budgets=RESIDUAL_RISK_PAPER_BUDGETS, official_code_verified=False):
        super().__init__()
        self.budgets = parse_budget_list(budgets, default=RESIDUAL_RISK_PAPER_BUDGETS)
        self.official_code_verified = bool(official_code_verified)
        self.last_risk_score = None
        self.last_official_pl_mask = None
        self.last_logits_shape = None
        self.fit_summary = {}

    @staticmethod
    def _input_boundary_audit(edge_index=None, edge_type=None, node_repr=None, **kwargs):
        return {
            "allowed_inputs": [
                "five_independent_gnn_training_logits_list",
                "reference_frozen_gnn_logits_or_probabilities_for_auxiliary_analysis_only",
                "validation_split_labels_for_auxiliary_budget_analysis_only",
                "test_split_labels_for_reporting_only",
            ],
            "raw_text": False,
            "raw_semantic_embedding": False,
            "qwen_embedding": False,
            "roberta_embedding": False,
            "gnn_hidden_state": False,
            "edge_index_argument_ignored": bool(edge_index is not None),
            "edge_type_argument_ignored": bool(edge_type is not None),
            "node_repr_argument_ignored": bool(node_repr is not None),
            "llm_response": False,
            "llm_call": False,
            "semantic_feature_update": False,
            "structure_refinement": False,
            "gnn_retrain": True,
            "test_labels_for_threshold": False,
            "forbidden_inputs_ignored": {
                name: kwargs.get(name) is not None
                for name in (
                    "raw_text",
                    "raw_semantic_embedding",
                    "qwen_embedding",
                    "roberta_embedding",
                    "llm_response",
                    "semantic_feature_update",
                    "structure_refinement",
                    "action_outputs",
                    "repair_outputs",
                )
            },
            "enforcement": "official_login_uncertainty_submodule_only",
        }

    def _posterior_probabilities(self, logits=None, probs=None):
        if logits is not None:
            logits_t = _to_tensor(logits).float()
            if logits_t.dim() != 2:
                raise ValueError("login_uncertainty_router expects logits shaped [num_nodes, num_classes].")
            return F.softmax(logits_t, dim=1).detach().cpu().numpy()
        if probs is None:
            raise ValueError("login_uncertainty_router requires frozen GNN logits or probabilities.")
        probs_t = _normalize_probs(probs)
        if probs_t.dim() != 2:
            raise ValueError("login_uncertainty_router expects probabilities shaped [num_nodes, num_classes].")
        return probs_t.detach().cpu().numpy()

    def _prepare_logits_list(self, logits_list, reference_logits=None):
        if logits_list is None:
            raise ValueError(
                "login_uncertainty_router requires logits_list from five independent GNN training runs. "
                "MC-dropout and single-posterior proxies are not allowed in the official uncertainty path."
            )
        if torch.is_tensor(logits_list):
            logits_stack = logits_list.detach().cpu().float()
        elif isinstance(logits_list, (list, tuple)):
            if not logits_list:
                raise ValueError("login_uncertainty_router received an empty logits_list.")
            logits_stack = torch.stack([_to_tensor(item).detach().cpu().float() for item in logits_list], dim=0)
        else:
            logits_stack = _to_tensor(logits_list).detach().cpu().float()
        if logits_stack.dim() != 3:
            raise ValueError("login_uncertainty_router expects logits_list shaped [runs, num_nodes, num_classes].")
        if logits_stack.size(0) < 2:
            raise ValueError("login_uncertainty_router requires at least two GNN runs to compute variance.")
        if reference_logits is not None:
            ref = _to_tensor(reference_logits)
            if ref.dim() != 2:
                raise ValueError("Reference logits for login_uncertainty_router must be shaped [num_nodes, num_classes].")
            if tuple(logits_stack.shape[1:]) != tuple(ref.shape):
                raise ValueError(
                    "logits_list runs must align with the reference frozen GNN logits shape "
                    f"{tuple(ref.shape)}, got {tuple(logits_stack.shape[1:])}."
                )
        return logits_stack

    def _compute_uncertainty_score(self, logits_list, logits=None, probs=None):
        reference_logits = logits
        if reference_logits is None and probs is not None:
            probs_t = _normalize_probs(probs)
            reference_logits = probs_t
        logits_stack = self._prepare_logits_list(logits_list, reference_logits=reference_logits)
        variance = torch.var(logits_stack, dim=0)
        uncertainty_score = torch.sum(variance, dim=1).detach().cpu().float()
        self.last_logits_shape = [int(item) for item in logits_stack.shape]
        return uncertainty_score, logits_stack

    def _official_topk_mask(self, uncertainty_score):
        uncertainty_score = _to_tensor(uncertainty_score).detach().cpu().float().reshape(-1)
        k = int(float(self.OFFICIAL_PL_RATE) * int(uncertainty_score.numel()))
        pl_mask = torch.zeros_like(uncertainty_score, dtype=torch.bool)
        if k > 0:
            topk_indices = torch.topk(uncertainty_score, k).indices
            pl_mask[topk_indices] = True
        return pl_mask

    def fit(self, logits, probs, labels, train_idx, val_idx, edge_index=None, edge_type=None, node_repr=None, **kwargs):
        self.budgets = parse_budget_list(kwargs.get("budgets", self.budgets), default=RESIDUAL_RISK_PAPER_BUDGETS)
        uncertainty_score, logits_stack = self._compute_uncertainty_score(
            kwargs.get("logits_list"),
            logits=logits,
            probs=probs,
        )
        risk_score = uncertainty_score.detach().cpu().numpy()
        val_idx_np = _valid_index_array(val_idx, risk_score.shape[0])
        budget = self.budgets[0] if self.budgets else RESIDUAL_RISK_PAPER_BUDGETS[0]
        k = max(int(val_idx_np.size * float(budget)), 1) if val_idx_np.size else 0
        self.val_threshold = float(np.sort(risk_score[val_idx_np])[-k]) if k else float("inf")
        self.last_risk_score = risk_score.astype(np.float32)
        self.last_official_pl_mask = self._official_topk_mask(uncertainty_score).detach().cpu().numpy().astype(bool)
        self.fit_summary = {
            "source": "login_uncertainty_router",
            "fit_scope": "official_uncertainty_score_plus_auxiliary_validation_budget_analysis",
            "validation_count": int(val_idx_np.size),
            "auxiliary_validation_budget_for_threshold": float(budget),
            "auxiliary_validation_risk_threshold": float(self.val_threshold),
            "official_code_verified": bool(self.official_code_verified),
            "paper_aligned": True,
            "repo_locally_verified": False,
            "verified_scope": "node_selection_uncertainty_only",
            "official_repo_path": self.OFFICIAL_REPO_PATH,
            "official_submodule_scope": "node_selection_uncertainty_only",
            "official_formula": self.OFFICIAL_FORMULA,
            "official_selection_contract": self.OFFICIAL_SELECTION_CONTRACT,
            "official_drop_out_list": list(self.OFFICIAL_DROPOUT_LIST),
            "official_pl_rate": float(self.OFFICIAL_PL_RATE),
            "auxiliary_budget_analysis": True,
            "full_LOGIN_loop": False,
            "prompt_llm": False,
            "feature_update": False,
            "edge_pruning": False,
            "logits_list_shape": list(self.last_logits_shape or [int(logits_stack.size(0)), int(logits_stack.size(1)), int(logits_stack.size(2))]),
            "test_labels_used_for_threshold": False,
        }
        self.calibration_metadata = {
            **self.metadata,
            **self.fit_summary,
            "source": "login_uncertainty_router",
            "posterior_only": False,
            "risk_score_semantics": "official_login_uncertainty_score",
            "risk_score_formula": self.OFFICIAL_FORMULA,
            "not_a_new_classifier": True,
            "input_boundary": self._input_boundary_audit(
                edge_index=edge_index,
                edge_type=edge_type,
                node_repr=node_repr,
                **kwargs,
            ),
            "budgets": [float(item) for item in self.budgets],
        }
        return self

    def score(self, logits, probs, edge_index=None, edge_type=None, node_repr=None, **kwargs):
        uncertainty_score, _ = self._compute_uncertainty_score(
            kwargs.get("logits_list"),
            logits=logits,
            probs=probs,
        )
        self.last_risk_score = uncertainty_score.detach().cpu().numpy().astype(np.float32)
        self.last_official_pl_mask = self._official_topk_mask(uncertainty_score).detach().cpu().numpy().astype(bool)
        return self.last_risk_score

    def build_manifest(self, logits, probs, labels, val_idx, test_idx, edge_index=None, edge_type=None, node_repr=None, **kwargs):
        risk_score = self.score(
            logits,
            probs,
            edge_index=edge_index,
            edge_type=edge_type,
            node_repr=node_repr,
            logits_list=kwargs.get("logits_list"),
        )
        posterior = self._posterior_probabilities(logits=logits, probs=probs)
        labels_np = _labels_to_numpy(labels)
        preds = posterior.argmax(axis=1)
        wrong = (preds != labels_np).astype(np.int32)
        val_idx_np = _valid_index_array(val_idx, risk_score.shape[0])
        test_idx_np = _valid_index_array(test_idx, risk_score.shape[0])
        if self.val_threshold is None:
            budget = self.budgets[0] if self.budgets else RESIDUAL_RISK_PAPER_BUDGETS[0]
            k = max(int(val_idx_np.size * float(budget)), 1) if val_idx_np.size else 0
            self.val_threshold = float(np.sort(risk_score[val_idx_np])[-k]) if k else float("inf")
        official_pl_mask = (
            np.asarray(self.last_official_pl_mask, dtype=bool)
            if self.last_official_pl_mask is not None
            else self._official_topk_mask(torch.tensor(risk_score, dtype=torch.float32)).detach().cpu().numpy().astype(bool)
        )
        metadata = {
            **self.metadata,
            **dict(self.calibration_metadata),
            "source": "login_uncertainty_router",
            "fit_scope": "official_uncertainty_score_plus_auxiliary_validation_budget_analysis",
            "official_code_verified": bool(self.official_code_verified),
            "paper_aligned": True,
            "repo_locally_verified": False,
            "verified_scope": "node_selection_uncertainty_only",
            "screening_only": True,
            "diagnosis_or_action": False,
            "llm_call": False,
            "login_scope": "node_selection_uncertainty_only",
            "test_labels_used_for_threshold": False,
            "threshold_source": "auxiliary_validation_split_top_budget",
            "official_selection_contract": self.OFFICIAL_SELECTION_CONTRACT,
            "official_formula": self.OFFICIAL_FORMULA,
            "official_repo_path": self.OFFICIAL_REPO_PATH,
            "official_submodule_scope": "node_selection_uncertainty_only",
            "official_drop_out_list": list(self.OFFICIAL_DROPOUT_LIST),
            "official_pl_rate": float(self.OFFICIAL_PL_RATE),
            "full_LOGIN_loop": False,
            "prompt_llm": False,
            "feature_update": False,
            "edge_pruning": False,
            "auxiliary_budget_analysis": True,
            "not_a_new_classifier": True,
            "input_boundary": self._input_boundary_audit(
                edge_index=edge_index,
                edge_type=edge_type,
                node_repr=node_repr,
                **kwargs,
            ),
            "uncertainty_components": ["logits_variance_sum"],
        }
        validation_metrics = _safe_router_metrics(wrong[val_idx_np], risk_score[val_idx_np], budgets=self.budgets) if val_idx_np.size else {}
        test_metrics = _safe_router_metrics(wrong[test_idx_np], risk_score[test_idx_np], budgets=self.budgets) if test_idx_np.size else {}
        if val_idx_np.size:
            validation_metrics["budget_curve"] = router_budget_curve(wrong[val_idx_np], risk_score[val_idx_np], budgets=self.budgets)
        if test_idx_np.size:
            test_metrics["budget_curve"] = router_budget_curve(wrong[test_idx_np], risk_score[test_idx_np], budgets=self.budgets)
        hcw_mask = ((1.0 - posterior.max(axis=1)) < 0.1) & (wrong == 1)
        return {
            "risk_score": risk_score.astype(np.float32),
            "official_pl_mask": official_pl_mask.tolist(),
            "official_topk_indices": [int(item) for item in np.flatnonzero(official_pl_mask).tolist()],
            "thresholds": {
                "validation_risk_threshold": float(self.val_threshold),
                "threshold_source": "auxiliary_validation_budget_analysis",
                "official_pl_rate": float(self.OFFICIAL_PL_RATE),
                "official_topk_count": int(official_pl_mask.sum()),
            },
            "calibration_metadata": metadata,
            "hcw_mask": hcw_mask.tolist(),
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
        }

    def state_dict_payload(self):
        return {
            "class": "LOGINUncertaintyRouter",
            "budgets": [float(item) for item in self.budgets],
            "val_threshold": float(self.val_threshold if self.val_threshold is not None else float("inf")),
            "official_code_verified": bool(self.official_code_verified),
            "paper_aligned": True,
            "repo_locally_verified": False,
            "verified_scope": "node_selection_uncertainty_only",
            "official_repo_path": self.OFFICIAL_REPO_PATH,
            "official_formula": self.OFFICIAL_FORMULA,
            "official_selection_contract": self.OFFICIAL_SELECTION_CONTRACT,
            "official_drop_out_list": list(self.OFFICIAL_DROPOUT_LIST),
            "official_pl_rate": float(self.OFFICIAL_PL_RATE),
            "fit_summary": dict(self.fit_summary),
        }


class GraphConformalSetEstimator(BaseRiskEstimator):
    metadata = {
        "claim_role": "stage3_ego_quality_estimator",
        "stage2_role": "hard_node_quality_gate",
        "canonical_stage2": False,
        "scientific_gate": "calibration_split_labels_only",
        "paper_identity": "Graph Conformal Prediction-Set Ego Quality Estimator",
        "paper_identity_risk": "low_if_prediction_set_quality_gate_only",
        "promotion_rule": "stage3_candidate_after_ablation",
        "prediction_set_estimator": True,
        "diagnosis_or_action": False,
    }

    def __init__(self, alpha=0.20, graph_smoothing=0.10, budgets=RESIDUAL_RISK_PAPER_BUDGETS):
        super().__init__()
        self.alpha = float(alpha)
        self.graph_smoothing = float(graph_smoothing)
        self.budgets = parse_budget_list(budgets, default=RESIDUAL_RISK_PAPER_BUDGETS)
        self.threshold = None
        self.num_classes = None
        self.fit_summary = {}

    def _posterior(self, logits=None, probs=None):
        if logits is not None:
            logits_t = _to_tensor(logits).float()
            if logits_t.dim() != 2:
                raise ValueError("graph_conformal_set_estimator expects logits shaped [num_nodes, num_classes].")
            return F.softmax(logits_t, dim=1).detach().cpu().numpy()
        probs_t = _normalize_probs(probs)
        if probs_t.dim() != 2:
            raise ValueError("graph_conformal_set_estimator expects probabilities shaped [num_nodes, num_classes].")
        return probs_t.detach().cpu().numpy()

    def _graph_smoothed_posterior(self, posterior, edge_index=None):
        posterior = np.asarray(posterior, dtype=np.float64)
        if edge_index is None or self.graph_smoothing <= 0.0:
            return posterior
        edge_np = _to_numpy(edge_index).astype(np.int64)
        if edge_np.ndim != 2 or edge_np.shape[0] != 2:
            return posterior
        num_nodes = posterior.shape[0]
        src, dst = edge_np
        valid = (src >= 0) & (src < num_nodes) & (dst >= 0) & (dst < num_nodes)
        src = src[valid]
        dst = dst[valid]
        if src.size == 0:
            return posterior
        neighbor_sum = np.zeros_like(posterior, dtype=np.float64)
        degree = np.zeros(num_nodes, dtype=np.float64)
        np.add.at(neighbor_sum, dst, posterior[src])
        np.add.at(degree, dst, 1.0)
        has_neighbor = degree > 0
        neighbor_mean = posterior.copy()
        neighbor_mean[has_neighbor] = neighbor_sum[has_neighbor] / degree[has_neighbor, None]
        smoothed = (1.0 - self.graph_smoothing) * posterior + self.graph_smoothing * neighbor_mean
        row_sum = smoothed.sum(axis=1, keepdims=True)
        return smoothed / np.clip(row_sum, 1e-12, None)

    def _nonconformity(self, posterior, labels_np):
        rows = np.arange(labels_np.shape[0])
        return 1.0 - posterior[rows, labels_np]

    def _fit_threshold(self, scores):
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if scores.size == 0:
            return 1.0
        q = float(np.ceil((scores.size + 1) * (1.0 - self.alpha)) / scores.size)
        q = min(max(q, 0.0), 1.0)
        return float(np.quantile(scores, q, method="higher"))

    def _prediction_set_payload(self, posterior):
        conformity = 1.0 - np.asarray(posterior, dtype=np.float64)
        threshold = float(self.threshold if self.threshold is not None else 1.0)
        in_set = conformity <= threshold + 1e-12
        prediction_sets = []
        for row_idx, row in enumerate(in_set):
            labels = np.flatnonzero(row).astype(int).tolist()
            if not labels:
                labels = [int(np.argmax(posterior[row_idx]))]
            prediction_sets.append(labels)
        set_size = np.asarray([len(labels) for labels in prediction_sets], dtype=np.float32)
        best_margin = threshold - conformity.min(axis=1)
        max_extra = max(float(self.num_classes or posterior.shape[1]) - 1.0, 1.0)
        size_risk = (set_size - 1.0) / max_extra
        margin_risk = 1.0 - np.clip(best_margin / max(threshold, 1e-8), 0.0, 1.0)
        abstain_risk = np.clip(0.70 * size_risk + 0.30 * margin_risk, 0.0, 1.0).astype(np.float32)
        return {
            "prediction_sets": prediction_sets,
            "set_size": set_size.astype(np.int64).tolist(),
            "coverage_margin": best_margin.astype(np.float32),
            "abstain_risk": abstain_risk,
        }

    def fit(self, logits, probs, labels, train_idx, val_idx, edge_index=None, edge_type=None, node_repr=None, **kwargs):
        self.budgets = parse_budget_list(kwargs.get("budgets", self.budgets), default=RESIDUAL_RISK_PAPER_BUDGETS)
        posterior = self._graph_smoothed_posterior(self._posterior(logits=logits, probs=probs), edge_index=edge_index)
        labels_np = _labels_to_numpy(labels).astype(np.int64)
        val_idx_np = _valid_index_array(val_idx, labels_np.shape[0])
        if val_idx_np.size == 0:
            raise ValueError("graph_conformal_set_estimator requires a non-empty calibration/validation split.")
        self.num_classes = int(posterior.shape[1])
        calibration_scores = self._nonconformity(posterior[val_idx_np], labels_np[val_idx_np])
        self.threshold = self._fit_threshold(calibration_scores)
        self.fit_summary = {
            "source": "graph_conformal_set_estimator",
            "fit_scope": "validation_split_labels_only",
            "calibration_count": int(val_idx_np.size),
            "alpha": float(self.alpha),
            "threshold": float(self.threshold),
            "graph_smoothing": float(self.graph_smoothing),
            "graph_context_used": edge_index is not None,
            "embedding_context_used": node_repr is not None,
            "prediction_set_contract": "graph_conformal_prediction_set_v1",
            "test_labels_used_for_threshold": False,
            "literature_basis": ["DAPS/NAPS", "CF-GNN", "SNAPS"],
        }
        self.calibration_metadata = dict(self.fit_summary)
        self.val_threshold = None
        return self

    def score(self, logits, probs, edge_index=None, edge_type=None, node_repr=None, **kwargs):
        posterior = self._graph_smoothed_posterior(self._posterior(logits=logits, probs=probs), edge_index=edge_index)
        return self._prediction_set_payload(posterior)["abstain_risk"]

    def build_manifest(self, logits, probs, labels, val_idx, test_idx, edge_index=None, edge_type=None, node_repr=None, **kwargs):
        posterior = self._graph_smoothed_posterior(self._posterior(logits=logits, probs=probs), edge_index=edge_index)
        payload = self._prediction_set_payload(posterior)
        risk_score = payload["abstain_risk"].astype(np.float32)
        labels_np = _labels_to_numpy(labels)
        labeled_count = int(labels_np.shape[0])
        preds = posterior.argmax(axis=1)
        preds_labeled = preds[:labeled_count]
        wrong = (preds_labeled != labels_np).astype(np.int32)
        val_idx_np = _valid_index_array(val_idx, wrong.shape[0])
        test_idx_np = _valid_index_array(test_idx, wrong.shape[0])
        if self.val_threshold is None:
            budget = self.budgets[0] if self.budgets else RESIDUAL_RISK_PAPER_BUDGETS[0]
            k = max(int(len(val_idx_np) * float(budget)), 1) if val_idx_np.size else 0
            self.val_threshold = float(np.sort(risk_score[val_idx_np])[-k]) if k else float("inf")
        metadata = {
            **dict(self.calibration_metadata),
            "graph_context_used": edge_index is not None,
            "embedding_context_used": node_repr is not None,
            "threshold_source": "validation_conformal_nonconformity",
            "test_labels_used_for_threshold": False,
        }
        return {
            "risk_score": risk_score,
            "prediction_sets": payload["prediction_sets"],
            "set_size": payload["set_size"],
            "coverage_margin": payload["coverage_margin"].astype(np.float32),
            "abstain_risk": risk_score,
            "thresholds": {
                "validation_risk_threshold": float(self.val_threshold),
                "conformal_nonconformity_threshold": float(self.threshold if self.threshold is not None else 1.0),
            },
            "calibration_metadata": metadata,
            "hcw_mask": ((1.0 - posterior[:labeled_count].max(axis=1)) < 0.1) & (wrong == 1),
            "validation_metrics": _safe_router_metrics(wrong[val_idx_np], risk_score[val_idx_np], budgets=self.budgets) if val_idx_np.size else {},
            "test_metrics": _safe_router_metrics(wrong[test_idx_np], risk_score[test_idx_np], budgets=self.budgets) if test_idx_np.size else {},
        }

    def state_dict_payload(self):
        return {
            "alpha": float(self.alpha),
            "graph_smoothing": float(self.graph_smoothing),
            "threshold": float(self.threshold if self.threshold is not None else 1.0),
            "num_classes": self.num_classes,
            "fit_summary": self.fit_summary,
        }


class PostHocCalibratedRanker(GraphConformalSetEstimator):
    metadata = {
        "claim_role": "stage2_canonical_posthoc_ranker",
        "stage2_role": "posthoc_calibrated_score_rank",
        "canonical_stage2": True,
        "scientific_gate": "posterior_only_validation_calibration",
        "paper_identity": "Post-hoc Calibrated Score/Rank Router",
        "paper_identity_risk": "low_if_router_only_no_classifier_claim",
        "promotion_rule": "canonical_scalar_only",
        "prediction_set_estimator": True,
        "diagnosis_or_action": False,
        "input_boundary": "frozen_gnn_posterior_only",
    }

    def __init__(self, alpha=0.20, budgets=RESIDUAL_RISK_PAPER_BUDGETS):
        super().__init__(alpha=alpha, graph_smoothing=0.0, budgets=budgets)

    @staticmethod
    def _rank_desc(values):
        values = np.nan_to_num(np.asarray(values, dtype=np.float64).reshape(-1), nan=-1e12, posinf=1e12, neginf=-1e12)
        ranks = np.zeros(values.shape[0], dtype=np.int64)
        if values.size == 0:
            return ranks
        order = np.argsort(-values, kind="mergesort")
        ranks[order] = np.arange(1, values.size + 1, dtype=np.int64)
        return ranks

    @staticmethod
    def _nonconformity_gap(posterior):
        class_scores = 1.0 - np.asarray(posterior, dtype=np.float64)
        if class_scores.shape[1] <= 1:
            return np.ones(class_scores.shape[0], dtype=np.float64)
        sorted_scores = np.sort(class_scores, axis=1)
        return sorted_scores[:, 1] - sorted_scores[:, 0]

    def fit(self, logits, probs, labels, train_idx, val_idx, edge_index=None, edge_type=None, node_repr=None, **kwargs):
        self.budgets = parse_budget_list(kwargs.get("budgets", self.budgets), default=RESIDUAL_RISK_PAPER_BUDGETS)
        super().fit(
            logits=logits,
            probs=probs,
            labels=labels,
            train_idx=train_idx,
            val_idx=val_idx,
            edge_index=None,
            edge_type=None,
            node_repr=None,
            **kwargs,
        )
        self.fit_summary = {
            **dict(self.fit_summary),
            "source": "posthoc_calibrated_ranker",
            "fit_scope": "validation_split_labels_only_posterior_scalar",
            "calibration_role": "posthoc_calibrated_score_rank",
            "posterior_only": True,
            "scalar_only": True,
            "graph_smoothing": 0.0,
            "graph_context_used": False,
            "edge_type_used": False,
            "relation_channel_used": False,
            "direction_channel_used": False,
            "embedding_context_used": False,
            "node_repr_used": False,
            "prediction_set_contract": "posterior_only_conformal_prediction_set_v1",
            "risk_score_contract": "posthoc_prediction_set_size_margin_rank_v1",
            "rank_contract": "rank_1_is_highest_predicted_error_risk",
            "test_labels_used_for_threshold": False,
            "not_a_new_bot_classifier": True,
            "screening_only": True,
            "diagnosis_or_action": False,
            "does_not_claim_conformal_coverage_after_rewrite": True,
            "literature_basis": [
                "Selective Classification",
                "Conformal Risk Control",
                "Localized Conformal Prediction",
                "CF-GNN",
                "Graph CP benchmarks",
                "Graph reject option",
            ],
            "literature_boundary": (
                "Uses frozen GNN posterior calibration for hard-node ranking only; "
                "relation/direction/local graph channels remain ablations, not the canonical router."
            ),
        }
        self.calibration_metadata = dict(self.fit_summary)
        return self

    def score(self, logits, probs, edge_index=None, edge_type=None, node_repr=None, **kwargs):
        posterior = self._posterior(logits=logits, probs=probs)
        return self._prediction_set_payload(posterior)["abstain_risk"]

    def build_manifest(self, logits, probs, labels, val_idx, test_idx, edge_index=None, edge_type=None, node_repr=None, **kwargs):
        posterior = self._posterior(logits=logits, probs=probs)
        payload = self._prediction_set_payload(posterior)
        risk_score = payload["abstain_risk"].astype(np.float32)
        labels_np = _labels_to_numpy(labels).astype(np.int64)
        labeled_count = int(labels_np.shape[0])
        preds = posterior.argmax(axis=1)
        pred_label_score = posterior[np.arange(posterior.shape[0]), preds]
        preds_labeled = preds[:labeled_count]
        wrong = (preds_labeled != labels_np).astype(np.int32)
        val_idx_np = _valid_index_array(val_idx, wrong.shape[0])
        test_idx_np = _valid_index_array(test_idx, wrong.shape[0])
        budgets = parse_budget_list(kwargs.get("budgets", self.budgets), default=RESIDUAL_RISK_PAPER_BUDGETS)
        if self.val_threshold is None:
            budget = budgets[0] if budgets else RESIDUAL_RISK_PAPER_BUDGETS[0]
            k = max(int(len(val_idx_np) * float(budget)), 1) if val_idx_np.size else 0
            self.val_threshold = float(np.sort(risk_score[val_idx_np])[-k]) if k else float("inf")
        metadata = {
            **dict(self.calibration_metadata),
            "source": "posthoc_calibrated_ranker",
            "threshold_source": "validation_conformal_nonconformity_posterior_only",
            "posterior_only": True,
            "scalar_only": True,
            "graph_context_used": False,
            "edge_type_used": False,
            "relation_channel_used": False,
            "direction_channel_used": False,
            "embedding_context_used": False,
            "node_repr_used": False,
            "test_labels_used_for_threshold": False,
        }
        return {
            "risk_score": risk_score,
            "rank": self._rank_desc(risk_score).tolist(),
            "router_score": risk_score,
            "prediction_sets": payload["prediction_sets"],
            "set_size": payload["set_size"],
            "coverage_margin": payload["coverage_margin"].astype(np.float32),
            "nonconformity_gap": self._nonconformity_gap(posterior).astype(np.float32),
            "pred_label_score": pred_label_score.astype(np.float32),
            "abstain_risk": risk_score,
            "thresholds": {
                "validation_risk_threshold": float(self.val_threshold),
                "conformal_nonconformity_threshold": float(self.threshold if self.threshold is not None else 1.0),
            },
            "calibration_metadata": metadata,
            "hcw_mask": ((1.0 - posterior[:labeled_count].max(axis=1)) < 0.1) & (wrong == 1),
            "validation_metrics": _safe_router_metrics(wrong[val_idx_np], risk_score[val_idx_np], budgets=budgets) if val_idx_np.size else {},
            "test_metrics": _safe_router_metrics(wrong[test_idx_np], risk_score[test_idx_np], budgets=budgets) if test_idx_np.size else {},
        }

    def state_dict_payload(self):
        return {
            "alpha": float(self.alpha),
            "graph_smoothing": 0.0,
            "threshold": float(self.threshold if self.threshold is not None else 1.0),
            "num_classes": self.num_classes,
            "fit_summary": self.fit_summary,
        }


def _relation_aware_scalar_risk_features(
    base_risk,
    edge_index=None,
    edge_type=None,
    node_repr=None,
    *,
    similarity_temperature=4.0,
    shrinkage_tau=3.0,
    high_quantile=0.80,
    low_quantile=0.35,
):
    risk = np.asarray(base_risk, dtype=np.float32).reshape(-1)
    num_nodes = int(risk.shape[0])
    zero = np.zeros(num_nodes, dtype=np.float32)
    if edge_index is None:
        return {
            "mean_in_rel0_risk": zero.copy(),
            "mean_in_rel1_risk": zero.copy(),
            "mean_out_rel0_risk": zero.copy(),
            "mean_out_rel1_risk": zero.copy(),
            "mean_in_risk": zero.copy(),
            "mean_out_risk": zero.copy(),
            "neighbor_risk_std": zero.copy(),
            "neighbor_risk_max": zero.copy(),
            "risk_gap_in_out": zero.copy(),
            "relation_gap_in": zero.copy(),
            "relation_gap_out": zero.copy(),
            "channel_peak_risk": zero.copy(),
            "channel_peak_std": zero.copy(),
            "high_risk_mass_peak": zero.copy(),
            "low_risk_mass_peak": zero.copy(),
            "channel_conflict_peak": zero.copy(),
            "similarity_weight_mean": zero.copy(),
            "effective_neighbor_count": zero.copy(),
        }

    edge_np = _to_numpy(edge_index).astype(np.int64)
    if edge_np.ndim != 2 or edge_np.shape[0] != 2:
        raise ValueError("calibrated_local_risk_router expects edge_index shaped [2, num_edges].")
    src, dst = edge_np
    valid = (src >= 0) & (src < num_nodes) & (dst >= 0) & (dst < num_nodes)
    src = src[valid]
    dst = dst[valid]

    if edge_type is None:
        rel = np.zeros(src.shape[0], dtype=np.int64)
    else:
        rel_all = _to_numpy(edge_type).astype(np.int64).reshape(-1)
        rel = rel_all[valid]

    rel = np.clip(rel, 0, 1)
    src_risk = risk[src].astype(np.float64)
    dst_risk = risk[dst].astype(np.float64)

    if node_repr is None:
        edge_weight = np.ones(src.shape[0], dtype=np.float64)
    else:
        repr_np = _to_numpy(node_repr).astype(np.float32)
        if repr_np.ndim != 2 or repr_np.shape[0] != num_nodes:
            raise ValueError(
                "calibrated_local_risk_router expects node_repr shaped [num_nodes, hidden_dim] "
                "and aligned with the graph-wide node count."
            )
        src_repr = repr_np[src]
        dst_repr = repr_np[dst]
        src_norm = np.linalg.norm(src_repr, axis=1)
        dst_norm = np.linalg.norm(dst_repr, axis=1)
        cosine = np.sum(src_repr * dst_repr, axis=1) / np.clip(src_norm * dst_norm, 1e-8, None)
        cosine = np.clip(cosine, -1.0, 1.0)
        localized_similarity = 0.5 * (cosine + 1.0)
        edge_weight = np.exp(float(similarity_temperature) * localized_similarity).astype(np.float64)

    channel_idx_in = rel
    channel_idx_out = rel + 2
    high_threshold = float(np.quantile(risk.astype(np.float64), float(high_quantile))) if risk.size else 0.0
    low_threshold = float(np.quantile(risk.astype(np.float64), float(low_quantile))) if risk.size else 0.0

    channel_weight = np.zeros((4, num_nodes), dtype=np.float64)
    channel_sum = np.zeros((4, num_nodes), dtype=np.float64)
    channel_sq_sum = np.zeros((4, num_nodes), dtype=np.float64)
    channel_count = np.zeros((4, num_nodes), dtype=np.float64)
    channel_high_weight = np.zeros((4, num_nodes), dtype=np.float64)
    channel_low_weight = np.zeros((4, num_nodes), dtype=np.float64)
    channel_max_risk = np.zeros((4, num_nodes), dtype=np.float64)

    np.add.at(channel_weight, (channel_idx_in, dst), edge_weight)
    np.add.at(channel_sum, (channel_idx_in, dst), edge_weight * src_risk)
    np.add.at(channel_sq_sum, (channel_idx_in, dst), edge_weight * (src_risk**2))
    np.add.at(channel_count, (channel_idx_in, dst), 1.0)
    np.add.at(channel_high_weight, (channel_idx_in, dst), edge_weight * (src_risk >= high_threshold).astype(np.float64))
    np.add.at(channel_low_weight, (channel_idx_in, dst), edge_weight * (src_risk <= low_threshold).astype(np.float64))
    np.maximum.at(channel_max_risk, (channel_idx_in, dst), src_risk)

    np.add.at(channel_weight, (channel_idx_out, src), edge_weight)
    np.add.at(channel_sum, (channel_idx_out, src), edge_weight * dst_risk)
    np.add.at(channel_sq_sum, (channel_idx_out, src), edge_weight * (dst_risk**2))
    np.add.at(channel_count, (channel_idx_out, src), 1.0)
    np.add.at(channel_high_weight, (channel_idx_out, src), edge_weight * (dst_risk >= high_threshold).astype(np.float64))
    np.add.at(channel_low_weight, (channel_idx_out, src), edge_weight * (dst_risk <= low_threshold).astype(np.float64))
    np.maximum.at(channel_max_risk, (channel_idx_out, src), dst_risk)

    total_weight = channel_weight.sum(axis=0)
    total_count = channel_count.sum(axis=0)
    total_sum = channel_sum.sum(axis=0)
    total_sq_sum = channel_sq_sum.sum(axis=0)

    base_prior = risk.astype(np.float64)[None, :]
    channel_mean = np.repeat(base_prior, channel_sum.shape[0], axis=0)
    weighted_mask = channel_weight > 0
    channel_mean[weighted_mask] = channel_sum[weighted_mask] / channel_weight[weighted_mask]
    shrink = channel_count / (channel_count + float(shrinkage_tau))
    channel_mean = shrink * channel_mean + (1.0 - shrink) * base_prior

    channel_var = np.zeros_like(channel_sum, dtype=np.float64)
    channel_var[weighted_mask] = np.clip(
        channel_sq_sum[weighted_mask] / channel_weight[weighted_mask] - (channel_sum[weighted_mask] / channel_weight[weighted_mask]) ** 2,
        0.0,
        None,
    )
    channel_std = np.sqrt(channel_var)
    channel_high_mass = np.zeros_like(channel_sum, dtype=np.float64)
    channel_low_mass = np.zeros_like(channel_sum, dtype=np.float64)
    channel_high_mass[weighted_mask] = channel_high_weight[weighted_mask] / channel_weight[weighted_mask]
    channel_low_mass[weighted_mask] = channel_low_weight[weighted_mask] / channel_weight[weighted_mask]

    mean_in_rel0 = channel_mean[0].astype(np.float32)
    mean_in_rel1 = channel_mean[1].astype(np.float32)
    mean_out_rel0 = channel_mean[2].astype(np.float32)
    mean_out_rel1 = channel_mean[3].astype(np.float32)

    mean_in = (0.5 * (channel_mean[0] + channel_mean[1])).astype(np.float32)
    mean_out = (0.5 * (channel_mean[2] + channel_mean[3])).astype(np.float32)
    neighbor_mean = np.zeros(num_nodes, dtype=np.float64)
    valid_neighbors = total_weight > 0
    neighbor_mean[valid_neighbors] = total_sum[valid_neighbors] / total_weight[valid_neighbors]
    neighbor_var = np.zeros(num_nodes, dtype=np.float64)
    neighbor_var[valid_neighbors] = np.clip(
        total_sq_sum[valid_neighbors] / total_weight[valid_neighbors] - neighbor_mean[valid_neighbors] ** 2,
        0.0,
        None,
    )
    neighbor_std = np.sqrt(neighbor_var).astype(np.float32)
    risk_gap = np.abs(mean_in - mean_out).astype(np.float32)
    relation_gap_in = np.abs(channel_mean[0] - channel_mean[1]).astype(np.float32)
    relation_gap_out = np.abs(channel_mean[2] - channel_mean[3]).astype(np.float32)
    channel_peak_risk = np.max(channel_mean, axis=0).astype(np.float32)
    channel_peak_std = np.max(channel_std, axis=0).astype(np.float32)
    high_risk_mass_peak = np.max(channel_high_mass, axis=0).astype(np.float32)
    low_risk_mass_peak = np.max(channel_low_mass, axis=0).astype(np.float32)
    channel_conflict_peak = np.max(np.minimum(channel_high_mass, channel_low_mass), axis=0).astype(np.float32)
    similarity_weight_mean = np.zeros(num_nodes, dtype=np.float32)
    similarity_weight_mean[valid_neighbors] = (total_weight[valid_neighbors] / np.clip(total_count[valid_neighbors], 1.0, None)).astype(np.float32)
    effective_neighbor_count = total_count.astype(np.float32)
    neighbor_risk_max = np.max(channel_max_risk, axis=0).astype(np.float32)

    return {
        "mean_in_rel0_risk": mean_in_rel0,
        "mean_in_rel1_risk": mean_in_rel1,
        "mean_out_rel0_risk": mean_out_rel0,
        "mean_out_rel1_risk": mean_out_rel1,
        "mean_in_risk": mean_in,
        "mean_out_risk": mean_out,
        "neighbor_risk_std": neighbor_std,
        "neighbor_risk_max": neighbor_risk_max,
        "risk_gap_in_out": risk_gap,
        "relation_gap_in": relation_gap_in,
        "relation_gap_out": relation_gap_out,
        "channel_peak_risk": channel_peak_risk,
        "channel_peak_std": channel_peak_std,
        "high_risk_mass_peak": high_risk_mass_peak,
        "low_risk_mass_peak": low_risk_mass_peak,
        "channel_conflict_peak": channel_conflict_peak,
        "similarity_weight_mean": similarity_weight_mean,
        "effective_neighbor_count": effective_neighbor_count,
    }


def _knn_candidate_rows(edge_index, num_nodes, labeled_count, candidate_scope):
    scope = str(candidate_scope or "labeled_full").strip().lower()
    labeled_limit = max(0, min(int(num_nodes), int(labeled_count)))
    if scope == "labeled_full":
        return None
    if scope not in {"labeled_relation_1hop", "undirected_relation_1hop"}:
        raise ValueError(
            "--conformal_knn_candidate_scope must be one of "
            "{labeled_full, hyperscan_full, labeled_relation_1hop, undirected_relation_1hop}."
        )
    if edge_index is None:
        return [[] for _ in range(int(num_nodes))]
    edge_np = _to_numpy(edge_index).astype(np.int64)
    if edge_np.ndim != 2 or edge_np.shape[0] != 2:
        raise ValueError("conformal_knn_risk_router expects edge_index shaped [2, num_edges].")
    rows = [set() for _ in range(int(num_nodes))]
    src, dst = edge_np
    valid = (src >= 0) & (src < int(num_nodes)) & (dst >= 0) & (dst < int(num_nodes))
    for u_raw, v_raw in zip(src[valid].tolist(), dst[valid].tolist()):
        u = int(u_raw)
        v = int(v_raw)
        if u == v:
            continue
        if scope == "labeled_relation_1hop":
            if v < labeled_limit:
                rows[u].add(v)
            if u < labeled_limit:
                rows[v].add(u)
        else:
            rows[u].add(v)
            rows[v].add(u)
    return [sorted(item) for item in rows]


def _weighted_quantile(values, weights, quantile):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.size == 0 or weights.size != values.size:
        return 0.0
    weights = np.clip(weights, 0.0, None)
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        return float(np.quantile(values, float(quantile)))
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights) / weight_sum
    index = int(np.searchsorted(cumulative, float(quantile), side="left"))
    index = min(max(index, 0), int(sorted_values.size) - 1)
    return float(sorted_values[index])


def _conformal_quantile(scores, alpha):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        return 1.0
    q = float(np.ceil((scores.size + 1) * (1.0 - float(alpha))) / scores.size)
    q = min(max(q, 0.0), 1.0)
    return float(np.quantile(scores, q, method="higher"))


def _prediction_set_risk_from_thresholds(posterior, thresholds, num_classes=None):
    posterior_np = np.asarray(posterior, dtype=np.float64)
    if posterior_np.ndim != 2 or posterior_np.shape[0] == 0:
        size = int(posterior_np.shape[0]) if posterior_np.ndim else 0
        zero = np.zeros(size, dtype=np.float32)
        return zero.copy(), zero.astype(np.int64), zero.copy(), zero.copy()
    threshold_np = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    if threshold_np.shape[0] != posterior_np.shape[0]:
        threshold_np = np.full(posterior_np.shape[0], 1.0, dtype=np.float64)
    threshold_np = np.clip(threshold_np, 1e-8, 1.0)
    conformity = 1.0 - posterior_np
    in_set = conformity <= threshold_np[:, None] + 1e-12
    set_size = in_set.sum(axis=1).astype(np.int64)
    set_size = np.maximum(set_size, 1)
    best_margin = threshold_np - conformity.min(axis=1)
    max_extra = max(float(num_classes or posterior_np.shape[1]) - 1.0, 1.0)
    size_risk = (set_size.astype(np.float64) - 1.0) / max_extra
    margin_risk = 1.0 - np.clip(best_margin / threshold_np, 0.0, 1.0)
    abstain_risk = np.clip(0.70 * size_risk + 0.30 * margin_risk, 0.0, 1.0)
    return (
        abstain_risk.astype(np.float32),
        set_size.astype(np.int64),
        best_margin.astype(np.float32),
        margin_risk.astype(np.float32),
    )


def _conformal_knn_scalar_risk_features(
    base_risk,
    edge_index=None,
    edge_type=None,
    node_repr=None,
    *,
    posterior=None,
    labeled_count=None,
    knn_k=8,
    candidate_scope="labeled_full",
    shrinkage_tau=3.0,
    ncp_lambda=1.0,
    neighbor_mode="standard",
    similarity_threshold=-1.0,
    min_support=1,
    adaptive_max_k=0,
    hubness_correction="none",
    local_calibration_labels=None,
    local_calibration_idx=None,
    local_calibration_alpha=0.20,
    global_conformal_threshold=None,
    local_calibration_scope="independent_knn",
):
    del edge_type
    risk = np.asarray(base_risk, dtype=np.float32).reshape(-1)
    num_nodes = int(risk.shape[0])
    zero = np.zeros(num_nodes, dtype=np.float32)
    if node_repr is None or num_nodes == 0:
        return {
            "knn_mean_risk": risk.copy(),
            "knn_max_risk": risk.copy(),
            "knn_risk_std": zero.copy(),
            "knn_prediction_disagreement": zero.copy(),
            "knn_high_risk_mass": zero.copy(),
            "knn_safe_support_mass": zero.copy(),
            "knn_similarity_mean": zero.copy(),
            "knn_similarity_min": zero.copy(),
            "knn_similarity_gap": zero.copy(),
            "knn_effective_neighbor_count": zero.copy(),
            "knn_support_neighbor_count": zero.copy(),
            "knn_hyperedge_member_count": zero.copy(),
            "ncp_weighted_mean_risk": risk.copy(),
            "ncp_shrunk_weighted_mean_risk": risk.copy(),
            "ncp_weighted_risk_std": zero.copy(),
            "ncp_weighted_risk_q80": risk.copy(),
            "ncp_weighted_prediction_disagreement": zero.copy(),
            "ncp_weighted_high_risk_mass": zero.copy(),
            "ncp_weighted_safe_support_mass": zero.copy(),
            "ncp_effective_sample_size": zero.copy(),
            "ncp_weight_sum": zero.copy(),
            "ncp_weight_max": zero.copy(),
            "ncp_local_abstain_risk": risk.copy(),
            "ncp_local_threshold": zero.copy(),
            "ncp_local_threshold_delta": zero.copy(),
            "ncp_local_set_size": zero.copy(),
            "ncp_local_coverage_margin": zero.copy(),
            "ncp_local_margin_risk": risk.copy(),
            "ncp_local_calibration_neighbor_count": zero.copy(),
            "ncp_local_effective_calibration_sample_size": zero.copy(),
            "ncp_local_fallback_to_global": np.ones(num_nodes, dtype=np.float32),
            "knn_has_candidates": zero.copy(),
        }

    repr_np = _to_numpy(node_repr).astype(np.float32)
    if repr_np.ndim != 2 or repr_np.shape[0] != num_nodes:
        raise ValueError(
            "conformal_knn_risk_router expects node_repr shaped [num_nodes, hidden_dim] "
            "and aligned with the graph-wide node count."
        )
    repr_np = np.nan_to_num(repr_np, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.linalg.norm(repr_np, axis=1, keepdims=True)
    repr_np = repr_np / np.clip(norm, 1e-12, None)

    labeled_limit = int(labeled_count) if labeled_count is not None else num_nodes
    labeled_limit = max(0, min(num_nodes, labeled_limit))
    center_limit = labeled_limit if labeled_count is not None else num_nodes
    center_limit = max(0, min(num_nodes, int(center_limit)))
    k = max(int(knn_k), 1)
    adaptive_k = int(adaptive_max_k or 0)
    support_k = max(k, adaptive_k) if adaptive_k > 0 else k
    min_support_count = max(int(min_support or 0), 0)
    sim_threshold = float(similarity_threshold)
    if not math.isfinite(sim_threshold):
        sim_threshold = -1.0
    neighbor_mode_value = str(neighbor_mode or "standard").strip().lower()
    if neighbor_mode_value not in {"standard", "mutual", "threshold", "adaptive", "mutual_adaptive"}:
        raise ValueError(
            "--conformal_knn_neighbor_mode must be one of "
            "{standard, mutual, threshold, adaptive, mutual_adaptive}."
        )
    hubness_correction_value = str(hubness_correction or "none").strip().lower()
    if hubness_correction_value not in {"none", "degree"}:
        raise ValueError("--conformal_knn_hubness_correction must be one of {none, degree}.")
    local_calibration_scope_value = str(local_calibration_scope or "independent_knn").strip().lower()
    if local_calibration_scope_value not in {"independent_knn", "same_hyperedge"}:
        raise ValueError(
            "--conformal_knn_local_calibration_scope must be one of "
            "{independent_knn, same_hyperedge}."
        )
    mutual_required = neighbor_mode_value in {"mutual", "mutual_adaptive"}
    threshold_active = bool(sim_threshold > -1.0) and neighbor_mode_value in {
        "threshold",
        "adaptive",
        "mutual_adaptive",
    }
    scope = str(candidate_scope or "labeled_full").strip().lower()
    posterior_np = np.asarray(posterior, dtype=np.float32) if posterior is not None else None
    preds = posterior_np.argmax(axis=1).astype(np.int64) if posterior_np is not None and posterior_np.ndim == 2 else None

    mean_risk = risk.copy().astype(np.float64)
    max_risk = risk.copy().astype(np.float64)
    std_risk = np.zeros(num_nodes, dtype=np.float64)
    pred_disagreement = np.zeros(num_nodes, dtype=np.float64)
    high_risk_mass = np.zeros(num_nodes, dtype=np.float64)
    safe_support_mass = np.zeros(num_nodes, dtype=np.float64)
    similarity_mean = np.zeros(num_nodes, dtype=np.float64)
    similarity_min = np.zeros(num_nodes, dtype=np.float64)
    neighbor_count = np.zeros(num_nodes, dtype=np.float64)
    support_neighbor_count = np.zeros(num_nodes, dtype=np.float64)
    hyperedge_member_count = np.zeros(num_nodes, dtype=np.float64)
    ncp_weighted_mean_risk = risk.copy().astype(np.float64)
    ncp_shrunk_weighted_mean_risk = risk.copy().astype(np.float64)
    ncp_weighted_risk_std = np.zeros(num_nodes, dtype=np.float64)
    ncp_weighted_risk_q80 = risk.copy().astype(np.float64)
    ncp_weighted_prediction_disagreement = np.zeros(num_nodes, dtype=np.float64)
    ncp_weighted_high_risk_mass = np.zeros(num_nodes, dtype=np.float64)
    ncp_weighted_safe_support_mass = np.zeros(num_nodes, dtype=np.float64)
    ncp_effective_sample_size = np.zeros(num_nodes, dtype=np.float64)
    ncp_weight_sum = np.zeros(num_nodes, dtype=np.float64)
    ncp_weight_max = np.zeros(num_nodes, dtype=np.float64)
    knn_hubness_mean = np.zeros(num_nodes, dtype=np.float64)
    knn_hubness_max = np.zeros(num_nodes, dtype=np.float64)
    knn_filter_fallback = np.zeros(num_nodes, dtype=np.float64)
    hubness_counts = np.zeros(num_nodes, dtype=np.float64)
    ncp_local_threshold = np.ones(num_nodes, dtype=np.float64)
    ncp_local_threshold_delta = np.zeros(num_nodes, dtype=np.float64)
    ncp_local_calibration_neighbor_count = np.zeros(num_nodes, dtype=np.float64)
    ncp_local_effective_calibration_sample_size = np.zeros(num_nodes, dtype=np.float64)
    ncp_local_fallback_to_global = np.ones(num_nodes, dtype=np.float64)
    same_hyperedge_target_nonconformity = risk.copy().astype(np.float64)
    same_hyperedge_global_tail_risk = risk.copy().astype(np.float64)
    same_hyperedge_selected_tail_pvalue = np.ones(num_nodes, dtype=np.float64)
    same_hyperedge_selected_tail_risk = risk.copy().astype(np.float64)
    same_hyperedge_selected_effective_sample_size = np.zeros(num_nodes, dtype=np.float64)
    same_hyperedge_selected_fallback_to_global = np.ones(num_nodes, dtype=np.float64)
    same_hyperedge_calibration_global_tail_risk = risk.copy().astype(np.float64)
    same_hyperedge_calibration_tail_pvalue = np.ones(num_nodes, dtype=np.float64)
    same_hyperedge_calibration_tail_risk = risk.copy().astype(np.float64)
    same_hyperedge_calibration_effective_sample_size = np.zeros(num_nodes, dtype=np.float64)
    same_hyperedge_calibration_fallback_to_global = np.ones(num_nodes, dtype=np.float64)
    same_hyperedge_calibration_shrink_weight = np.zeros(num_nodes, dtype=np.float64)
    same_hyperedge_calibration_shrunk_tail_risk = risk.copy().astype(np.float64)
    support_candidate_rows = [[] for _ in range(center_limit)]
    support_similarity_rows = [[] for _ in range(center_limit)]

    high_threshold = float(np.quantile(risk[:labeled_limit].astype(np.float64), 0.80)) if labeled_limit else 1.0
    low_threshold = float(np.quantile(risk[:labeled_limit].astype(np.float64), 0.35)) if labeled_limit else 0.0
    ncp_lambda_value = max(float(ncp_lambda), 1e-6)
    labels_np = None
    calibration_idx_np = np.empty(0, dtype=np.int64)
    calibration_nonconformity = np.empty(0, dtype=np.float64)
    calibration_lookup = np.zeros(num_nodes, dtype=bool)
    if (
        posterior_np is not None
        and posterior_np.ndim == 2
        and local_calibration_labels is not None
        and labeled_limit > 0
    ):
        labels_np = _labels_to_numpy(local_calibration_labels).astype(np.int64).reshape(-1)
        labels_limit = min(int(labels_np.shape[0]), int(posterior_np.shape[0]), int(labeled_limit))
        labels_np = labels_np[:labels_limit]
        raw_idx = local_calibration_idx if local_calibration_idx is not None else np.arange(labels_limit)
        calibration_idx_np = _valid_index_array(raw_idx, labels_limit)
        if calibration_idx_np.size:
            calibration_idx_np = np.unique(np.sort(calibration_idx_np.astype(np.int64, copy=False)))
            valid_label = (labels_np[calibration_idx_np] >= 0) & (labels_np[calibration_idx_np] < int(posterior_np.shape[1]))
            calibration_idx_np = calibration_idx_np[valid_label]
        if calibration_idx_np.size:
            calibration_nonconformity = (
                1.0 - posterior_np[calibration_idx_np, labels_np[calibration_idx_np]]
            ).astype(np.float64)
            calibration_lookup[calibration_idx_np] = True
    if global_conformal_threshold is None:
        global_threshold = _conformal_quantile(calibration_nonconformity, local_calibration_alpha)
    else:
        global_threshold = float(global_conformal_threshold)
    if not math.isfinite(global_threshold):
        global_threshold = 1.0
    global_threshold = float(np.clip(global_threshold, 1e-8, 1.0))
    ncp_local_threshold.fill(global_threshold)
    local_calibration_active = bool(calibration_idx_np.size > 0 and posterior_np is not None and posterior_np.ndim == 2)
    same_hyperedge_tail_active = bool(
        local_calibration_scope_value == "same_hyperedge"
        and posterior_np is not None
        and posterior_np.ndim == 2
    )
    same_hyperedge_global_reference_count = 0
    if same_hyperedge_tail_active:
        same_hyperedge_target_nonconformity = (1.0 - posterior_np.max(axis=1)).astype(np.float64)
        same_hyperedge_global_reference_count = int(labeled_limit if labeled_limit > 0 else num_nodes)
        global_scores = same_hyperedge_target_nonconformity[:same_hyperedge_global_reference_count]
        if global_scores.size and center_limit > 0:
            sorted_global_scores = np.sort(global_scores, kind="stable")
            center_scores = same_hyperedge_target_nonconformity[:center_limit]
            ge_count = int(sorted_global_scores.size) - np.searchsorted(sorted_global_scores, center_scores, side="left")
            tail_pvalue = (ge_count.astype(np.float64) + 1.0) / (float(sorted_global_scores.size) + 1.0)
            tail_risk = 1.0 - tail_pvalue
            same_hyperedge_selected_tail_pvalue[:center_limit] = np.clip(tail_pvalue, 1e-8, 1.0)
            same_hyperedge_global_tail_risk[:center_limit] = np.clip(tail_risk, 0.0, 1.0)
            same_hyperedge_selected_tail_risk[:center_limit] = same_hyperedge_global_tail_risk[:center_limit]
        if calibration_nonconformity.size and center_limit > 0:
            sorted_calibration_scores = np.sort(calibration_nonconformity, kind="stable")
            center_scores = same_hyperedge_target_nonconformity[:center_limit]
            ge_count = int(sorted_calibration_scores.size) - np.searchsorted(sorted_calibration_scores, center_scores, side="left")
            cal_tail_pvalue = (ge_count.astype(np.float64) + 1.0) / (float(sorted_calibration_scores.size) + 1.0)
            cal_tail_risk = 1.0 - cal_tail_pvalue
            same_hyperedge_calibration_tail_pvalue[:center_limit] = np.clip(cal_tail_pvalue, 1e-8, 1.0)
            same_hyperedge_calibration_global_tail_risk[:center_limit] = np.clip(cal_tail_risk, 0.0, 1.0)
            same_hyperedge_calibration_tail_risk[:center_limit] = same_hyperedge_calibration_global_tail_risk[:center_limit]
            same_hyperedge_calibration_shrunk_tail_risk[:center_limit] = same_hyperedge_calibration_global_tail_risk[:center_limit]

    def _torch_exact_feature_topk(query_repr, candidate_repr, query_k, backend_label="feature_pool"):
        if query_repr.shape[0] == 0 or candidate_repr.shape[0] == 0 or query_k <= 0:
            return (
                np.empty((int(query_repr.shape[0]), 0), dtype=np.int64),
                np.empty((int(query_repr.shape[0]), 0), dtype=np.float32),
                f"torch_exact_empty_{backend_label}",
            )
        use_cuda = bool(torch.cuda.is_available())
        device = torch.device("cuda" if use_cuda else "cpu")
        batch_size = 512 if use_cuda else 256
        candidate_t = torch.from_numpy(candidate_repr.astype(np.float32, copy=False)).to(device=device)
        neighbor_chunks = []
        similarity_chunks = []
        with torch.no_grad():
            for start in range(0, int(query_repr.shape[0]), batch_size):
                end = min(start + batch_size, int(query_repr.shape[0]))
                query_t = torch.from_numpy(query_repr[start:end].astype(np.float32, copy=False)).to(device=device)
                scores = query_t @ candidate_t.t()
                values, indices = torch.topk(scores, k=int(query_k), largest=True, dim=1)
                neighbor_chunks.append(indices.cpu().numpy().astype(np.int64, copy=False))
                similarity_chunks.append(values.clamp(-1.0, 1.0).cpu().numpy().astype(np.float32, copy=False))
                del query_t, scores, values, indices
        if use_cuda:
            torch.cuda.empty_cache()
        return (
            np.concatenate(neighbor_chunks, axis=0),
            np.concatenate(similarity_chunks, axis=0),
            f"torch_exact_cuda_{backend_label}" if use_cuda else f"torch_exact_cpu_{backend_label}",
        )

    if scope in {"labeled_full", "hyperscan_full"}:
        candidate_limit = num_nodes if scope == "hyperscan_full" else labeled_limit
        if candidate_limit == 0:
            backend = "empty_labeled_candidate_pool"
            candidate_rows = []
            neighbors = np.empty((num_nodes, 0), dtype=np.int64)
            similarities = np.empty((num_nodes, 0), dtype=np.float32)
        else:
            query_repr = repr_np[:center_limit]
            query_k = min(support_k, candidate_limit) if scope == "hyperscan_full" else min(support_k + 1, candidate_limit)
            if scope == "hyperscan_full" or bool(torch.cuda.is_available()) or int(repr_np.shape[1]) > 128:
                neighbors, similarities, backend = _torch_exact_feature_topk(
                    query_repr,
                    repr_np[:candidate_limit],
                    query_k,
                    backend_label=scope,
                )
            else:
                try:
                    from scipy.spatial import cKDTree

                    tree = cKDTree(repr_np[:candidate_limit])
                    distances, neighbors = tree.query(query_repr, k=query_k)
                    backend = "scipy_ckdtree_labeled_full"
                except Exception as exc:
                    if candidate_limit > 4096 or num_nodes > 4096:
                        raise RuntimeError(
                            "conformal_knn_risk_router requires SciPy cKDTree for labeled_full KNN "
                            "on graphs larger than 4096 nodes."
                        ) from exc
                    distance = torch.cdist(torch.from_numpy(query_repr), torch.from_numpy(repr_np[:candidate_limit]), p=2)
                    topk = torch.topk(distance, k=query_k, largest=False, dim=1)
                    distances = topk.values.numpy()
                    neighbors = topk.indices.numpy()
                    backend = "torch_cdist_labeled_full"
                neighbors = np.asarray(neighbors, dtype=np.int64)
                distances = np.asarray(distances, dtype=np.float32)
                if neighbors.ndim == 1:
                    neighbors = neighbors.reshape(-1, 1)
                    distances = distances.reshape(-1, 1)
                similarities = np.clip(1.0 - 0.5 * (distances.astype(np.float64) ** 2), -1.0, 1.0).astype(np.float32)
            candidate_rows = None
    else:
        backend = "relation_local_cosine_topk"
        candidate_rows = _knn_candidate_rows(edge_index, num_nodes, labeled_limit, scope)
        neighbors = None
        similarities = None

    mutual_neighbor_sets = None
    if scope in {"labeled_full", "hyperscan_full"} and neighbors is not None and int(neighbors.size) > 0:
        for center in range(min(center_limit, int(neighbors.shape[0]))):
            row = neighbors[center]
            valid = (row >= 0) & (row < num_nodes) & (row != int(center))
            if np.any(valid):
                hubness_counts[row[valid].astype(np.int64)] += 1.0
        if mutual_required:
            mutual_neighbor_sets = []
            for center in range(min(center_limit, int(neighbors.shape[0]))):
                row = neighbors[center]
                valid = (row >= 0) & (row < num_nodes) & (row != int(center))
                mutual_neighbor_sets.append(set(int(item) for item in row[valid].tolist()))

    def _filter_support(center, selected, sims):
        raw_count = int(selected.size)
        if selected.size == 0:
            return selected.astype(np.int64), sims.astype(np.float32)
        keep = np.ones(int(selected.size), dtype=bool)
        if threshold_active:
            keep &= sims.astype(np.float64) >= sim_threshold
        if mutual_required and mutual_neighbor_sets is not None:
            mutual_keep = np.zeros(int(selected.size), dtype=bool)
            for pos, neighbor in enumerate(selected.astype(np.int64, copy=False).tolist()):
                if 0 <= int(neighbor) < len(mutual_neighbor_sets):
                    mutual_keep[pos] = int(center) in mutual_neighbor_sets[int(neighbor)]
            keep &= mutual_keep
        selected = selected[keep].astype(np.int64, copy=False)
        sims = sims[keep].astype(np.float32, copy=False)
        if selected.size > support_k:
            selected = selected[:support_k]
            sims = sims[:support_k]
        if int(selected.size) < min_support_count:
            if raw_count > 0:
                knn_filter_fallback[center] = 1.0
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        return selected, sims

    def _apply_selected(center, selected, sims, member_count=None):
        support_count = float(selected.size)
        support_neighbor_count[center] = support_count
        hyperedge_member_count[center] = float(member_count) if member_count is not None else support_count
        if 0 <= int(center) < int(center_limit):
            support_candidate_rows[int(center)] = [int(item) for item in selected.astype(np.int64).tolist()]
            support_similarity_rows[int(center)] = [float(item) for item in sims.astype(np.float32).tolist()]
        if selected.size == 0:
            return
        selected_hubness = hubness_counts[selected] if hubness_counts.size else np.zeros(selected.size, dtype=np.float64)
        if selected_hubness.size:
            knn_hubness_mean[center] = float(selected_hubness.mean())
            knn_hubness_max[center] = float(selected_hubness.max())
        values = risk[selected].astype(np.float64)
        count = support_count
        shrink = count / (count + float(shrinkage_tau))
        local_mean = float(values.mean())
        mean_risk[center] = shrink * local_mean + (1.0 - shrink) * float(risk[center])
        max_risk[center] = float(values.max())
        std_risk[center] = float(values.std())
        high_risk_mass[center] = float((values >= high_threshold).mean())
        safe_support_mass[center] = float((values <= low_threshold).mean())
        similarity_mean[center] = float(np.mean(sims)) if sims.size else 0.0
        similarity_min[center] = float(np.min(sims)) if sims.size else 0.0
        neighbor_count[center] = count
        if sims.size:
            # Official NCP uses exp(-distance / lambda_L) over the KNN support.
            clipped_sims = np.clip(sims.astype(np.float64), -1.0, 1.0)
            distances = np.sqrt(np.clip(2.0 - 2.0 * clipped_sims, 0.0, None))
            weights = np.exp(-distances / ncp_lambda_value)
        else:
            weights = np.ones_like(values, dtype=np.float64)
        weights = np.clip(weights.astype(np.float64), 0.0, None)
        if hubness_correction_value == "degree" and selected_hubness.size:
            weights = weights / np.sqrt(1.0 + selected_hubness)
        weight_sum = float(weights.sum())
        if weight_sum > 0.0:
            normalized_weights = weights / weight_sum
            weighted_mean = float(np.sum(normalized_weights * values))
            ncp_weighted_mean_risk[center] = weighted_mean
            ncp_shrunk_weighted_mean_risk[center] = shrink * weighted_mean + (1.0 - shrink) * float(risk[center])
            ncp_weighted_risk_std[center] = float(np.sqrt(np.sum(normalized_weights * ((values - weighted_mean) ** 2))))
            ncp_weighted_risk_q80[center] = _weighted_quantile(values, normalized_weights, 0.80)
            ncp_weighted_high_risk_mass[center] = float(np.sum(normalized_weights[values >= high_threshold]))
            ncp_weighted_safe_support_mass[center] = float(np.sum(normalized_weights[values <= low_threshold]))
            ncp_effective_sample_size[center] = float((weight_sum ** 2) / max(float(np.sum(weights ** 2)), 1e-12))
            ncp_weight_sum[center] = weight_sum
            ncp_weight_max[center] = float(weights.max()) if weights.size else 0.0
        if preds is not None:
            pred_disagreement[center] = float((preds[selected] != preds[center]).mean())
            if selected.size and ncp_weight_sum[center] > 0.0:
                mismatch = (preds[selected] != preds[center]).astype(np.float64)
                ncp_weighted_prediction_disagreement[center] = float(np.sum((weights / weight_sum) * mismatch))

    def _apply_local_calibration(center, selected, sims):
        if not local_calibration_active or selected.size == 0:
            return
        selected = selected.astype(np.int64, copy=False)
        keep = (selected >= 0) & (selected < num_nodes) & calibration_lookup[selected] & (selected != int(center))
        selected = selected[keep]
        sims = sims[keep] if sims.size == keep.size else np.asarray([], dtype=np.float32)
        if selected.size == 0:
            return
        cal_positions = np.searchsorted(calibration_idx_np, selected)
        valid_positions = (
            (cal_positions >= 0)
            & (cal_positions < calibration_idx_np.size)
            & (calibration_idx_np[cal_positions] == selected)
        )
        if not np.all(valid_positions):
            selected = selected[valid_positions]
            sims = sims[valid_positions] if sims.size == valid_positions.size else np.asarray([], dtype=np.float32)
            cal_positions = cal_positions[valid_positions]
        if selected.size == 0:
            return
        scores = calibration_nonconformity[cal_positions].astype(np.float64)
        if sims.size:
            clipped_sims = np.clip(sims.astype(np.float64), -1.0, 1.0)
            distances = np.sqrt(np.clip(2.0 - 2.0 * clipped_sims, 0.0, None))
            weights = np.exp(-distances / ncp_lambda_value)
        else:
            weights = np.ones_like(scores, dtype=np.float64)
        weights = np.clip(weights.astype(np.float64), 0.0, None)
        if hubness_correction_value == "degree" and hubness_counts.size:
            weights = weights / np.sqrt(1.0 + hubness_counts[selected])
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0:
            return
        normalized_weights = weights / weight_sum
        local_q = float(np.ceil((selected.size + 1) * (1.0 - float(local_calibration_alpha))) / selected.size)
        local_q = min(max(local_q, 0.0), 1.0)
        local_threshold = _weighted_quantile(scores, normalized_weights, local_q)
        ncp_local_threshold[center] = float(np.clip(local_threshold, 1e-8, 1.0))
        ncp_local_threshold_delta[center] = float(ncp_local_threshold[center] - global_threshold)
        ncp_local_calibration_neighbor_count[center] = float(selected.size)
        ncp_local_effective_calibration_sample_size[center] = float(
            (weight_sum ** 2) / max(float(np.sum(weights ** 2)), 1e-12)
        )
        ncp_local_fallback_to_global[center] = 0.0

    def _apply_same_hyperedge_tail_risk(center, selected, sims):
        if not same_hyperedge_tail_active or selected.size == 0:
            return
        selected = selected.astype(np.int64, copy=False)
        keep = (selected >= 0) & (selected < num_nodes) & (selected != int(center))
        selected = selected[keep]
        sims = sims[keep] if sims.size == keep.size else np.asarray([], dtype=np.float32)
        if selected.size == 0:
            return
        support_scores = same_hyperedge_target_nonconformity[selected].astype(np.float64)
        target_score = float(same_hyperedge_target_nonconformity[int(center)])
        if sims.size:
            clipped_sims = np.clip(sims.astype(np.float64), -1.0, 1.0)
            distances = np.sqrt(np.clip(2.0 - 2.0 * clipped_sims, 0.0, None))
            weights = np.exp(-distances / ncp_lambda_value)
        else:
            weights = np.ones_like(support_scores, dtype=np.float64)
        weights = np.clip(weights.astype(np.float64), 0.0, None)
        if hubness_correction_value == "degree" and hubness_counts.size:
            weights = weights / np.sqrt(1.0 + hubness_counts[selected])
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0:
            return
        tail_weight = float(weights[support_scores >= target_score].sum())
        tail_pvalue = (tail_weight + 1.0) / (weight_sum + 1.0)
        same_hyperedge_selected_tail_pvalue[center] = float(np.clip(tail_pvalue, 1e-8, 1.0))
        same_hyperedge_selected_tail_risk[center] = float(np.clip(1.0 - tail_pvalue, 0.0, 1.0))
        same_hyperedge_selected_effective_sample_size[center] = float(
            (weight_sum ** 2) / max(float(np.sum(weights ** 2)), 1e-12)
        )
        same_hyperedge_selected_fallback_to_global[center] = 0.0

    def _apply_same_hyperedge_calibration_tail_risk(center, selected, sims):
        if not same_hyperedge_tail_active or not local_calibration_active or selected.size == 0:
            return
        selected = selected.astype(np.int64, copy=False)
        keep = (selected >= 0) & (selected < num_nodes) & calibration_lookup[selected] & (selected != int(center))
        selected = selected[keep]
        sims = sims[keep] if sims.size == keep.size else np.asarray([], dtype=np.float32)
        if selected.size == 0:
            return
        cal_positions = np.searchsorted(calibration_idx_np, selected)
        valid_positions = (
            (cal_positions >= 0)
            & (cal_positions < calibration_idx_np.size)
            & (calibration_idx_np[cal_positions] == selected)
        )
        if not np.all(valid_positions):
            selected = selected[valid_positions]
            sims = sims[valid_positions] if sims.size == valid_positions.size else np.asarray([], dtype=np.float32)
            cal_positions = cal_positions[valid_positions]
        if selected.size == 0:
            return
        support_scores = calibration_nonconformity[cal_positions].astype(np.float64)
        target_score = float(same_hyperedge_target_nonconformity[int(center)])
        if sims.size:
            clipped_sims = np.clip(sims.astype(np.float64), -1.0, 1.0)
            distances = np.sqrt(np.clip(2.0 - 2.0 * clipped_sims, 0.0, None))
            weights = np.exp(-distances / ncp_lambda_value)
        else:
            weights = np.ones_like(support_scores, dtype=np.float64)
        weights = np.clip(weights.astype(np.float64), 0.0, None)
        if hubness_correction_value == "degree" and hubness_counts.size:
            weights = weights / np.sqrt(1.0 + hubness_counts[selected])
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0:
            return
        tail_weight = float(weights[support_scores >= target_score].sum())
        tail_pvalue = (tail_weight + 1.0) / (weight_sum + 1.0)
        same_hyperedge_calibration_tail_pvalue[center] = float(np.clip(tail_pvalue, 1e-8, 1.0))
        same_hyperedge_calibration_tail_risk[center] = float(np.clip(1.0 - tail_pvalue, 0.0, 1.0))
        effective_sample_size = float(
            (weight_sum ** 2) / max(float(np.sum(weights ** 2)), 1e-12)
        )
        same_hyperedge_calibration_effective_sample_size[center] = effective_sample_size
        shrink = effective_sample_size / (effective_sample_size + float(shrinkage_tau))
        same_hyperedge_calibration_shrink_weight[center] = float(np.clip(shrink, 0.0, 1.0))
        same_hyperedge_calibration_shrunk_tail_risk[center] = float(
            np.clip(
                shrink * same_hyperedge_calibration_tail_risk[center]
                + (1.0 - shrink) * same_hyperedge_calibration_global_tail_risk[center],
                0.0,
                1.0,
            )
        )
        same_hyperedge_calibration_fallback_to_global[center] = 0.0

    if scope in {"labeled_full", "hyperscan_full"}:
        candidate_limit = num_nodes if scope == "hyperscan_full" else labeled_limit
        for center in range(center_limit):
            row = neighbors[center] if neighbors is not None else np.asarray([], dtype=np.int64)
            sim_row = similarities[center] if similarities is not None else np.asarray([], dtype=np.float32)
            valid = (row >= 0) & (row < candidate_limit)
            row_valid = row[valid].astype(np.int64)
            sim_valid = sim_row[valid].astype(np.float32)
            if scope == "hyperscan_full":
                # DHG from_feature_kNN builds a k-member group for each center.
                # If duplicate features make the backend omit the center from
                # the returned top-k row, reserve one slot for the center rather
                # than growing the hyperedge to k + 1.
                has_center = bool(np.any(row_valid == int(center)))
                support_budget = max(int(support_k) - 1, 0) if not has_center else int(row_valid.size)
                keep = row_valid != int(center)
                selected = row_valid[keep][:support_budget]
                sims = sim_valid[keep][:support_budget]
                selected, sims = _filter_support(center, selected.astype(np.int64), sims.astype(np.float32))
                member_count = min(int(support_k), int(selected.size) + 1)
                _apply_selected(center, selected.astype(np.int64), sims.astype(np.float32), member_count=member_count)
                if same_hyperedge_tail_active:
                    _apply_same_hyperedge_tail_risk(
                        center,
                        selected.astype(np.int64),
                        sims.astype(np.float32),
                    )
                    _apply_same_hyperedge_calibration_tail_risk(
                        center,
                        selected.astype(np.int64),
                        sims.astype(np.float32),
                    )
                if local_calibration_active and local_calibration_scope_value == "same_hyperedge":
                    _apply_local_calibration(
                        center,
                        selected.astype(np.int64),
                        sims.astype(np.float32),
                    )
            else:
                keep = (row_valid != int(center))
                selected = row_valid[keep][:support_k]
                sims = sim_valid[keep][:support_k]
                selected, sims = _filter_support(center, selected.astype(np.int64), sims.astype(np.float32))
                _apply_selected(center, selected.astype(np.int64), sims.astype(np.float32))
                if same_hyperedge_tail_active:
                    _apply_same_hyperedge_tail_risk(
                        center,
                        selected.astype(np.int64),
                        sims.astype(np.float32),
                    )
                    _apply_same_hyperedge_calibration_tail_risk(
                        center,
                        selected.astype(np.int64),
                        sims.astype(np.float32),
                    )
                if local_calibration_active and local_calibration_scope_value == "same_hyperedge":
                    _apply_local_calibration(
                        center,
                        selected.astype(np.int64),
                        sims.astype(np.float32),
                    )
        if local_calibration_active and local_calibration_scope_value == "independent_knn":
            calibration_candidate_count = int(calibration_idx_np.size)
            query_k = min(support_k + 1, calibration_candidate_count)
            if query_k > 0:
                cal_repr = repr_np[calibration_idx_np]
                if bool(torch.cuda.is_available()) or int(repr_np.shape[1]) > 128:
                    cal_neighbor_pos, cal_sims, cal_backend = _torch_exact_feature_topk(
                        repr_np[:center_limit],
                        cal_repr,
                        query_k,
                        backend_label="calibration_local",
                    )
                else:
                    try:
                        from scipy.spatial import cKDTree

                        tree = cKDTree(cal_repr)
                        cal_distances, cal_neighbor_pos = tree.query(repr_np[:center_limit], k=query_k)
                        cal_backend = "scipy_ckdtree_calibration_local"
                    except Exception:
                        distance = torch.cdist(torch.from_numpy(repr_np[:center_limit]), torch.from_numpy(cal_repr), p=2)
                        topk = torch.topk(distance, k=query_k, largest=False, dim=1)
                        cal_distances = topk.values.numpy()
                        cal_neighbor_pos = topk.indices.numpy()
                        cal_backend = "torch_cdist_calibration_local"
                    cal_neighbor_pos = np.asarray(cal_neighbor_pos, dtype=np.int64)
                    cal_distances = np.asarray(cal_distances, dtype=np.float32)
                    if cal_neighbor_pos.ndim == 1:
                        cal_neighbor_pos = cal_neighbor_pos.reshape(-1, 1)
                        cal_distances = cal_distances.reshape(-1, 1)
                    cal_sims = np.clip(1.0 - 0.5 * (cal_distances.astype(np.float64) ** 2), -1.0, 1.0).astype(np.float32)
                for center in range(center_limit):
                    row = calibration_idx_np[cal_neighbor_pos[center]]
                    sim_row = cal_sims[center]
                    keep = row != int(center)
                    selected, selected_sims = _filter_support(
                        center,
                        row[keep][:support_k].astype(np.int64),
                        sim_row[keep][:support_k].astype(np.float32),
                    )
                    _apply_local_calibration(
                        center,
                        selected.astype(np.int64),
                        selected_sims.astype(np.float32),
                    )
                backend = f"{backend}+{cal_backend}"
    else:
        for center, row in enumerate(candidate_rows[:center_limit]):
            if not row:
                continue
            candidate_arr = np.asarray([int(item) for item in row if int(item) != int(center)], dtype=np.int64)
            if candidate_arr.size == 0:
                continue
            sims = repr_np[candidate_arr] @ repr_np[int(center)]
            topk = min(support_k, int(candidate_arr.size))
            if topk == int(candidate_arr.size):
                order = np.argsort(-sims, kind="stable")
            else:
                partial = np.argpartition(-sims, topk - 1)[:topk]
                order = partial[np.argsort(-sims[partial], kind="stable")]
            selected, selected_sims = _filter_support(
                center,
                candidate_arr[order].astype(np.int64),
                sims[order].astype(np.float32),
            )
            _apply_selected(center, selected.astype(np.int64), selected_sims.astype(np.float32))
            if same_hyperedge_tail_active:
                _apply_same_hyperedge_tail_risk(center, selected.astype(np.int64), selected_sims.astype(np.float32))
                _apply_same_hyperedge_calibration_tail_risk(center, selected.astype(np.int64), selected_sims.astype(np.float32))
            if local_calibration_active and local_calibration_scope_value == "same_hyperedge":
                _apply_local_calibration(center, selected.astype(np.int64), selected_sims.astype(np.float32))
            if local_calibration_active and local_calibration_scope_value == "independent_knn":
                cal_mask = calibration_lookup[candidate_arr]
                cal_candidates = candidate_arr[cal_mask]
                if cal_candidates.size:
                    cal_sims_all = repr_np[cal_candidates] @ repr_np[int(center)]
                    cal_topk = min(support_k, int(cal_candidates.size))
                    if cal_topk == int(cal_candidates.size):
                        cal_order = np.argsort(-cal_sims_all, kind="stable")
                    else:
                        partial = np.argpartition(-cal_sims_all, cal_topk - 1)[:cal_topk]
                        cal_order = partial[np.argsort(-cal_sims_all[partial], kind="stable")]
                    selected, selected_sims = _filter_support(
                        center,
                        cal_candidates[cal_order].astype(np.int64),
                        cal_sims_all[cal_order].astype(np.float32),
                    )
                    _apply_local_calibration(center, selected.astype(np.int64), selected_sims.astype(np.float32))

    has_candidates = (neighbor_count > 0).astype(np.float32)
    similarity_gap = (1.0 - np.clip(similarity_mean, -1.0, 1.0)).astype(np.float32)
    if posterior_np is not None and posterior_np.ndim == 2:
        (
            ncp_local_abstain_risk,
            ncp_local_set_size,
            ncp_local_coverage_margin,
            ncp_local_margin_risk,
        ) = _prediction_set_risk_from_thresholds(
            posterior_np,
            ncp_local_threshold,
            num_classes=int(posterior_np.shape[1]),
        )
    else:
        ncp_local_abstain_risk = risk.copy().astype(np.float32)
        ncp_local_set_size = np.zeros(num_nodes, dtype=np.int64)
        ncp_local_coverage_margin = np.zeros(num_nodes, dtype=np.float32)
        ncp_local_margin_risk = risk.copy().astype(np.float32)
    features = {
        "knn_mean_risk": np.clip(mean_risk, 0.0, 1.0).astype(np.float32),
        "knn_max_risk": np.clip(max_risk, 0.0, 1.0).astype(np.float32),
        "knn_risk_std": std_risk.astype(np.float32),
        "knn_prediction_disagreement": pred_disagreement.astype(np.float32),
        "knn_high_risk_mass": high_risk_mass.astype(np.float32),
        "knn_safe_support_mass": safe_support_mass.astype(np.float32),
        "knn_similarity_mean": similarity_mean.astype(np.float32),
        "knn_similarity_min": similarity_min.astype(np.float32),
        "knn_similarity_gap": similarity_gap,
        "knn_effective_neighbor_count": neighbor_count.astype(np.float32),
        "knn_support_neighbor_count": support_neighbor_count.astype(np.float32),
        "knn_hyperedge_member_count": hyperedge_member_count.astype(np.float32),
        "ncp_weighted_mean_risk": np.clip(ncp_weighted_mean_risk, 0.0, 1.0).astype(np.float32),
        "ncp_shrunk_weighted_mean_risk": np.clip(ncp_shrunk_weighted_mean_risk, 0.0, 1.0).astype(np.float32),
        "ncp_weighted_risk_std": ncp_weighted_risk_std.astype(np.float32),
        "ncp_weighted_risk_q80": np.clip(ncp_weighted_risk_q80, 0.0, 1.0).astype(np.float32),
        "ncp_weighted_prediction_disagreement": ncp_weighted_prediction_disagreement.astype(np.float32),
        "ncp_weighted_high_risk_mass": ncp_weighted_high_risk_mass.astype(np.float32),
        "ncp_weighted_safe_support_mass": ncp_weighted_safe_support_mass.astype(np.float32),
        "ncp_effective_sample_size": ncp_effective_sample_size.astype(np.float32),
        "ncp_weight_sum": ncp_weight_sum.astype(np.float32),
        "ncp_weight_max": ncp_weight_max.astype(np.float32),
        "knn_hubness_mean": knn_hubness_mean.astype(np.float32),
        "knn_hubness_max": knn_hubness_max.astype(np.float32),
        "knn_filter_fallback": knn_filter_fallback.astype(np.float32),
        "ncp_local_abstain_risk": np.clip(ncp_local_abstain_risk, 0.0, 1.0).astype(np.float32),
        "ncp_local_threshold": np.clip(ncp_local_threshold, 0.0, 1.0).astype(np.float32),
        "ncp_local_threshold_delta": ncp_local_threshold_delta.astype(np.float32),
        "ncp_local_set_size": ncp_local_set_size.astype(np.float32),
        "ncp_local_coverage_margin": ncp_local_coverage_margin.astype(np.float32),
        "ncp_local_margin_risk": np.clip(ncp_local_margin_risk, 0.0, 1.0).astype(np.float32),
        "ncp_local_calibration_neighbor_count": ncp_local_calibration_neighbor_count.astype(np.float32),
        "ncp_local_effective_calibration_sample_size": ncp_local_effective_calibration_sample_size.astype(np.float32),
        "ncp_local_fallback_to_global": ncp_local_fallback_to_global.astype(np.float32),
        "same_hyperedge_target_nonconformity": np.clip(same_hyperedge_target_nonconformity, 0.0, 1.0).astype(np.float32),
        "same_hyperedge_global_tail_risk": np.clip(same_hyperedge_global_tail_risk, 0.0, 1.0).astype(np.float32),
        "same_hyperedge_selected_tail_pvalue": np.clip(same_hyperedge_selected_tail_pvalue, 0.0, 1.0).astype(np.float32),
        "same_hyperedge_selected_tail_risk": np.clip(same_hyperedge_selected_tail_risk, 0.0, 1.0).astype(np.float32),
        "same_hyperedge_selected_effective_sample_size": same_hyperedge_selected_effective_sample_size.astype(np.float32),
        "same_hyperedge_selected_fallback_to_global": same_hyperedge_selected_fallback_to_global.astype(np.float32),
        "same_hyperedge_calibration_global_tail_risk": np.clip(same_hyperedge_calibration_global_tail_risk, 0.0, 1.0).astype(np.float32),
        "same_hyperedge_calibration_tail_pvalue": np.clip(same_hyperedge_calibration_tail_pvalue, 0.0, 1.0).astype(np.float32),
        "same_hyperedge_calibration_tail_risk": np.clip(same_hyperedge_calibration_tail_risk, 0.0, 1.0).astype(np.float32),
        "same_hyperedge_calibration_effective_sample_size": same_hyperedge_calibration_effective_sample_size.astype(np.float32),
        "same_hyperedge_calibration_fallback_to_global": same_hyperedge_calibration_fallback_to_global.astype(np.float32),
        "same_hyperedge_calibration_shrink_weight": same_hyperedge_calibration_shrink_weight.astype(np.float32),
        "same_hyperedge_calibration_shrunk_tail_risk": np.clip(same_hyperedge_calibration_shrunk_tail_risk, 0.0, 1.0).astype(np.float32),
        "knn_has_candidates": has_candidates,
    }
    features["_support_payload"] = {
        "contract": "conformal_knn_router_support_group_v1",
        "source": "conformal_knn_risk_router",
        "candidate_scope": str(scope),
        "backend": str(backend),
        "neighbor_mode": str(neighbor_mode_value),
        "knn_k": int(k),
        "adaptive_max_k": int(adaptive_k),
        "min_support": int(min_support_count),
        "local_calibration_scope": str(local_calibration_scope_value),
        "center_count": int(center_limit),
        "center_node_ids": [int(idx) for idx in range(center_limit)],
        "center_candidate_node_ids": support_candidate_rows,
        "center_candidate_similarity": support_similarity_rows,
        "center_support_neighbor_count": support_neighbor_count[:center_limit].astype(np.int64).tolist(),
        "center_hyperedge_member_count": hyperedge_member_count[:center_limit].astype(np.int64).tolist(),
    }
    features["_metadata"] = {
        "knn_k": int(k),
        "knn_support_k": int(support_k),
        "candidate_scope": scope,
        "neighbor_mode": str(neighbor_mode_value),
        "similarity_threshold": float(sim_threshold),
        "similarity_threshold_active": bool(threshold_active),
        "min_support": int(min_support_count),
        "adaptive_max_k": int(adaptive_k),
        "hubness_correction": str(hubness_correction_value),
        "center_scope": "labeled_graph_only" if labeled_count is not None else "all_nodes",
        "target_risk_contract": "target_node_to_knn_support_group_risk_v1",
        "backend": backend,
        "labeled_candidate_count": int(labeled_limit),
        "hyperscan_candidate_count": int(num_nodes if scope == "hyperscan_full" else labeled_limit),
        "hyperscan_k_includes_center": bool(scope == "hyperscan_full"),
        "knn_center_count": int(center_limit),
        "mean_effective_neighbor_count": float(neighbor_count.mean()) if neighbor_count.size else 0.0,
        "mean_effective_neighbor_count_on_centers": float(neighbor_count[:center_limit].mean()) if center_limit else 0.0,
        "mean_knn_filter_fallback_on_centers": float(knn_filter_fallback[:center_limit].mean()) if center_limit else 0.0,
        "mean_knn_hubness_on_centers": float(knn_hubness_mean[:center_limit].mean()) if center_limit else 0.0,
        "nodes_with_knn_candidates": int(has_candidates.sum()),
        "support_centers_skipped": int(max(num_nodes - center_limit, 0)),
        "ncp_weighting_function": "exp(-euclidean_distance/lambda_L)",
        "ncp_lambda": float(ncp_lambda_value),
        "ncp_official_code_reference": "1995subhankar1995/NCP CP/Classification.py HypertuningBothLamdas/TestProcedure",
        "mean_ncp_effective_sample_size_on_centers": float(ncp_effective_sample_size[:center_limit].mean()) if center_limit else 0.0,
        "ncp_local_calibration_active": bool(local_calibration_active),
        "ncp_local_calibration_split": "cal_idx_labels_only",
        "ncp_local_global_threshold": float(global_threshold),
        "ncp_local_alpha": float(local_calibration_alpha),
        "ncp_local_calibration_count": int(calibration_idx_np.size),
        "ncp_local_centers_with_calibration_neighbors": int((ncp_local_calibration_neighbor_count[:center_limit] > 0).sum()),
        "ncp_local_mean_calibration_neighbor_count_on_centers": float(ncp_local_calibration_neighbor_count[:center_limit].mean()) if center_limit else 0.0,
        "ncp_local_mean_effective_calibration_sample_size_on_centers": float(ncp_local_effective_calibration_sample_size[:center_limit].mean()) if center_limit else 0.0,
        "ncp_local_fallback_count_on_centers": int((ncp_local_fallback_to_global[:center_limit] > 0).sum()) if center_limit else 0,
        "ncp_local_calibration_scope": str(local_calibration_scope_value),
        "ncp_local_score_semantics": (
            "weighted_same_hyperedge_selected_k_tail_risk_on_predicted_class_nonconformity"
            if same_hyperedge_tail_active
            else "weighted_local_conformal_prediction_set_abstain_risk"
        ),
        "ncp_local_quantile": "weighted_quantile_of_calibration_nonconformity_with_finite_sample_correction",
        "same_hyperedge_tail_active": bool(same_hyperedge_tail_active),
        "same_hyperedge_tail_reference": "target_centered_selected_k_support" if same_hyperedge_tail_active else "inactive",
        "same_hyperedge_tail_nonconformity": "1_minus_max_posterior" if same_hyperedge_tail_active else "inactive",
        "same_hyperedge_tail_global_reference_count": int(same_hyperedge_global_reference_count),
        "same_hyperedge_mean_effective_sample_size_on_centers": float(
            same_hyperedge_selected_effective_sample_size[:center_limit].mean()
        ) if center_limit else 0.0,
        "same_hyperedge_fallback_count_on_centers": int(
            (same_hyperedge_selected_fallback_to_global[:center_limit] > 0).sum()
        ) if center_limit else 0,
        "same_hyperedge_calibration_mean_effective_sample_size_on_centers": float(
            same_hyperedge_calibration_effective_sample_size[:center_limit].mean()
        ) if center_limit else 0.0,
        "same_hyperedge_calibration_mean_shrink_weight_on_centers": float(
            same_hyperedge_calibration_shrink_weight[:center_limit].mean()
        ) if center_limit else 0.0,
        "same_hyperedge_calibration_fallback_count_on_centers": int(
            (same_hyperedge_calibration_fallback_to_global[:center_limit] > 0).sum()
        ) if center_limit else 0,
    }
    return features


def _split_selected_nodes_from_risk(risk_score, train_idx, val_idx, test_idx, budgets):
    risk_score = np.asarray(risk_score, dtype=np.float32).reshape(-1)

    def _top_for(idx, budget):
        idx_np = _valid_index_array(idx, risk_score.shape[0])
        if idx_np.size == 0:
            return []
        count = max(int(idx_np.size * float(budget)), 1)
        order = idx_np[np.argsort(-risk_score[idx_np], kind="mergesort")]
        return [int(item) for item in order[:count].tolist()]

    payload = {}
    for budget in parse_budget_list(budgets, default=RESIDUAL_RISK_PAPER_BUDGETS):
        key = f"budget_{int(round(float(budget) * 1000)):03d}"
        train_nodes = _top_for(train_idx, budget)
        valid_nodes = _top_for(val_idx, budget)
        test_nodes = _top_for(test_idx, budget)
        seen = set()
        all_nodes = []
        for item in train_nodes + valid_nodes + test_nodes:
            if int(item) in seen:
                continue
            seen.add(int(item))
            all_nodes.append(int(item))
        payload[key] = {
            "budget": float(budget),
            "train": train_nodes,
            "valid": valid_nodes,
            "test": test_nodes,
            "all": all_nodes,
            "split_counts": {
                "train": int(len(train_nodes)),
                "valid": int(len(valid_nodes)),
                "test": int(len(test_nodes)),
                "all": int(len(all_nodes)),
            },
        }
    return payload


def _top_target_rows(risk_score, split_idx, top_n, *, preds=None, pred_label_score=None, base_risk=None, local_features=None):
    risk_score = np.asarray(risk_score, dtype=np.float32).reshape(-1)
    idx_np = _valid_index_array(split_idx, risk_score.shape[0])
    if idx_np.size == 0 or int(top_n) <= 0:
        return []
    order = idx_np[np.argsort(-risk_score[idx_np], kind="mergesort")][: int(top_n)]
    preds_np = np.asarray(preds).reshape(-1) if preds is not None else None
    pred_score_np = np.asarray(pred_label_score, dtype=np.float32).reshape(-1) if pred_label_score is not None else None
    base_risk_np = np.asarray(base_risk, dtype=np.float32).reshape(-1) if base_risk is not None else None
    local_features = dict(local_features or {})
    rows = []
    for rank, node_id in enumerate(order.tolist(), start=1):
        row = {
            "rank": int(rank),
            "node_id": int(node_id),
            "risk_score": float(risk_score[int(node_id)]),
        }
        if preds_np is not None and int(node_id) < preds_np.shape[0]:
            row["prediction"] = int(preds_np[int(node_id)])
        if pred_score_np is not None and int(node_id) < pred_score_np.shape[0]:
            row["pred_label_score"] = float(pred_score_np[int(node_id)])
        if base_risk_np is not None and int(node_id) < base_risk_np.shape[0]:
            row["base_abstain_risk"] = float(base_risk_np[int(node_id)])
        for name in (
            "knn_mean_risk",
            "knn_prediction_disagreement",
            "knn_high_risk_mass",
            "knn_safe_support_mass",
            "knn_similarity_mean",
            "knn_effective_neighbor_count",
            "knn_support_neighbor_count",
            "knn_hyperedge_member_count",
            "ncp_weighted_mean_risk",
            "ncp_shrunk_weighted_mean_risk",
            "ncp_weighted_risk_q80",
            "ncp_weighted_prediction_disagreement",
            "ncp_weighted_high_risk_mass",
            "ncp_effective_sample_size",
            "ncp_local_abstain_risk",
            "ncp_local_threshold",
            "ncp_local_threshold_delta",
            "ncp_local_set_size",
            "ncp_local_margin_risk",
            "ncp_local_calibration_neighbor_count",
            "ncp_local_effective_calibration_sample_size",
            "ncp_local_fallback_to_global",
            "same_hyperedge_global_tail_risk",
            "same_hyperedge_selected_tail_risk",
            "same_hyperedge_calibration_tail_risk",
            "same_hyperedge_calibration_shrunk_tail_risk",
            "same_hyperedge_calibration_fallback_to_global",
        ):
            values = local_features.get(name)
            if values is not None:
                arr = np.asarray(values).reshape(-1)
                if int(node_id) < arr.shape[0]:
                    row[name] = float(arr[int(node_id)])
        rows.append(row)
    return rows


class CalibratedLocalRiskRouter(PostHocCalibratedRanker):
    metadata = {
        "claim_role": "stage2_calibrated_local_risk_router",
        "stage2_role": "posthoc_calibrated_local_graph_ranker",
        "canonical_stage2": False,
        "scientific_gate": "posterior_calibration_then_local_risk_aggregation",
        "paper_identity": "Calibrated Local-Risk Conformal Router v2",
        "paper_identity_risk": "low_if_router_only_no_classifier_claim",
        "promotion_rule": "conformal_family_ablation_only",
        "prediction_set_estimator": True,
        "diagnosis_or_action": False,
        "input_boundary": "frozen_gnn_posterior_plus_local_scalar_graph_risk",
    }

    SCORE_FAMILIES = {
        "base_only": lambda r0, f: r0,
        "local_dispersion_legacy": lambda r0, f: 0.70 * r0 + 0.20 * 0.5 * (f["mean_in_risk"] + f["mean_out_risk"]) + 0.10 * f["neighbor_risk_std"],
        "direction_gap_legacy": lambda r0, f: 0.65 * r0 + 0.20 * 0.5 * (f["mean_in_risk"] + f["mean_out_risk"]) + 0.15 * f["risk_gap_in_out"],
        "localized_peak_dispersion": lambda r0, f: (
            0.58 * r0
            + 0.17 * f["channel_peak_risk"]
            + 0.10 * f["channel_peak_std"]
            + 0.10 * f["high_risk_mass_peak"]
            + 0.05 * f["risk_gap_in_out"]
        ),
        "directional_relation_conflict": lambda r0, f: (
            0.55 * r0
            + 0.15 * f["channel_peak_risk"]
            + 0.10 * f["high_risk_mass_peak"]
            + 0.10 * f["channel_conflict_peak"]
            + 0.05 * f["risk_gap_in_out"]
            + 0.025 * f["relation_gap_in"]
            + 0.025 * f["relation_gap_out"]
        ),
        "peak_minus_safe_support": lambda r0, f: (
            0.62 * r0
            + 0.20 * f["channel_peak_risk"]
            + 0.12 * f["high_risk_mass_peak"]
            + 0.08 * f["channel_peak_std"]
            - 0.12 * f["low_risk_mass_peak"]
        ),
        "asymmetric_localized_peak": lambda r0, f: (
            0.56 * r0
            + 0.14 * f["channel_peak_risk"]
            + 0.10 * f["neighbor_risk_max"]
            + 0.08 * f["high_risk_mass_peak"]
            + 0.07 * f["risk_gap_in_out"]
            + 0.025 * f["relation_gap_in"]
            + 0.025 * f["relation_gap_out"]
        ),
    }
    ESTIMATOR_SOURCE = "calibrated_local_risk_router"
    ALLOW_SCORE_FAMILY_OVERRIDE = False
    LOCAL_RISK_CONTRACT = "relation_aware_scalar_risk_aggregation_v2"
    LOCAL_RISK_BOUNDARY = (
        "Keeps posterior calibration scalar-only, then adjusts the risk object with localized 1-hop relation/direction-aware aggregation; "
        "does not smooth posterior, does not introduce a learned router head, and uses tune/cal split discipline for family selection vs conformal calibration."
    )
    LITERATURE_BASIS = [
        "Localized Conformal Prediction",
        "SNAPS",
        "RR-GNN",
        "CoRel",
        "Post-hoc Calibrated Ranker",
    ]

    def __init__(self, alpha=0.20, budgets=RESIDUAL_RISK_PAPER_BUDGETS):
        super().__init__(alpha=alpha, budgets=budgets)
        self.selected_score_family = "base_only"
        self.selected_score_family_metrics = {}
        self.local_risk_features_ = {}
        self.local_risk_feature_metadata_ = {}
        self.local_support_payload_ = {}
        self.candidate_score_family_metrics = {}
        self.split_selection_metadata = {}
        self.score_family_override = "auto"

    def _base_scalar_risk(self, logits, probs):
        posterior = self._posterior(logits=logits, probs=probs)
        return self._prediction_set_payload(posterior)["abstain_risk"].astype(np.float32)

    def _compute_local_risk_features(self, base_risk, edge_index=None, edge_type=None, node_repr=None, **kwargs):
        return _relation_aware_scalar_risk_features(
            base_risk,
            edge_index=edge_index,
            edge_type=edge_type,
            node_repr=node_repr,
        )

    def fit(self, logits, probs, labels, train_idx, val_idx, edge_index=None, edge_type=None, node_repr=None, **kwargs):
        self.budgets = parse_budget_list(kwargs.get("budgets", self.budgets), default=RESIDUAL_RISK_PAPER_BUDGETS)
        labels_np = _labels_to_numpy(labels).astype(np.int64)
        labeled_count = int(labels_np.shape[0])
        tune_idx = kwargs.get("tune_idx")
        cal_idx = kwargs.get("cal_idx")
        tune_idx_np = _valid_index_array(tune_idx if tune_idx is not None else val_idx, labeled_count)
        cal_idx_np = _valid_index_array(cal_idx if cal_idx is not None else val_idx, labeled_count)

        super().fit(
            logits=logits,
            probs=probs,
            labels=labels,
            train_idx=train_idx,
            val_idx=cal_idx_np,
            edge_index=None,
            edge_type=None,
            node_repr=None,
            **kwargs,
        )
        posterior = self._posterior(logits=logits, probs=probs)
        preds = posterior.argmax(axis=1)
        wrong = (preds[:labeled_count] != labels_np).astype(np.int32)
        r0 = self._base_scalar_risk(logits, probs)
        local_features = self._compute_local_risk_features(
            r0,
            edge_index=edge_index,
            edge_type=edge_type,
            node_repr=node_repr,
            posterior=posterior,
            labeled_count=labeled_count,
            **kwargs,
        )
        self.local_risk_feature_metadata_ = dict(local_features.pop("_metadata", {}))
        self.local_support_payload_ = dict(local_features.pop("_support_payload", {}) or {})
        self.local_risk_features_ = {name: values.astype(np.float32) for name, values in local_features.items()}
        override = (
            str(kwargs.get("conformal_knn_score_family_override", "auto") or "auto").strip()
            if bool(getattr(self, "ALLOW_SCORE_FAMILY_OVERRIDE", False))
            else "auto"
        )
        if override and override.lower() != "auto":
            if override not in self.SCORE_FAMILIES:
                raise ValueError(
                    f"Unsupported conformal_knn_score_family_override={override!r}; "
                    f"available families: {sorted(self.SCORE_FAMILIES)}."
                )
            self.score_family_override = override
        else:
            self.score_family_override = "auto"

        best_key = None
        best_metrics = None
        best_score = None
        candidate_metrics = {}
        for family_name, scorer in self.SCORE_FAMILIES.items():
            score = np.clip(np.asarray(scorer(r0, self.local_risk_features_), dtype=np.float32), 0.0, 1.0)
            metrics = residual_risk_metrics(wrong[tune_idx_np], score[tune_idx_np], budgets=(0.15,)) if tune_idx_np.size else {}
            candidate_metrics[family_name] = metrics
            family_score = (
                float(metrics.get("utility_at_15", 0.0)),
                float(metrics.get("auprc_error", float("-inf"))) if math.isfinite(float(metrics.get("auprc_error", float("nan")))) else float("-inf"),
                float(metrics.get("auroc_error", float("-inf"))) if math.isfinite(float(metrics.get("auroc_error", float("nan")))) else float("-inf"),
            )
            if best_score is None or family_score > best_score:
                best_score = family_score
                best_key = family_name
                best_metrics = metrics
        if self.score_family_override != "auto":
            best_key = self.score_family_override
            best_metrics = candidate_metrics.get(best_key, {})
        self.selected_score_family = str(best_key or "base_only")
        self.selected_score_family_metrics = dict(best_metrics or {})
        self.candidate_score_family_metrics = dict(candidate_metrics)
        self.split_selection_metadata = {
            "tune_idx_count": int(tune_idx_np.size),
            "cal_idx_count": int(cal_idx_np.size),
            "family_selection_split": "tune_idx" if tune_idx is not None else "val_idx",
            "conformal_threshold_split": "cal_idx" if cal_idx is not None else "val_idx",
            "tune_cal_split_metadata": dict(kwargs.get("tune_cal_split_metadata", {})),
        }
        self.fit_summary = {
            **dict(self.fit_summary),
            "source": self.ESTIMATOR_SOURCE,
            "fit_scope": "validation_split_labels_only_posterior_then_local_risk_v2",
            "base_risk_source": "posthoc_calibrated_ranker",
            "local_risk_contract": self.LOCAL_RISK_CONTRACT,
            "selected_score_family": self.selected_score_family,
            "selected_score_family_metrics": dict(self.selected_score_family_metrics),
            "candidate_score_family_metrics": dict(self.candidate_score_family_metrics),
            "score_family_override": self.score_family_override,
            "local_risk_features": list(self.local_risk_features_.keys()),
            "local_risk_feature_metadata": dict(self.local_risk_feature_metadata_),
            "posterior_smoothed": False,
            "risk_object_smoothed": True,
            "graph_context_used": edge_index is not None,
            "edge_type_used": edge_type is not None,
            "relation_channel_used": edge_type is not None,
            "direction_channel_used": edge_index is not None,
            "embedding_context_used": node_repr is not None,
            "node_repr_used": node_repr is not None,
            "localized_similarity_used": node_repr is not None,
            "local_hop_contract": "one_hop_only",
            "local_graph_scope": "relation_aware_directional_channels",
            "literature_basis": list(self.LITERATURE_BASIS),
            "literature_boundary": self.LOCAL_RISK_BOUNDARY,
            **self.split_selection_metadata,
        }
        self.calibration_metadata = dict(self.fit_summary)
        return self

    def _score_from_local_risk_features(self, r0, local_features):
        scorer = self.SCORE_FAMILIES[self.selected_score_family]
        return np.clip(np.asarray(scorer(r0, local_features), dtype=np.float32), 0.0, 1.0)

    def score(self, logits, probs, edge_index=None, edge_type=None, node_repr=None, **kwargs):
        r0 = self._base_scalar_risk(logits, probs)
        posterior = self._posterior(logits=logits, probs=probs)
        features = self._compute_local_risk_features(
            r0,
            edge_index=edge_index,
            edge_type=edge_type,
            node_repr=node_repr,
            posterior=posterior,
            **kwargs,
        )
        features.pop("_metadata", None)
        return self._score_from_local_risk_features(r0, features)

    def build_manifest(self, logits, probs, labels, val_idx, test_idx, edge_index=None, edge_type=None, node_repr=None, **kwargs):
        posterior = self._posterior(logits=logits, probs=probs)
        base_risk = self._prediction_set_payload(posterior)["abstain_risk"].astype(np.float32)
        labels_np = _labels_to_numpy(labels).astype(np.int64)
        labeled_count = int(labels_np.shape[0])
        local_features = self._compute_local_risk_features(
            base_risk,
            edge_index=edge_index,
            edge_type=edge_type,
            node_repr=node_repr,
            posterior=posterior,
            labeled_count=labeled_count,
            **kwargs,
        )
        local_metadata = dict(local_features.pop("_metadata", {}))
        support_payload = dict(local_features.pop("_support_payload", {}) or {})
        risk_score = self._score_from_local_risk_features(base_risk, local_features)
        preds = posterior.argmax(axis=1)
        preds_labeled = preds[:labeled_count]
        pred_label_score = posterior[np.arange(posterior.shape[0]), preds]
        wrong = (preds_labeled != labels_np).astype(np.int32)
        val_idx_np = _valid_index_array(val_idx, labeled_count)
        test_idx_np = _valid_index_array(test_idx, labeled_count)
        budgets = parse_budget_list(kwargs.get("budgets", self.budgets), default=RESIDUAL_RISK_PAPER_BUDGETS)
        if self.val_threshold is None:
            budget = budgets[0] if budgets else RESIDUAL_RISK_PAPER_BUDGETS[0]
            k = max(int(len(val_idx_np) * float(budget)), 1) if val_idx_np.size else 0
            self.val_threshold = float(np.sort(risk_score[val_idx_np])[-k]) if k else float("inf")
        metadata = {
            **dict(self.calibration_metadata),
            "source": self.ESTIMATOR_SOURCE,
            "threshold_source": "validation_calibrated_local_risk",
            "base_risk_source": "posthoc_calibrated_ranker",
            "local_risk_contract": self.LOCAL_RISK_CONTRACT,
            "selected_score_family": self.selected_score_family,
            "selected_score_family_metrics": dict(self.selected_score_family_metrics),
            "candidate_score_family": dict(self.candidate_score_family_metrics),
            "local_risk_features": list(local_features.keys()),
            "local_risk_feature_metadata": local_metadata,
            "posterior_smoothed": False,
            "risk_object_smoothed": True,
            "embedding_context_used": node_repr is not None,
            "node_repr_used": node_repr is not None,
            "support_group_payload_available": bool(support_payload),
            "local_calibration_router_metadata": dict(getattr(self, "local_calibration_router_metadata_", {}) or {}),
            "learned_router_metadata": dict(getattr(self, "learned_router_metadata_", {}) or {}),
        }
        target_top_n = int(
            kwargs.get(
                "conformal_knn_target_top_n",
                kwargs.get("conformal_knn_anchor_top_n", kwargs.get("anchor_top_n", 200)),
            )
            or 0
        )
        selected_nodes = _split_selected_nodes_from_risk(
            risk_score,
            train_idx=kwargs.get("train_idx", []),
            val_idx=val_idx,
            test_idx=test_idx,
            budgets=budgets,
        )
        default_selected_key = next(iter(selected_nodes), "")
        default_selected_nodes = dict(selected_nodes.get(default_selected_key, {}))
        default_selected_nodes["source_budget_key"] = default_selected_key
        return {
            "risk_score": risk_score,
            "router_score": risk_score,
            "abstain_risk": base_risk,
            "pred_label_score": pred_label_score.astype(np.float32),
            "local_risk_features": {name: values.astype(np.float32).tolist() for name, values in local_features.items()},
            "local_risk_feature_metadata": local_metadata,
            "selected_score_family": self.selected_score_family,
            "selected_score_family_metrics": dict(self.selected_score_family_metrics),
            "candidate_score_family": dict(self.candidate_score_family_metrics),
            "local_calibration_router_metadata": dict(getattr(self, "local_calibration_router_metadata_", {}) or {}),
            "learned_router_metadata": dict(getattr(self, "learned_router_metadata_", {}) or {}),
            "base_risk_source": "posthoc_calibrated_ranker",
            "selected_nodes": default_selected_nodes,
            "selected_nodes_by_budget": selected_nodes,
            "target_selection_contract": "rank_target_nodes_by_target_plus_optional_support_evidence_v1",
            "support_group_contract": "knn_neighbors_are_evidence_only_not_routed_outputs",
            "support_group_payload": support_payload,
            "top_ranked_targets": {
                "test": _top_target_rows(
                    risk_score,
                    test_idx,
                    target_top_n,
                    preds=preds,
                    pred_label_score=pred_label_score,
                    base_risk=base_risk,
                    local_features=local_features,
                ),
                "valid": _top_target_rows(
                    risk_score,
                    val_idx,
                    target_top_n,
                    preds=preds,
                    pred_label_score=pred_label_score,
                    base_risk=base_risk,
                    local_features=local_features,
                ),
            },
            "top_ranked_anchors": {
                "test": _top_target_rows(
                    risk_score,
                    test_idx,
                    target_top_n,
                    preds=preds,
                    pred_label_score=pred_label_score,
                    base_risk=base_risk,
                    local_features=local_features,
                ),
                "valid": _top_target_rows(
                    risk_score,
                    val_idx,
                    target_top_n,
                    preds=preds,
                    pred_label_score=pred_label_score,
                    base_risk=base_risk,
                    local_features=local_features,
                ),
            },
            "thresholds": {
                "validation_risk_threshold": float(self.val_threshold),
                "conformal_nonconformity_threshold": float(self.threshold if self.threshold is not None else 1.0),
            },
            "calibration_metadata": metadata,
            "hcw_mask": ((1.0 - posterior[:labeled_count].max(axis=1)) < 0.1) & (wrong == 1),
            "validation_metrics": _safe_router_metrics(wrong[val_idx_np], risk_score[val_idx_np], budgets=budgets) if val_idx_np.size else {},
            "test_metrics": _safe_router_metrics(wrong[test_idx_np], risk_score[test_idx_np], budgets=budgets) if test_idx_np.size else {},
        }

    def state_dict_payload(self):
        return {
            "alpha": float(self.alpha),
            "graph_smoothing": 0.0,
            "threshold": float(self.threshold if self.threshold is not None else 1.0),
            "num_classes": self.num_classes,
            "fit_summary": self.fit_summary,
            "selected_score_family": self.selected_score_family,
        }


class ConformalKNNRiskRouter(CalibratedLocalRiskRouter):
    metadata = {
        "claim_role": "stage2_target_node_conformal_knn_risk_selector",
        "stage2_role": "posthoc_calibrated_target_knn_risk_ranker",
        "canonical_stage2": False,
        "scientific_gate": "posterior_calibration_then_target_node_knn_similarity_risk_aggregation",
        "paper_identity": "Target-node Conformal-KNN Risk Router",
        "paper_identity_risk": "medium_combination_innovation_if_used_for_llm_anchor_selection",
        "promotion_rule": "candidate_after_target_knn_risk_ablation",
        "prediction_set_estimator": True,
        "diagnosis_or_action": False,
        "input_boundary": "frozen_gnn_posterior_plus_target_node_knn_similarity_group_risk",
    }
    ESTIMATOR_SOURCE = "conformal_knn_risk_router"
    ALLOW_SCORE_FAMILY_OVERRIDE = True
    LOCAL_RISK_CONTRACT = "target_node_ncp_weighted_knn_similarity_risk_aggregation_v2"
    LOCAL_RISK_BOUNDARY = (
        "Reuses the calibrated_local_risk_router conformal threshold and split discipline, "
        "but replaces relation-neighbor aggregation with target-node KNN support-group risk features, "
        "including NCP-style exp(-distance/lambda_L) weighted support evidence. "
        "It ranks target nodes by fixed KNN-informed risk or NCP-style local calibration risk for downstream LLM/refiner use and does not alter classifier logits."
    )
    LITERATURE_BASIS = [
        "Localized Conformal Prediction",
        "Neighborhood Conformal Prediction",
        "1995subhankar1995/NCP official code",
        "SNAPS",
        "Conformal Risk Control",
        "HyperScan-style dynamic KNN group structure",
    ]
    SCORE_FAMILIES = {
        "base_only": lambda r0, f: r0,
        "knn_mean_blend": lambda r0, f: 0.62 * r0 + 0.25 * f["knn_mean_risk"] + 0.13 * f["knn_prediction_disagreement"],
        "knn_peak_threat": lambda r0, f: (
            0.54 * r0
            + 0.18 * f["knn_max_risk"]
            + 0.12 * f["knn_high_risk_mass"]
            + 0.10 * f["knn_prediction_disagreement"]
            + 0.06 * f["knn_similarity_gap"]
        ),
        "knn_conflict_dispersion": lambda r0, f: (
            0.52 * r0
            + 0.18 * f["knn_mean_risk"]
            + 0.12 * f["knn_risk_std"]
            + 0.12 * f["knn_prediction_disagreement"]
            + 0.06 * f["knn_high_risk_mass"]
        ),
        "knn_safe_support_subtract": lambda r0, f: (
            0.60 * r0
            + 0.20 * f["knn_mean_risk"]
            + 0.12 * f["knn_prediction_disagreement"]
            + 0.08 * f["knn_high_risk_mass"]
            - 0.10 * f["knn_safe_support_mass"]
        ),
    }
    NCP_LOCAL_SCORE_FAMILIES = {
        "ncp_local_conformal": lambda r0, f: f["ncp_local_abstain_risk"],
        "ncp_local_margin": lambda r0, f: f["ncp_local_margin_risk"],
        "ncp_knn_weighted_mean": lambda r0, f: f["ncp_shrunk_weighted_mean_risk"],
        "ncp_local_base_blend": lambda r0, f: 0.80 * r0 + 0.20 * f["ncp_local_abstain_risk"],
        "ncp_local_margin_blend": lambda r0, f: 0.80 * r0 + 0.20 * f["ncp_local_margin_risk"],
        "ncp_local_knn_conformal_blend": lambda r0, f: (
            0.60 * r0
            + 0.25 * f["ncp_shrunk_weighted_mean_risk"]
            + 0.15 * f["ncp_local_abstain_risk"]
        ),
        "ncp_local_knn_margin_blend": lambda r0, f: (
            0.60 * r0
            + 0.25 * f["ncp_shrunk_weighted_mean_risk"]
            + 0.15 * f["ncp_local_margin_risk"]
        ),
    }
    SAME_HYPEREDGE_SUPPORT_SCORE_FAMILIES = {
        "same_hyperedge_selected_tail": lambda r0, f: f["same_hyperedge_selected_tail_risk"],
        "same_hyperedge_global_tail": lambda r0, f: f["same_hyperedge_global_tail_risk"],
        "same_hyperedge_calibration_tail": lambda r0, f: f["same_hyperedge_calibration_tail_risk"],
        "same_hyperedge_calibration_shrunk_tail": lambda r0, f: f["same_hyperedge_calibration_shrunk_tail_risk"],
    }
    def __init__(self, alpha=0.20, budgets=RESIDUAL_RISK_PAPER_BUDGETS):
        super().__init__(alpha=alpha, budgets=budgets)
        self.learning_mode = "fixed"
        self.local_calibration_router_metadata_ = {}
        self.local_calibration_labels_ = None
        self.local_calibration_idx_ = np.empty(0, dtype=np.int64)
        self.local_calibration_scope_ = "independent_knn"
        self.nonparametric_family_pool_name_ = "ncp_local"
        self.learned_router_scaler_ = None
        self.learned_router_classifier_ = None
        self.learned_router_feature_names_ = []
        self.learned_router_metrics_ = {}

    def _learned_feature_names(self, local_features):
        preferred = [
            "knn_mean_risk",
            "knn_max_risk",
            "knn_risk_std",
            "knn_prediction_disagreement",
            "knn_high_risk_mass",
            "knn_safe_support_mass",
            "knn_similarity_mean",
            "knn_similarity_gap",
            "knn_effective_neighbor_count",
            "knn_support_neighbor_count",
            "knn_hyperedge_member_count",
            "ncp_weighted_mean_risk",
            "ncp_shrunk_weighted_mean_risk",
            "ncp_weighted_risk_q80",
            "ncp_weighted_prediction_disagreement",
            "ncp_weighted_high_risk_mass",
            "ncp_effective_sample_size",
            "ncp_local_abstain_risk",
            "ncp_local_threshold",
            "ncp_local_threshold_delta",
            "ncp_local_set_size",
            "ncp_local_margin_risk",
            "ncp_local_calibration_neighbor_count",
            "ncp_local_effective_calibration_sample_size",
            "same_hyperedge_selected_tail_risk",
            "same_hyperedge_global_tail_risk",
            "same_hyperedge_calibration_tail_risk",
            "same_hyperedge_calibration_shrunk_tail_risk",
            "same_hyperedge_selected_effective_sample_size",
            "same_hyperedge_calibration_effective_sample_size",
        ]
        return [name for name in preferred if name in local_features]

    def _learned_feature_matrix(self, base_risk, local_features):
        feature_map = {"base_abstain_risk": np.asarray(base_risk, dtype=np.float32).reshape(-1)}
        feature_names = ["base_abstain_risk"]
        for name in self._learned_feature_names(local_features):
            values = np.asarray(local_features.get(name), dtype=np.float32).reshape(-1)
            feature_map[name] = values
            feature_names.append(name)
        matrix = _feature_matrix_from_map(feature_map, feature_names, int(feature_map["base_abstain_risk"].shape[0]))
        return matrix, feature_names

    def _fit_learned_router(self, feature_matrix, target_labels):
        if feature_matrix.shape[0] == 0 or feature_matrix.shape[1] == 0:
            return False
        labels = np.asarray(target_labels, dtype=np.int64).reshape(-1)
        if np.unique(labels).size < 2:
            return False
        self.learned_router_scaler_ = StandardScaler()
        scaled = self.learned_router_scaler_.fit_transform(feature_matrix)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(scaled, labels)
        self.learned_router_classifier_ = clf
        return True

    def _learned_router_score(self, feature_matrix):
        if self.learned_router_scaler_ is None or self.learned_router_classifier_ is None:
            return None
        scaled = self.learned_router_scaler_.transform(feature_matrix)
        return self.learned_router_classifier_.predict_proba(scaled)[:, 1].astype(np.float32)

    def _local_calibration_kwargs(self, kwargs):
        payload = dict(kwargs)
        if payload.get("conformal_knn_local_calibration_labels") is None:
            payload["conformal_knn_local_calibration_labels"] = self.local_calibration_labels_
        if payload.get("conformal_knn_local_calibration_idx") is None:
            payload["conformal_knn_local_calibration_idx"] = self.local_calibration_idx_
        payload["conformal_knn_global_threshold"] = float(self.threshold if self.threshold is not None else 1.0)
        payload["conformal_knn_local_calibration_alpha"] = float(self.alpha)
        return payload

    def fit(self, logits, probs, labels, train_idx, val_idx, edge_index=None, edge_type=None, node_repr=None, **kwargs):
        self.learning_mode = str(kwargs.get("conformal_knn_learning_mode", "fixed") or "fixed").strip().lower()
        if self.learning_mode not in {"fixed", "ncp_local", "learned_logistic"}:
            raise ValueError("--conformal_knn_learning_mode must be one of {fixed, ncp_local, learned_logistic}.")
        requested_override = str(kwargs.get("conformal_knn_score_family_override", "auto") or "auto").strip()
        requested_override_key = requested_override.lower()
        self.local_calibration_scope_ = str(
            kwargs.get("conformal_knn_local_calibration_scope", "independent_knn") or "independent_knn"
        ).strip().lower()
        labels_np = _labels_to_numpy(labels).astype(np.int64)
        self.local_calibration_labels_ = labels
        self.local_calibration_idx_ = _valid_index_array(
            kwargs.get("cal_idx", val_idx),
            int(labels_np.shape[0]),
        )
        fit_kwargs = self._local_calibration_kwargs(kwargs)
        if self.learning_mode == "ncp_local" and requested_override_key != "auto":
            available = (
                set(self.SCORE_FAMILIES)
                | set(self.NCP_LOCAL_SCORE_FAMILIES)
                | set(self.SAME_HYPEREDGE_SUPPORT_SCORE_FAMILIES)
            )
            if requested_override not in available:
                raise ValueError(
                    f"Unsupported conformal_knn_score_family_override={requested_override!r}; "
                    f"available families: {sorted(available)}."
                )
            if (
                requested_override in self.NCP_LOCAL_SCORE_FAMILIES
                or requested_override in self.SAME_HYPEREDGE_SUPPORT_SCORE_FAMILIES
            ):
                # Parent fit only knows fixed KNN score families; compute features
                # there, then apply the NCP-local override below.
                fit_kwargs["conformal_knn_score_family_override"] = "auto"
        super().fit(
            logits=logits,
            probs=probs,
            labels=labels,
            train_idx=train_idx,
            val_idx=val_idx,
            edge_index=edge_index,
            edge_type=edge_type,
            node_repr=node_repr,
            **fit_kwargs,
        )
        tune_idx_np = _valid_index_array(kwargs.get("tune_idx", val_idx), int(labels_np.shape[0]))
        posterior = self._posterior(logits=logits, probs=probs)
        wrong = (posterior.argmax(axis=1)[: int(labels_np.shape[0])] != labels_np).astype(np.int32)
        r0 = self._base_scalar_risk(logits, probs)
        local_metadata = dict(self.local_risk_feature_metadata_ or {})
        self.local_calibration_router_metadata_ = {
            "learning_mode": self.learning_mode,
            "local_calibration_router_active": bool(self.learning_mode == "ncp_local"),
            "router_model": "logistic_regression" if self.learning_mode == "learned_logistic" else "none",
            "uses_logistic": bool(self.learning_mode == "learned_logistic"),
            "calibration_split": "cal_idx",
            "calibration_count": int(self.local_calibration_idx_.size),
            "score_semantics": (
                "validation_selected_nonparametric_ncp_local_risk_family"
                if self.learning_mode == "ncp_local"
                else (
                    "train_fit_logistic_residual_risk_scorer_over_structured_knn_features"
                    if self.learning_mode == "learned_logistic"
                    else "fixed_validation_selected_knn_risk_family"
                )
            ),
            "official_ncp_alignment": (
                "Uses NCP-style KNN neighborhood localization weights with no learned logistic risk head. "
                "Under independent_knn it keeps calibration-only local conformal thresholds; under same_hyperedge "
                "it now scores the target directly from the selected-K HyperScan-aligned support tail risk."
            ),
            "local_risk_feature_metadata": {
                key: local_metadata.get(key)
                for key in (
                    "ncp_local_calibration_active",
                    "ncp_local_global_threshold",
                    "ncp_local_alpha",
                    "ncp_local_calibration_count",
                    "ncp_local_centers_with_calibration_neighbors",
                    "ncp_local_mean_calibration_neighbor_count_on_centers",
                    "ncp_local_fallback_count_on_centers",
                    "neighbor_mode",
                    "similarity_threshold",
                    "similarity_threshold_active",
                    "min_support",
                    "adaptive_max_k",
                    "hubness_correction",
                    "mean_knn_filter_fallback_on_centers",
                    "mean_knn_hubness_on_centers",
                )
                if key in local_metadata
            },
        }
        if self.learning_mode == "learned_logistic":
            fit_idx_np = _valid_index_array(kwargs.get("train_idx", train_idx), int(labels_np.shape[0]))
            feature_matrix, feature_names = self._learned_feature_matrix(self.local_risk_features_.get("base_risk", r0), self.local_risk_features_)
            self.learned_router_feature_names_ = list(feature_names)
            router_available = self._fit_learned_router(feature_matrix[fit_idx_np], wrong[fit_idx_np]) if fit_idx_np.size else False
            learned_score = self._learned_router_score(feature_matrix) if router_available else None
            val_idx_np = _valid_index_array(val_idx, int(labels_np.shape[0]))
            train_metrics = (
                residual_risk_metrics(wrong[fit_idx_np], learned_score[fit_idx_np], budgets=(0.15,))
                if router_available and fit_idx_np.size
                else {}
            )
            valid_metrics = (
                residual_risk_metrics(wrong[val_idx_np], learned_score[val_idx_np], budgets=(0.15,))
                if router_available and val_idx_np.size
                else {}
            )
            self.learned_router_metrics_ = {
                "router_available": bool(router_available),
                "fit_count": int(fit_idx_np.size),
                "fit_positive_count": int(wrong[fit_idx_np].sum()) if fit_idx_np.size else 0,
                "train_metrics": train_metrics,
                "valid_metrics": valid_metrics,
            }
            self.local_calibration_router_metadata_.update(
                {
                    "learned_router_feature_names": list(feature_names),
                    "learned_router_router_available": bool(router_available),
                    "learned_router_fit_scope": "train_split_residual_error_labels_only",
                    "learned_router_target_semantics": "predict_base_detector_error_probability",
                    "learned_router_train_metrics": dict(train_metrics),
                    "learned_router_valid_metrics": dict(valid_metrics),
                }
            )
            self.selected_score_family = "learned_logistic_residual_router"
            self.selected_score_family_metrics = dict(valid_metrics)
            self.candidate_score_family_metrics = {
                "learned_logistic_residual_router": dict(valid_metrics),
            }
            self.fit_summary = {
                **dict(self.fit_summary),
                "selected_score_family": self.selected_score_family,
                "selected_score_family_metrics": dict(self.selected_score_family_metrics),
                "candidate_score_family_metrics": dict(self.candidate_score_family_metrics),
                "learned_router_feature_names": list(feature_names),
                "learned_router_metrics": dict(self.learned_router_metrics_),
            }
            self.calibration_metadata = dict(self.fit_summary)
            return self
        if self.learning_mode == "ncp_local" and requested_override_key != "auto" and requested_override in self.SCORE_FAMILIES:
            self.score_family_override = requested_override
            self.local_calibration_router_metadata_["selected_ncp_local_family"] = self.selected_score_family
            self.local_calibration_router_metadata_["ncp_local_family_selection_rule"] = (
                "explicit_fixed_score_family_override"
            )
        elif self.learning_mode == "ncp_local":
            if self.local_calibration_scope_ == "same_hyperedge":
                candidate_family_pool = dict(self.SAME_HYPEREDGE_SUPPORT_SCORE_FAMILIES)
                default_family = "same_hyperedge_selected_tail"
                self.nonparametric_family_pool_name_ = "same_hyperedge_selected_tail"
            else:
                candidate_family_pool = dict(self.NCP_LOCAL_SCORE_FAMILIES)
                default_family = "ncp_local_conformal"
                self.nonparametric_family_pool_name_ = "ncp_local"
            local_candidate_metrics = {}
            best_key = None
            best_score = None
            best_metrics = None
            for family_name, scorer in candidate_family_pool.items():
                score = np.clip(np.asarray(scorer(r0, self.local_risk_features_), dtype=np.float32), 0.0, 1.0)
                metrics = residual_risk_metrics(wrong[tune_idx_np], score[tune_idx_np], budgets=(0.15,)) if tune_idx_np.size else {}
                local_candidate_metrics[family_name] = metrics
                family_score = (
                    float(metrics.get("auprc_error", float("-inf"))) if math.isfinite(float(metrics.get("auprc_error", float("nan")))) else float("-inf"),
                    -float(metrics.get("aurc", float("inf"))) if math.isfinite(float(metrics.get("aurc", float("nan")))) else float("-inf"),
                    float(metrics.get("utility_at_15", 0.0)),
                    float(metrics.get("auroc_error", float("-inf"))) if math.isfinite(float(metrics.get("auroc_error", float("nan")))) else float("-inf"),
                )
                if best_score is None or family_score > best_score:
                    best_score = family_score
                    best_key = family_name
                    best_metrics = metrics
            if requested_override_key != "auto":
                best_key = requested_override
                best_metrics = local_candidate_metrics.get(best_key, {})
                self.score_family_override = requested_override
            self.selected_score_family = str(best_key or default_family)
            self.selected_score_family_metrics = dict(best_metrics or {})
            self.local_calibration_router_metadata_["ncp_local_candidate_family_metrics"] = dict(local_candidate_metrics)
            self.local_calibration_router_metadata_["selected_ncp_local_family"] = self.selected_score_family
            self.local_calibration_router_metadata_["nonparametric_family_pool"] = self.nonparametric_family_pool_name_
            self.local_calibration_router_metadata_["ncp_local_family_selection_rule"] = (
                "explicit_ncp_local_score_family_override"
                if requested_override_key != "auto"
                else (
                    "valid_tune_auprc_error_then_lower_aurc_then_utility_at_15_then_auroc_error_same_hyperedge_selected_tail"
                    if self.local_calibration_scope_ == "same_hyperedge"
                    else "valid_tune_auprc_error_then_lower_aurc_then_utility_at_15_then_auroc_error"
                )
            )
            self.candidate_score_family_metrics = {
                **dict(self.candidate_score_family_metrics),
                **dict(local_candidate_metrics),
            }
            self.fit_summary = {
                **dict(self.fit_summary),
                "selected_score_family": self.selected_score_family,
                "selected_score_family_metrics": dict(self.selected_score_family_metrics),
                "candidate_score_family_metrics": dict(self.candidate_score_family_metrics),
            }
        self.fit_summary = {
            **dict(self.fit_summary),
            "learning_mode": self.learning_mode,
            "local_calibration_router_metadata": dict(self.local_calibration_router_metadata_),
        }
        self.calibration_metadata = dict(self.fit_summary)
        return self

    def _score_from_local_risk_features(self, r0, local_features):
        if self.learning_mode == "learned_logistic":
            feature_matrix, _ = self._learned_feature_matrix(r0, local_features)
            scores = self._learned_router_score(feature_matrix)
            if scores is None:
                return np.asarray(r0, dtype=np.float32).reshape(-1)
            return np.clip(np.asarray(scores, dtype=np.float32).reshape(-1), 0.0, 1.0).astype(np.float32)
        if self.learning_mode == "ncp_local" and self.selected_score_family in self.NCP_LOCAL_SCORE_FAMILIES:
            scorer = self.NCP_LOCAL_SCORE_FAMILIES[self.selected_score_family]
            values = np.asarray(scorer(r0, local_features), dtype=np.float32).reshape(-1)
            return np.clip(values, 0.0, 1.0).astype(np.float32)
        if self.learning_mode == "ncp_local" and self.selected_score_family in self.SAME_HYPEREDGE_SUPPORT_SCORE_FAMILIES:
            scorer = self.SAME_HYPEREDGE_SUPPORT_SCORE_FAMILIES[self.selected_score_family]
            values = np.asarray(scorer(r0, local_features), dtype=np.float32).reshape(-1)
            return np.clip(values, 0.0, 1.0).astype(np.float32)
        return super()._score_from_local_risk_features(r0, local_features)

    def _compute_local_risk_features(self, base_risk, edge_index=None, edge_type=None, node_repr=None, **kwargs):
        kwargs = self._local_calibration_kwargs(kwargs)
        return _conformal_knn_scalar_risk_features(
            base_risk,
            edge_index=edge_index,
            edge_type=edge_type,
            node_repr=node_repr,
            posterior=kwargs.get("posterior"),
            labeled_count=kwargs.get("labeled_count"),
            knn_k=kwargs.get("conformal_knn_k", kwargs.get("knn_k", 8)),
            candidate_scope=kwargs.get("conformal_knn_candidate_scope", kwargs.get("candidate_scope", "labeled_full")),
            shrinkage_tau=kwargs.get("conformal_knn_shrinkage_tau", 3.0),
            ncp_lambda=kwargs.get("conformal_knn_ncp_lambda", 1.0),
            neighbor_mode=kwargs.get("conformal_knn_neighbor_mode", "standard"),
            similarity_threshold=kwargs.get("conformal_knn_similarity_threshold", -1.0),
            min_support=kwargs.get("conformal_knn_min_support", 1),
            adaptive_max_k=kwargs.get("conformal_knn_adaptive_max_k", 0),
            hubness_correction=kwargs.get("conformal_knn_hubness_correction", "none"),
            local_calibration_labels=kwargs.get("conformal_knn_local_calibration_labels"),
            local_calibration_idx=kwargs.get("conformal_knn_local_calibration_idx"),
            local_calibration_alpha=kwargs.get("conformal_knn_local_calibration_alpha", self.alpha),
            global_conformal_threshold=kwargs.get("conformal_knn_global_threshold", self.threshold),
            local_calibration_scope=kwargs.get("conformal_knn_local_calibration_scope", "independent_knn"),
        )

    def state_dict_payload(self):
        payload = super().state_dict_payload()
        payload.update(
            {
                "learning_mode": self.learning_mode,
                "local_calibration_router_metadata": dict(self.local_calibration_router_metadata_),
                "learned_router_feature_names": list(self.learned_router_feature_names_),
                "learned_router_metrics": dict(self.learned_router_metrics_),
            }
        )
        return payload


class GNN2HopConformalEstimator(GraphConformalSetEstimator):
    metadata = {
        "claim_role": "stage3_ego_quality_estimator",
        "stage2_role": "gnn_side_2hop_hard_node_quality_gate",
        "canonical_stage2": False,
        "scientific_gate": "valid_tune_then_valid_cal_labels_only",
        "paper_identity": "GNN-side 2-hop Conformal Ego Quality Estimator",
        "paper_identity_risk": "low_if_used_as_prediction_set_quality_gate_only",
        "promotion_rule": "main_gnn_side_estimator_after_ablation",
        "prediction_set_estimator": True,
        "diagnosis_or_action": False,
        "lm_gnn_disagreement_used": False,
        "input_boundary": "gnn_posterior_optional_gnn_or_simteg_embedding_original_graph_only",
    }

    def __init__(
        self,
        alpha=0.20,
        rho_grid=((0.1, 0.0), (0.2, 0.0), (0.1, 0.1), (0.2, 0.1)),
        temperatures=(None, 0.5, 1.0),
        budgets=RESIDUAL_RISK_PAPER_BUDGETS,
    ):
        super().__init__(alpha=alpha, graph_smoothing=0.0, budgets=budgets)
        self.rho_grid = tuple((float(rho1), float(rho2)) for rho1, rho2 in rho_grid)
        self.temperatures = tuple(temperatures)
        self.rho1 = 0.0
        self.rho2 = 0.0
        self.temperature = None
        self.nonconformity_scores_ = None
        self.ego2_metadata_ = {}
        self.tune_summary = {}
        self.split_metadata_ = {}
        self.direction_mode = "incoming"
        self.relation_mode = "agnostic"

    @staticmethod
    def _index_sha256(indices):
        arr = np.asarray(indices, dtype=np.int64).reshape(-1)
        digest = hashlib.sha256()
        digest.update(str(tuple(arr.shape)).encode("utf-8"))
        digest.update(str(arr.dtype).encode("utf-8"))
        digest.update(arr.tobytes())
        return digest.hexdigest()

    def _valid_rho_grid(self):
        grid = []
        for rho1, rho2 in self.rho_grid:
            if rho1 < 0.0 or rho2 < 0.0:
                continue
            if rho1 + rho2 > 1.0:
                continue
            grid.append((float(rho1), float(rho2)))
        return grid or [(0.0, 0.0)]

    def _directed_hop_rows(
        self,
        edge_index,
        num_nodes,
        edge_type=None,
        edge_weight=None,
        direction_mode="incoming",
        relation_mode="agnostic",
    ):
        hop1 = [dict() for _ in range(num_nodes)]
        if edge_index is None:
            return hop1, [dict() for _ in range(num_nodes)]
        edge_np = _to_numpy(edge_index).astype(np.int64)
        if edge_np.ndim != 2 or edge_np.shape[0] != 2:
            raise ValueError("gnn_2hop_conformal expects edge_index shaped [2, num_edges].")
        src, dst = edge_np
        valid = (src >= 0) & (src < num_nodes) & (dst >= 0) & (dst < num_nodes)
        src = src[valid]
        dst = dst[valid]
        if edge_weight is None:
            weights = np.ones(src.shape[0], dtype=np.float64)
        else:
            weights_all = _to_numpy(edge_weight).astype(np.float64).reshape(-1)
            weights = weights_all[valid]
        if edge_type is None:
            types = np.zeros(src.shape[0], dtype=np.int64)
        else:
            types_all = _to_numpy(edge_type).astype(np.int64).reshape(-1)
            types = types_all[valid]

        direction_mode = str(direction_mode or "incoming").lower()
        relation_mode = str(relation_mode or "agnostic").lower()
        if direction_mode not in {"incoming", "outgoing", "undirected"}:
            raise ValueError(f"Unsupported gnn_2hop_conformal direction_mode: {direction_mode}")
        if relation_mode not in {"agnostic", "typed_normalized"}:
            raise ValueError(f"Unsupported gnn_2hop_conformal relation_mode: {relation_mode}")

        relation_norm = {}
        if relation_mode == "typed_normalized":
            for rel in np.unique(types):
                relation_norm[int(rel)] = float(np.clip(weights[types == rel].sum(), 1e-12, None))

        def add_edge(center, neighbor, rel, weight):
            if int(center) == int(neighbor):
                return
            value = float(weight)
            if relation_mode == "typed_normalized":
                value = value / relation_norm.get(int(rel), 1.0)
            row = hop1[int(center)]
            row[int(neighbor)] = row.get(int(neighbor), 0.0) + value

        for s, d, rel, weight in zip(src, dst, types, weights):
            if direction_mode in {"incoming", "undirected"}:
                add_edge(d, s, rel, weight)
            if direction_mode in {"outgoing", "undirected"}:
                add_edge(s, d, rel, weight)
        hop2 = [dict() for _ in range(num_nodes)]
        for node_idx, row in enumerate(hop1):
            two_hop = hop2[node_idx]
            for mid, first_count in row.items():
                for source, second_count in hop1[mid].items():
                    if source == node_idx:
                        continue
                    two_hop[source] = two_hop.get(source, 0.0) + first_count * second_count
        return hop1, hop2

    def _row_normalized_sparse_mean(self, scores, row_counts, center_idx, node_repr=None, temperature=None):
        if not row_counts:
            return scores[center_idx], 0, 0.0
        ids = np.asarray(sorted(row_counts.keys()), dtype=np.int64)
        weights = np.asarray([row_counts[int(idx)] for idx in ids], dtype=np.float64)
        if node_repr is not None and temperature is not None:
            repr_np = np.asarray(node_repr, dtype=np.float64)
            center_vec = repr_np[int(center_idx)]
            neighbor_vecs = repr_np[ids]
            center_norm = np.linalg.norm(center_vec)
            neighbor_norm = np.linalg.norm(neighbor_vecs, axis=1)
            denom = np.clip(center_norm * neighbor_norm, 1e-12, None)
            cosine = np.matmul(neighbor_vecs, center_vec) / denom
            sim_weight = np.exp(np.clip(cosine / max(float(temperature), 1e-8), -50.0, 50.0))
            weights = weights * sim_weight
        weights = weights / np.clip(weights.sum(), 1e-12, None)
        return np.sum(scores[ids] * weights[:, None], axis=0), int(ids.size), float(weights.max())

    def _ego2_nonconformity(
        self,
        posterior,
        edge_index=None,
        edge_type=None,
        node_repr=None,
        edge_weight=None,
        rho1=0.0,
        rho2=0.0,
        temperature=None,
        direction_mode="incoming",
        relation_mode="agnostic",
    ):
        posterior = np.asarray(posterior, dtype=np.float64)
        scores = 1.0 - posterior
        num_nodes = int(scores.shape[0])
        if edge_index is None or (rho1 <= 0.0 and rho2 <= 0.0):
            metadata = {
                "hop1_count": [0 for _ in range(num_nodes)],
                "hop2_count": [0 for _ in range(num_nodes)],
                "max_hop1_weight": [0.0 for _ in range(num_nodes)],
                "max_hop2_weight": [0.0 for _ in range(num_nodes)],
            }
            return scores, metadata
        repr_np = None
        if node_repr is not None and temperature is not None:
            repr_np = _to_numpy(node_repr).astype(np.float64)
            if repr_np.ndim != 2 or repr_np.shape[0] != num_nodes:
                raise ValueError("gnn_2hop_conformal node_repr must be shaped [num_nodes, dim].")
        hop1_rows, hop2_rows = self._directed_hop_rows(
            edge_index,
            num_nodes,
            edge_type=edge_type,
            edge_weight=edge_weight,
            direction_mode=direction_mode,
            relation_mode=relation_mode,
        )
        aggregated = np.zeros_like(scores, dtype=np.float64)
        hop1_count = []
        hop2_count = []
        max_hop1_weight = []
        max_hop2_weight = []
        for node_idx in range(num_nodes):
            hop1_mean, h1_count, h1_max = self._row_normalized_sparse_mean(
                scores,
                hop1_rows[node_idx],
                node_idx,
                node_repr=repr_np,
                temperature=temperature,
            )
            hop2_mean, h2_count, h2_max = self._row_normalized_sparse_mean(
                scores,
                hop2_rows[node_idx],
                node_idx,
                node_repr=repr_np,
                temperature=temperature,
            )
            local_rho1 = float(rho1) if h1_count else 0.0
            local_rho2 = float(rho2) if h2_count else 0.0
            center_weight = 1.0 - local_rho1 - local_rho2
            aggregated[node_idx] = center_weight * scores[node_idx] + local_rho1 * hop1_mean + local_rho2 * hop2_mean
            hop1_count.append(h1_count)
            hop2_count.append(h2_count)
            max_hop1_weight.append(h1_max)
            max_hop2_weight.append(h2_max)
        metadata = {
            "hop1_count": hop1_count,
            "hop2_count": hop2_count,
            "max_hop1_weight": max_hop1_weight,
            "max_hop2_weight": max_hop2_weight,
        }
        return aggregated, metadata

    def _threshold_for(self, score_matrix, labels_np, idx_np):
        if idx_np.size == 0:
            return 1.0
        calibration_scores = score_matrix[idx_np, labels_np[idx_np]]
        return self._fit_threshold(calibration_scores)

    def _prediction_set_payload_from_scores(self, scores):
        scores = np.asarray(scores, dtype=np.float64)
        threshold = float(self.threshold if self.threshold is not None else 1.0)
        in_set = scores <= threshold + 1e-12
        prediction_sets = []
        for row_idx, row in enumerate(in_set):
            labels = np.flatnonzero(row).astype(int).tolist()
            if not labels:
                labels = [int(np.argmin(scores[row_idx]))]
            prediction_sets.append(labels)
        set_size = np.asarray([len(labels) for labels in prediction_sets], dtype=np.float32)
        sorted_scores = np.sort(scores, axis=1)
        pred_labels = scores.argmin(axis=1)
        pred_label_score = scores[np.arange(scores.shape[0]), pred_labels]
        coverage_margin = threshold - pred_label_score
        if scores.shape[1] > 1:
            nonconformity_gap = sorted_scores[:, 1] - sorted_scores[:, 0]
        else:
            nonconformity_gap = np.ones(scores.shape[0], dtype=np.float64)
        max_extra = max(float(self.num_classes or scores.shape[1]) - 1.0, 1.0)
        size_risk = (set_size - 1.0) / max_extra
        finite_margin = np.clip(coverage_margin, 0.0, None)
        margin_risk = 1.0 - np.clip(finite_margin / max(threshold, 1e-8), 0.0, 1.0)
        abstain_risk = np.clip(0.70 * size_risk + 0.30 * margin_risk, 0.0, 1.0).astype(np.float32)
        order = np.lexsort((-pred_label_score, nonconformity_gap, coverage_margin, -set_size))
        router_score = np.zeros(scores.shape[0], dtype=np.float32)
        if order.size > 1:
            router_score[order] = np.linspace(1.0, 0.0, num=order.size, dtype=np.float32)
        elif order.size == 1:
            router_score[order[0]] = 1.0
        return {
            "prediction_sets": prediction_sets,
            "set_size": set_size.astype(np.int64).tolist(),
            "pred_label_score": pred_label_score.astype(np.float32),
            "coverage_margin": coverage_margin.astype(np.float32),
            "nonconformity_gap": nonconformity_gap.astype(np.float32),
            "abstain_risk": abstain_risk,
            "router_score": router_score.astype(np.float32),
            "router_priority_components": {
                "set_size": set_size.astype(np.int64).tolist(),
                "coverage_margin": coverage_margin.astype(np.float32),
                "nonconformity_gap": nonconformity_gap.astype(np.float32),
                "pred_label_score": pred_label_score.astype(np.float32),
                "ordering": "larger_set_size_then_smaller_margin_then_smaller_gap_then_larger_pred_label_score",
            },
        }

    def _choose_hyperparameters(
        self,
        posterior,
        labels_np,
        tune_idx_np,
        cal_idx_np,
        edge_index=None,
        edge_type=None,
        node_repr=None,
        edge_weight=None,
        direction_mode="incoming",
        relation_mode="agnostic",
    ):
        best = None
        for rho1, rho2 in self._valid_rho_grid():
            for temperature in self.temperatures:
                scores, _ = self._ego2_nonconformity(
                    posterior,
                    edge_index=edge_index,
                    edge_type=edge_type,
                    node_repr=node_repr,
                    edge_weight=edge_weight,
                    rho1=rho1,
                    rho2=rho2,
                    temperature=temperature,
                    direction_mode=direction_mode,
                    relation_mode=relation_mode,
                )
                threshold = self._threshold_for(scores, labels_np, cal_idx_np)
                if tune_idx_np.size:
                    tune_sets = scores[tune_idx_np] <= threshold + 1e-12
                    avg_size = float(tune_sets.sum(axis=1).mean())
                    coverage = float(tune_sets[np.arange(tune_idx_np.size), labels_np[tune_idx_np]].mean())
                else:
                    avg_size = float((scores <= threshold + 1e-12).sum(axis=1).mean())
                    coverage = 1.0
                feasible = coverage + 1e-12 >= (1.0 - self.alpha)
                key = (0 if feasible else 1, avg_size, -coverage, rho1 + rho2)
                record = {
                    "rho1": float(rho1),
                    "rho2": float(rho2),
                    "temperature": None if temperature is None else float(temperature),
                    "threshold": float(threshold),
                    "tune_average_set_size": avg_size,
                    "tune_empirical_coverage": coverage,
                    "coverage_constraint_satisfied": bool(feasible),
                    "selection_key": key,
                }
                if best is None or key < best["selection_key"]:
                    best = record
        return best

    def fit(self, logits, probs, labels, train_idx, val_idx, edge_index=None, edge_type=None, node_repr=None, **kwargs):
        posterior = self._posterior(logits=logits, probs=probs)
        labels_np = _labels_to_numpy(labels).astype(np.int64)
        val_idx_np = _valid_index_array(val_idx, labels_np.shape[0])
        if val_idx_np.size == 0:
            raise ValueError("gnn_2hop_conformal requires a non-empty calibration/validation split.")
        tune_idx_np = _valid_index_array(kwargs.get("tune_idx", val_idx_np), labels_np.shape[0])
        cal_idx_np = _valid_index_array(kwargs.get("cal_idx", val_idx_np), labels_np.shape[0])
        if cal_idx_np.size == 0:
            raise ValueError("gnn_2hop_conformal requires a non-empty conformal calibration split.")
        edge_weight = kwargs.get("edge_weight", None)
        direction_mode = str(kwargs.get("direction_mode", "incoming")).lower()
        relation_mode = str(kwargs.get("relation_mode", "agnostic")).lower()
        self.direction_mode = direction_mode
        self.relation_mode = relation_mode
        self.split_metadata_ = {
            "tune_count": int(tune_idx_np.size),
            "calibration_count": int(cal_idx_np.size),
            "tune_idx_sha256": self._index_sha256(tune_idx_np),
            "cal_idx_sha256": self._index_sha256(cal_idx_np),
            "tune_cal_disjoint": bool(set(tune_idx_np.tolist()).isdisjoint(set(cal_idx_np.tolist()))),
            **dict(kwargs.get("tune_cal_split_metadata", {})),
        }
        self.num_classes = int(posterior.shape[1])
        best = self._choose_hyperparameters(
            posterior,
            labels_np,
            tune_idx_np,
            cal_idx_np,
            edge_index=edge_index,
            edge_type=edge_type,
            node_repr=node_repr,
            edge_weight=edge_weight,
            direction_mode=direction_mode,
            relation_mode=relation_mode,
        )
        self.rho1 = float(best["rho1"])
        self.rho2 = float(best["rho2"])
        self.temperature = best["temperature"]
        self.threshold = float(best["threshold"])
        self.nonconformity_scores_, self.ego2_metadata_ = self._ego2_nonconformity(
            posterior,
            edge_index=edge_index,
            edge_type=edge_type,
            node_repr=node_repr,
            edge_weight=edge_weight,
            rho1=self.rho1,
            rho2=self.rho2,
            temperature=self.temperature,
            direction_mode=direction_mode,
            relation_mode=relation_mode,
        )
        self.tune_summary = dict(best)
        self.tune_summary.pop("selection_key", None)
        self.fit_summary = {
            "source": "gnn_2hop_conformal",
            "fit_scope": "valid_tune_then_valid_cal_labels_only",
            "tune_count": int(tune_idx_np.size),
            "calibration_count": int(cal_idx_np.size),
            "alpha": float(self.alpha),
            "threshold": float(self.threshold),
            "rho1": float(self.rho1),
            "rho2": float(self.rho2),
            "temperature": self.temperature,
            "snaps_similarity_used": bool(node_repr is not None and self.temperature is not None),
            "similarity_temperature": self.temperature,
            "similarity_source": "GNN node_repr from SimTeG-style GNN" if node_repr is not None else None,
            "aggregation_hops": 2,
            "score_contract": "class_conditional_2hop_nonconformity",
            "graph_context_used": edge_index is not None,
            "embedding_context_used": node_repr is not None and self.temperature is not None,
            "direction_mode": direction_mode,
            "relation_mode": relation_mode,
            "edge_weight_used": edge_weight is not None,
            "lm_gnn_disagreement_used": False,
            "prediction_set_contract": "gnn_2hop_conformal_prediction_set_v1",
            "risk_score_contract": "lexicographic_conformal_priority_v1",
            "abstain_risk_role": "legacy_compatibility_not_main_router",
            "test_labels_used_for_threshold": False,
            "rewriter_outcome_used_for_threshold": False,
            "llm_output_used": False,
            "literature_basis": ["SimTeG", "CF-GNN", "DAPS/NAPS", "SNAPS", "GATS", "CaGCN"],
            "snaps_code_reference": "SNAPS gnn_cp/cp/graph_transformations.py KHopVertexMPTransformation row-normalized hop aggregation",
            "literature_boundary": (
                "Uses GNN-side posterior and local 2-hop nonconformity aggregation; does not claim "
                "exchangeability after graph rewriting and does not use LM-GNN disagreement as a main signal."
            ),
            "feature_similarity_channel_status": (
                "active_when_temperature_is_not_null; similarity is an aggregation weight, not edge reliability"
            ),
            "hyperparameter_selection": dict(self.tune_summary),
            "tune_cal_split_metadata": dict(self.split_metadata_),
        }
        self.calibration_metadata = dict(self.fit_summary)
        self.val_threshold = None
        return self

    def score(self, logits, probs, edge_index=None, edge_type=None, node_repr=None, **kwargs):
        posterior = self._posterior(logits=logits, probs=probs)
        scores, _ = self._ego2_nonconformity(
            posterior,
            edge_index=edge_index,
            edge_type=edge_type,
            node_repr=node_repr,
            edge_weight=kwargs.get("edge_weight", None),
            rho1=self.rho1,
            rho2=self.rho2,
            temperature=self.temperature,
            direction_mode=kwargs.get("direction_mode", self.direction_mode),
            relation_mode=kwargs.get("relation_mode", self.relation_mode),
        )
        return self._prediction_set_payload_from_scores(scores)["router_score"]

    def build_manifest(self, logits, probs, labels, val_idx, test_idx, edge_index=None, edge_type=None, node_repr=None, **kwargs):
        posterior = self._posterior(logits=logits, probs=probs)
        scores, ego2_metadata = self._ego2_nonconformity(
            posterior,
            edge_index=edge_index,
            edge_type=edge_type,
            node_repr=node_repr,
            edge_weight=kwargs.get("edge_weight", None),
            rho1=self.rho1,
            rho2=self.rho2,
            temperature=self.temperature,
            direction_mode=kwargs.get("direction_mode", self.direction_mode),
            relation_mode=kwargs.get("relation_mode", self.relation_mode),
        )
        payload = self._prediction_set_payload_from_scores(scores)
        risk_score = payload["router_score"].astype(np.float32)
        labels_np = _labels_to_numpy(labels)
        labeled_count = int(labels_np.shape[0])
        preds = posterior.argmax(axis=1)
        preds_labeled = preds[:labeled_count]
        wrong = (preds_labeled != labels_np).astype(np.int32)
        val_idx_np = _valid_index_array(val_idx, wrong.shape[0])
        test_idx_np = _valid_index_array(test_idx, wrong.shape[0])
        if self.val_threshold is None:
            budget = self.budgets[0] if self.budgets else RESIDUAL_RISK_PAPER_BUDGETS[0]
            k = max(int(len(val_idx_np) * float(budget)), 1) if val_idx_np.size else 0
            self.val_threshold = float(np.sort(risk_score[val_idx_np])[-k]) if k else float("inf")
        metadata = {
            **dict(self.calibration_metadata),
            "graph_context_used": edge_index is not None,
            "embedding_context_used": node_repr is not None and self.temperature is not None,
            "threshold_source": "valid_cal_conformal_nonconformity",
            "test_labels_used_for_threshold": False,
            "lm_gnn_disagreement_used": False,
        }
        return {
            "risk_score": risk_score,
            "prediction_sets": payload["prediction_sets"],
            "set_size": payload["set_size"],
            "pred_label_score": payload["pred_label_score"],
            "coverage_margin": payload["coverage_margin"].astype(np.float32),
            "nonconformity_gap": payload["nonconformity_gap"].astype(np.float32),
            "abstain_risk": payload["abstain_risk"].astype(np.float32),
            "router_score": risk_score,
            "router_priority_components": payload["router_priority_components"],
            "ego2_metadata": ego2_metadata,
            "thresholds": {
                "validation_risk_threshold": float(self.val_threshold),
                "conformal_nonconformity_threshold": float(self.threshold if self.threshold is not None else 1.0),
            },
            "calibration_metadata": metadata,
            "hcw_mask": ((1.0 - posterior[:labeled_count].max(axis=1)) < 0.1) & (wrong == 1),
            "validation_metrics": _safe_router_metrics(wrong[val_idx_np], risk_score[val_idx_np], budgets=self.budgets) if val_idx_np.size else {},
            "test_metrics": _safe_router_metrics(wrong[test_idx_np], risk_score[test_idx_np], budgets=self.budgets) if test_idx_np.size else {},
        }

    def state_dict_payload(self):
        return {
            "alpha": float(self.alpha),
            "rho1": float(self.rho1),
            "rho2": float(self.rho2),
            "temperature": self.temperature,
            "threshold": float(self.threshold if self.threshold is not None else 1.0),
            "num_classes": self.num_classes,
            "fit_summary": self.fit_summary,
        }


class GlanceForContextResidualRiskSelector(BaseRiskEstimator):
    metadata = {
        "claim_role": "stage_b_ablation_residual_risk_estimator",
        "stage2_role": "hard_node_selector",
        "canonical_stage2": False,
        "scientific_gate": "glance_for_context_node_aware_ablation",
        "paper_identity": "GLANCE-inspired Node-Aware Residual-Risk Selector",
        "paper_identity_risk": "medium_if_framed_as_stage3_router",
        "promotion_rule": "ablation_only_never_canonical",
        "screening_only": True,
        "diagnosis_or_action": False,
        "glance_reference": "Loveland et al. 2025, Glance for Context",
    }

    def __init__(
        self,
        budgets=RESIDUAL_RISK_PAPER_BUDGETS,
        max_iter=1000,
        max_embedding_dims=32,
        max_node_feature_dims=0,
        llm_query_cost=0.2,
        training_objective="residual_error",
    ):
        super().__init__()
        self.budgets = tuple(float(item) for item in budgets)
        self.max_iter = int(max_iter)
        self.max_embedding_dims = int(max_embedding_dims)
        self.max_node_feature_dims = int(max_node_feature_dims)
        self.llm_query_cost = float(llm_query_cost)
        self.training_objective = str(training_objective).lower()
        self.scaler = None
        self.classifier = None
        self.feature_names = []
        self.structural_feature_names = []
        self.embedding_feature_names = []
        self.node_feature_names = []
        self.selected_candidate = None
        self.candidate_selection = {}
        self.candidate_metadata = {}
        self.training_summary = {}
        self.last_view_weights = None
        self.last_view_scores = None

    def _features(
        self,
        logits,
        probs,
        edge_index=None,
        edge_type=None,
        node_repr=None,
        **kwargs,
    ):
        return build_glance_for_context_router_features(
            logits=logits,
            probs=probs,
            edge_index=edge_index,
            edge_type=edge_type,
            lm_probs=kwargs.get("lm_probs"),
            q_probs=kwargs.get("q_probs"),
            node_repr=node_repr,
            node_features=kwargs.get("node_features"),
            mc_dropout_uncertainty=kwargs.get("mc_dropout_uncertainty"),
            max_embedding_dims=self.max_embedding_dims,
            max_node_feature_dims=self.max_node_feature_dims,
        )

    def _aligned_feature_matrix(self, feature_bundle):
        names = self.feature_names or list(feature_bundle["feature_names"])
        return _feature_matrix_from_map(feature_bundle["feature_map"], names, int(feature_bundle["num_nodes"]))

    def _fit_logistic_score_model(self, feature_matrix, target_labels):
        if feature_matrix.shape[0] == 0 or feature_matrix.shape[1] == 0:
            return False
        if np.unique(target_labels).size < 2:
            return False
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(feature_matrix)
        self.classifier = LogisticRegression(max_iter=self.max_iter, class_weight="balanced")
        self.classifier.fit(scaled, target_labels)
        return True

    def _fit_logistic_router(self, feature_matrix, residual_fit_labels):
        """Legacy internal alias for older GLANCE residual-risk code paths."""
        return self._fit_logistic_score_model(feature_matrix, residual_fit_labels)

    def _logistic_router_score(self, feature_matrix):
        if self.scaler is None or self.classifier is None:
            return None
        return self.classifier.predict_proba(self.scaler.transform(feature_matrix))[:, 1].astype(np.float32)

    def _build_training_target(self, labels, probs, fit_idx, **kwargs):
        objective = str(kwargs.get("training_objective", self.training_objective)).lower()
        if objective in {"residual", "residual_error", "stage2_residual_error"}:
            self.training_objective = "residual_error"
            return build_residual_error_training_target(labels=labels, probs=probs, fit_idx=fit_idx)
        if objective in {"glance_advantage", "advantage", "paper_advantage"}:
            self.training_objective = "glance_advantage"
            return build_glance_advantage_training_target(
                fit_idx=fit_idx,
                llm_query_cost=kwargs.get("llm_query_cost", self.llm_query_cost),
                gnn_loss=kwargs.get("gnn_loss"),
                llm_loss=kwargs.get("llm_loss"),
                gnn_correct=kwargs.get("gnn_correct"),
                llm_correct=kwargs.get("llm_correct"),
            )
        raise ValueError(f"Unknown GLANCE training objective: {objective}")

    def _training_objective_adaptation_note(self, training_target):
        if training_target.objective == "glance_advantage":
            return (
                "GLANCE paper-style advantage training: learns whether an explicit LLM/refiner "
                "counterfactual improves over the GNN after subtracting query cost. This is an "
                "ablation and is not the canonical Stage 2 residual-risk estimator."
            )
        return (
            "GLANCE routing features are adapted to residual-risk logistic ranking; "
            "no LLM counterfactual outcome or Stage 3 repair action is used."
        )

    def _metadata_for_candidates(self, feature_bundle, router_available):
        if self.training_objective == "glance_advantage":
            fit_scope = "explicit_glance_counterfactual_advantage_labels"
            family = "GLANCE_for_Context_advantage_router_ablation"
            screening_only = False
            diagnosis_or_action = True
        else:
            fit_scope = "train_split_oof_residual_labels_only"
            family = "GLANCE_for_Context_lightweight_router"
            screening_only = True
            diagnosis_or_action = False
        metadata = {
            **_base_residual_candidate_metadata(),
            GLANCE_SOFT_HOMOPHILY_PRIOR: {
                "status": "available",
                "family": "GLANCE_for_Context_label_free_homophily_prior",
                "definition": "1 - soft local homophily estimated from q_probs, LM probabilities, or Stage 1 probabilities",
                "formula": GLANCE_FORMULA_REFERENCE["soft_estimated_homophily"],
                "homophily_estimator_source": feature_bundle.get("homophily_estimator_source"),
            },
            GLANCE_DEGREE_HOMOPHILY_PRIOR: {
                "status": "available",
                "family": "GLANCE_for_Context_degree_homophily_prior",
                "definition": "blend of estimated heterophily, sparse degree, and low relative degree",
                "formula": GLANCE_FORMULA_REFERENCE["relative_degree"],
            },
            GLANCE_UNCERTAINTY_HOMOPHILY_BLEND: {
                "status": "available",
                "family": "GLANCE_for_Context_uncertainty_homophily_blend",
                "definition": "blend of MSP risk, uncertainty, estimated heterophily, and local inconsistency",
            },
            GLANCE_ROUTER_LOGISTIC: {
                "status": "available" if router_available else "not_available",
                "family": family,
                "formula": GLANCE_FORMULA_REFERENCE["router_probability"],
                "fit_scope": fit_scope,
                "training_objective": self.training_objective,
                "screening_only": screening_only,
                "diagnosis_or_action": diagnosis_or_action,
                "llm_query_cost_reference": float(self.llm_query_cost),
                "feature_family": list(self.feature_names or feature_bundle["feature_names"]),
                "structural_feature_family": list(feature_bundle.get("structural_feature_names", [])),
                "embedding_feature_family": list(feature_bundle.get("embedding_feature_names", [])),
                "node_feature_family": list(feature_bundle.get("node_feature_names", [])),
                "glance_reference": "cheap node-aware router features: GNN embedding, uncertainty, degree, soft homophily, and node features",
            },
        }
        return metadata

    def _candidate_scores(self, logits, probs, edge_index=None, edge_type=None, node_repr=None, **kwargs):
        feature_bundle = self._features(
            logits,
            probs,
            edge_index=edge_index,
            edge_type=edge_type,
            node_repr=node_repr,
            **kwargs,
        )
        matrix = self._aligned_feature_matrix(feature_bundle)
        candidate_scores = dict(feature_bundle["base_scores"])
        router_score = self._logistic_router_score(matrix)
        if router_score is not None:
            candidate_scores[GLANCE_ROUTER_LOGISTIC] = router_score
        metadata = self._metadata_for_candidates(feature_bundle, router_score is not None)
        return candidate_scores, metadata, feature_bundle, matrix

    def fit(self, logits, probs, labels, train_idx, val_idx, edge_index=None, edge_type=None, node_repr=None, **kwargs):
        fit_idx = kwargs.get("fit_idx", None)
        if fit_idx is None:
            raise ValueError(
                "GLANCE-inspired residual-risk selector requires OOF fit_idx; "
                "direct train correctness is not an allowed training target."
            )

        _, _, residual_errors, _ = _stage1_residual_errors(labels, probs)
        training_kwargs = dict(kwargs)
        training_kwargs.pop("fit_idx", None)
        training_target = self._build_training_target(labels, probs, fit_idx, **training_kwargs)
        fit_idx_np = training_target.fit_idx
        feature_bundle = self._features(
            logits,
            probs,
            edge_index=edge_index,
            edge_type=edge_type,
            node_repr=node_repr,
            **kwargs,
        )
        self.feature_names = list(feature_bundle["feature_names"])
        self.structural_feature_names = list(feature_bundle.get("structural_feature_names", []))
        self.embedding_feature_names = list(feature_bundle.get("embedding_feature_names", []))
        self.node_feature_names = list(feature_bundle.get("node_feature_names", []))
        feature_matrix = self._aligned_feature_matrix(feature_bundle)
        router_available = self._fit_logistic_score_model(feature_matrix[fit_idx_np], training_target.labels)
        train_score = self._logistic_router_score(feature_matrix)

        self.training_summary = {
            "source": "glance_for_context_residual_risk_selector",
            "paper_identity": "GLANCE-inspired Node-Aware Residual-Risk Selector",
            "fit_scope": "train_split_oof_artifact_only",
            "validation_selection_scope": "validation_split_residual_labels_only",
            "fit_count": int(fit_idx_np.size),
            "positive_count": int(training_target.labels.sum()) if fit_idx_np.size else 0,
            "residual_error_positive_count": int(residual_errors[fit_idx_np].sum()) if fit_idx_np.size else 0,
            "router_available": bool(router_available),
            "feature_family": list(self.feature_names),
            "structural_feature_family": list(self.structural_feature_names),
            "embedding_feature_family": list(self.embedding_feature_names),
            "node_feature_family": list(self.node_feature_names),
            "homophily_estimator_source": feature_bundle.get("homophily_estimator_source"),
            "homophily_status": feature_bundle.get("homophily_status"),
            "formula_reference": feature_bundle.get("formula_reference", dict(GLANCE_FORMULA_REFERENCE)),
            "llm_query_cost_reference": float(self.llm_query_cost),
            "training_objective": training_target.objective,
            "target_semantics": training_target.target_semantics,
            "target_metadata": dict(training_target.metadata),
            "screening_only": bool(training_target.objective != "glance_advantage"),
            "diagnosis_or_action": bool(training_target.objective == "glance_advantage"),
            "paper_faithful_glance_inspired": bool(training_target.objective == "glance_advantage"),
            "official_code_verified": False,
            "counterfactual_outcome_used": bool(training_target.metadata.get("counterfactual_outcome_used", False)),
            "not_a_new_bot_classifier": True,
            "training_objective_adaptation": self._training_objective_adaptation_note(training_target),
            "forbidden_inputs_audit": {
                "raw_text": False,
                "stage3_action_or_repair_outcome": bool(
                    training_target.metadata.get("stage3_action_or_repair_outcome_used", False)
                ),
                "llm_counterfactual_outcome": bool(
                    training_target.metadata.get("llm_counterfactual_outcome_used", False)
                ),
                "gnn_hidden_state": bool(feature_bundle.get("used_node_repr", False)),
                "node_repr_used_for_glance_ablation": bool(feature_bundle.get("used_node_repr", False)),
            },
            "oof_provenance": kwargs.get("oof_metadata", {}),
            "train_metrics": _safe_router_metrics(
                training_target.labels,
                train_score[fit_idx_np],
                budgets=self.budgets,
            )
            if router_available and fit_idx_np.size
            else {},
        }
        self.calibration_metadata = dict(self.training_summary)
        return self

    def score(self, logits, probs, edge_index=None, edge_type=None, node_repr=None, **kwargs):
        candidates, _, _, _ = self._candidate_scores(
            logits,
            probs,
            edge_index=edge_index,
            edge_type=edge_type,
            node_repr=node_repr,
            **kwargs,
        )
        selected = self.selected_candidate if self.selected_candidate in candidates else None
        if selected is None:
            selected = GLANCE_ROUTER_LOGISTIC if GLANCE_ROUTER_LOGISTIC in candidates else GLANCE_UNCERTAINTY_HOMOPHILY_BLEND
        return np.asarray(candidates[selected], dtype=np.float32)

    def build_manifest(self, logits, probs, labels, val_idx, test_idx, edge_index=None, edge_type=None, node_repr=None, **kwargs):
        _, _, residual_errors, probs_np = _stage1_residual_errors(labels, probs)
        val_idx_np = _valid_index_array(val_idx, residual_errors.shape[0])
        test_idx_np = _valid_index_array(test_idx, residual_errors.shape[0])
        candidate_scores, candidate_metadata, feature_bundle, _ = self._candidate_scores(
            logits,
            probs,
            edge_index=edge_index,
            edge_type=edge_type,
            node_repr=node_repr,
            **kwargs,
        )
        if self.training_objective == "glance_advantage" and GLANCE_ROUTER_LOGISTIC in candidate_scores:
            selection = {
                "selected_candidate": GLANCE_ROUTER_LOGISTIC,
                "candidate_order": [GLANCE_ROUTER_LOGISTIC],
                "selection_metric": "paper_faithful_counterfactual_advantage_router",
                "selection_scope": "trained_advantage_router_no_residual_error_candidate_switch",
                "note": "Advantage mode routes by the learned GLANCE-style counterfactual router, not by validation residual-risk candidate selection.",
            }
            self.selected_candidate = GLANCE_ROUTER_LOGISTIC
        else:
            selection = _select_residual_risk_candidate(
                residual_errors,
                candidate_scores,
                val_idx_np,
                budgets=self.budgets,
            )
            self.selected_candidate = selection["selected_candidate"]
        self.candidate_selection = selection
        self.candidate_metadata = candidate_metadata
        risk_score = np.asarray(candidate_scores[self.selected_candidate], dtype=np.float32)
        candidate_names, self.last_view_scores, self.last_view_weights = _one_hot_candidate_view_tensors(
            candidate_scores,
            self.selected_candidate,
        )
        if self.val_threshold is None:
            budget = max(int(len(val_idx_np) * 0.15), 1)
            self.val_threshold = float(np.sort(risk_score[val_idx_np])[-budget]) if val_idx_np.size else float("inf")
        hcw_mask = ((1.0 - probs_np.max(axis=1)) < 0.1) & (residual_errors == 1)
        calibration_metadata = {
            **dict(self.calibration_metadata),
            "selected_candidate": self.selected_candidate,
            "candidate_order": selection.get("candidate_order", []),
            "candidate_selection": selection,
            "candidate_metadata": candidate_metadata,
            "candidate_names": candidate_names,
            "input_boundary": self.training_summary.get("forbidden_inputs_audit", {}),
            "selection_scope": "validation_split_residual_labels_only",
            "screening_only": bool(self.training_objective != "glance_advantage"),
            "diagnosis_or_action": bool(self.training_objective == "glance_advantage"),
            "paper_faithful_glance_inspired": bool(self.training_objective == "glance_advantage"),
            "official_code_verified": False,
            "counterfactual_outcome_used": bool(self.training_objective == "glance_advantage"),
            "not_a_new_bot_classifier": True,
            "positioning": (
                "GLANCE-inspired counterfactual advantage router; selects refiner action utility"
                if self.training_objective == "glance_advantage"
                else "GLANCE-inspired node-aware residual-risk ranking; not a Stage 3 action router"
            ),
            "homophily_estimator_source": feature_bundle.get("homophily_estimator_source"),
        }
        self.calibration_metadata = calibration_metadata
        return {
            "risk_score": risk_score.astype(np.float32),
            "thresholds": {"validation_risk_threshold": float(self.val_threshold)},
            "calibration_metadata": calibration_metadata,
            "hcw_mask": hcw_mask.tolist(),
            "validation_metrics": _safe_router_metrics(
                residual_errors[val_idx_np],
                risk_score[val_idx_np],
                budgets=self.budgets,
            )
            if val_idx_np.size
            else {},
            "test_metrics": _safe_router_metrics(
                residual_errors[test_idx_np],
                risk_score[test_idx_np],
                budgets=self.budgets,
            )
            if test_idx_np.size
            else {},
        }

    def state_dict_payload(self):
        return {
            "feature_names": self.feature_names,
            "structural_feature_names": self.structural_feature_names,
            "embedding_feature_names": self.embedding_feature_names,
            "node_feature_names": self.node_feature_names,
            "selected_candidate": self.selected_candidate,
            "candidate_selection": self.candidate_selection,
            "candidate_metadata": self.candidate_metadata,
            "training_summary": self.training_summary,
            "llm_query_cost_reference": float(self.llm_query_cost),
            "classifier": {
                "coef": self.classifier.coef_.tolist() if self.classifier is not None else None,
                "intercept": self.classifier.intercept_.tolist() if self.classifier is not None else None,
                "classes": self.classifier.classes_.tolist() if self.classifier is not None else None,
            },
            "scaler": {
                "mean": self.scaler.mean_.tolist() if self.scaler is not None else None,
                "scale": self.scaler.scale_.tolist() if self.scaler is not None else None,
            },
        }


GlanceForContextResidualRiskRouter = GlanceForContextResidualRiskSelector


class CalibratedExpectedErrorProxy:
    """External accept-rule for the v8 EQC Stage-4 soft rewriter.

    Learns a 5-parameter logistic regression
        e_phi(f) = sigma(phi_0 + phi_1*conf + phi_2*entropy + phi_3*q + phi_4*set_size)
    on held-out validation-calibration labels ``1[y_hat(v) != y(v)]``, fit
    with 5-fold cross-fitting so the calibrator never sees its own training
    fold at prediction time. At test time, the proxy value is the mean of
    the five out-of-fold predictors.

    Crucially, this proxy is *external* to the composite score q(v): an
    increase in e_phi is a predicted increase in empirical error on the
    calibration distribution, independent of the model's own argmax
    confidence. The Stage-4 rewriter accepts a proposed rewrite only if
    ``e_phi(post) <= e_phi(pre)`` (paired with an L-infinity stability
    cap on the posterior; see EvidenceGraphRewriter).
    """

    metadata = {
        "claim_role": "eqc_v8_stage4_calibrated_accept_gate",
        "scientific_gate": "valid_cal_held_out_labels_only",
        "paper_identity_risk": "low_plumbing_not_novelty",
        "promotion_rule": "pre_committed_plumbing_never_primary_claim",
    }

    def __init__(self, n_folds=5, seed=0):
        self.n_folds = int(n_folds)
        self.seed = int(seed)
        self._fold_models = []
        self._is_fitted = False
        self.fit_summary = {}

    @staticmethod
    def _node_features(posterior, q_composite, set_size):
        """Stack ``(confidence, entropy, q_composite, set_size)`` into an
        ``[N, 4]`` feature matrix expected by the calibrator.
        """
        posterior = np.clip(np.asarray(posterior, dtype=np.float64), 1e-12, 1.0)
        confidence = posterior.max(axis=1)
        entropy = -np.sum(posterior * np.log(posterior), axis=1)
        q_composite = np.asarray(q_composite, dtype=np.float64).reshape(-1)
        set_size = np.asarray(set_size, dtype=np.float64).reshape(-1)
        return np.stack([confidence, entropy, q_composite, set_size], axis=1)

    def fit(self, posterior_val, q_composite_val, set_size_val, labels_val, preds_val):
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import KFold
        except ImportError as exc:  # pragma: no cover - sklearn is already a project dep
            raise ImportError("CalibratedExpectedErrorProxy requires scikit-learn.") from exc

        features = self._node_features(posterior_val, q_composite_val, set_size_val)
        targets = (np.asarray(preds_val, dtype=np.int64) != np.asarray(labels_val, dtype=np.int64)).astype(np.int64)
        if features.shape[0] != targets.shape[0]:
            raise ValueError("Feature rows and target rows disagree in CalibratedExpectedErrorProxy.fit.")
        if features.shape[0] < self.n_folds:
            raise ValueError(
                f"Validation calibration split too small for {self.n_folds}-fold cross-fitting."
            )

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        fold_models = []
        per_fold_brier = []
        per_fold_auroc = []
        for fold_idx, (train_fold, val_fold) in enumerate(kf.split(features)):
            fold_targets = targets[train_fold]
            unique_classes = np.unique(fold_targets)
            if unique_classes.size < 2:
                # Single-class fold: store the constant class and handle in predict.
                fold_models.append({"_constant_class": int(unique_classes[0])})
            else:
                clf = LogisticRegression(max_iter=1000, solver="lbfgs")
                clf.fit(features[train_fold], fold_targets)
                fold_models.append(clf)
            if val_fold.size:
                stored = fold_models[-1]
                if isinstance(stored, dict) and "_constant_class" in stored:
                    const_p = float(stored["_constant_class"])
                    probs_val = np.full(val_fold.size, const_p, dtype=np.float64)
                else:
                    probs_val = stored.predict_proba(features[val_fold])[:, 1]
                brier = float(np.mean((probs_val - targets[val_fold]) ** 2))
                per_fold_brier.append(brier)
                try:
                    from sklearn.metrics import roc_auc_score
                    if len(np.unique(targets[val_fold])) == 2:
                        per_fold_auroc.append(float(roc_auc_score(targets[val_fold], probs_val)))
                except Exception:  # pragma: no cover - optional metric path
                    pass

        self._fold_models = fold_models
        self._is_fitted = True
        self.fit_summary = {
            "source": "calibrated_expected_error_proxy",
            "n_folds": int(self.n_folds),
            "n_calibration_nodes": int(features.shape[0]),
            "per_fold_brier": per_fold_brier,
            "per_fold_auroc": per_fold_auroc,
            "feature_names": ["confidence", "entropy", "q_composite", "set_size"],
            "fit_scope": "validation_split_labels_only",
            "framing": "externalized_accept_rule_not_self_referential",
        }
        return self

    def predict(self, posterior, q_composite, set_size):
        if not self._is_fitted:
            raise RuntimeError("CalibratedExpectedErrorProxy.predict called before fit().")
        features = self._node_features(posterior, q_composite, set_size)
        fold_prob_arrays = []
        for model in self._fold_models:
            if isinstance(model, dict) and "_constant_class" in model:
                # Single-class fold: constant probability (0.0 if class=0, 1.0 if class=1).
                const_p = float(model["_constant_class"])
                fold_prob_arrays.append(np.full(features.shape[0], const_p, dtype=np.float64))
            else:
                fold_prob_arrays.append(model.predict_proba(features)[:, 1].astype(np.float64))
        fold_probs = np.stack(fold_prob_arrays, axis=0)
        return fold_probs.mean(axis=0).astype(np.float64)

    def state_dict_payload(self):
        return {
            "n_folds": int(self.n_folds),
            "seed": int(self.seed),
            "is_fitted": bool(self._is_fitted),
            "fit_summary": dict(self.fit_summary),
        }


def build_estimator(mode):
    if mode == "msp_ts":
        return MSPTemperatureEstimator()
    if mode == "posthoc_calibrated_ranker":
        return PostHocCalibratedRanker()
    if mode == "calibrated_local_risk_router":
        return CalibratedLocalRiskRouter()
    if mode == "conformal_knn_risk_router":
        return ConformalKNNRiskRouter()
    if mode == "login_uncertainty_router":
        return LOGINUncertaintyRouter()
    if mode == "graph_conformal_set_estimator":
        return GraphConformalSetEstimator()
    if mode == "gnn_2hop_conformal":
        return GNN2HopConformalEstimator()
    if mode in {"glance_for_context_residual_risk_selector", "glance_residual_risk_selector", "glance_router"}:
        return GlanceForContextResidualRiskSelector()
    raise ValueError(f"Unknown estimator mode: {mode}")
