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
        preds = posterior.argmax(axis=1)
        wrong = (preds != labels_np).astype(np.int32)
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
            "hcw_mask": ((1.0 - posterior.max(axis=1)) < 0.1) & (wrong == 1),
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
        preds = posterior.argmax(axis=1)
        pred_label_score = posterior[np.arange(posterior.shape[0]), preds]
        wrong = (preds != labels_np).astype(np.int32)
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
            "hcw_mask": ((1.0 - posterior.max(axis=1)) < 0.1) & (wrong == 1),
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
        preds = posterior.argmax(axis=1)
        wrong = (preds != labels_np).astype(np.int32)
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
            "hcw_mask": ((1.0 - posterior.max(axis=1)) < 0.1) & (wrong == 1),
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
    if mode == "login_uncertainty_router":
        return LOGINUncertaintyRouter()
    if mode == "graph_conformal_set_estimator":
        return GraphConformalSetEstimator()
    if mode == "gnn_2hop_conformal":
        return GNN2HopConformalEstimator()
    if mode in {"glance_for_context_residual_risk_selector", "glance_residual_risk_selector", "glance_router"}:
        return GlanceForContextResidualRiskSelector()
    raise ValueError(f"Unknown estimator mode: {mode}")
