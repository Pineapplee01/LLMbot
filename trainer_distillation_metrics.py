import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def _compute_budget_curve(labels, base_pred, new_pred, risk_score, eval_mask, hcw_mask=None, budgets=(0.05, 0.10, 0.15, 0.20)):
    labels_np = np.asarray(labels)
    base_pred = np.asarray(base_pred)
    new_pred = np.asarray(new_pred)
    risk_score = np.asarray(risk_score, dtype=np.float32)
    eval_mask = np.asarray(eval_mask, dtype=bool)
    hcw_mask = np.asarray(hcw_mask, dtype=bool) if hcw_mask is not None else np.zeros_like(eval_mask, dtype=bool)

    candidate_idx = np.flatnonzero(eval_mask)
    if candidate_idx.size == 0:
        return []
    ranked_idx = candidate_idx[np.argsort(-risk_score[candidate_idx])]
    base_correct = base_pred == labels_np
    new_correct = new_pred == labels_np

    rows = []
    for budget in budgets:
        k = max(int(candidate_idx.size * budget), 1)
        current_idx = ranked_idx[:k]
        current_mask = np.zeros_like(eval_mask, dtype=bool)
        current_mask[current_idx] = True
        fix = int((~base_correct & new_correct & current_mask).sum())
        broke = int((base_correct & ~new_correct & current_mask).sum())
        net = fix - broke
        touched = int(current_mask.sum())
        wrong_rate = float((~base_correct[current_idx]).mean()) if current_idx.size else 0.0
        screened_acc = float(new_correct[current_idx].mean()) if current_idx.size else 0.0
        hcw_hits = int((hcw_mask & current_mask).sum())
        hcw_total = int((hcw_mask & eval_mask).sum())
        rows.append(
            {
                "budget": float(budget),
                "triggered": int(current_idx.size),
                "touched": touched,
                "fix": fix,
                "broke": broke,
                "net": net,
                "utility": wrong_rate,
                "screened_set_accuracy": screened_acc,
                "hcw_capture": float(hcw_hits / max(hcw_total, 1)),
            }
        )
    return rows


def _paired_bootstrap_delta(labels, base_pred, new_pred, eval_mask, n_samples=512, seed=0):
    labels_np = np.asarray(labels)
    base_pred = np.asarray(base_pred)
    new_pred = np.asarray(new_pred)
    eval_idx = np.flatnonzero(np.asarray(eval_mask, dtype=bool))
    if eval_idx.size == 0:
        return {}

    rng = np.random.default_rng(seed)
    macro_delta = []
    acc_delta = []
    bot_delta = []
    for _ in range(int(n_samples)):
        sample_idx = rng.choice(eval_idx, size=eval_idx.size, replace=True)
        y = labels_np[sample_idx]
        base_s = base_pred[sample_idx]
        new_s = new_pred[sample_idx]
        acc_delta.append(accuracy_score(y, new_s) - accuracy_score(y, base_s))
        macro_delta.append(
            f1_score(y, new_s, average="macro", zero_division=0) - f1_score(y, base_s, average="macro", zero_division=0)
        )
        bot_delta.append(
            f1_score(y, new_s, average="binary", zero_division=0) - f1_score(y, base_s, average="binary", zero_division=0)
        )

    def _ci(values):
        values = np.asarray(values, dtype=np.float32)
        return {
            "mean": float(values.mean()),
            "p05": float(np.quantile(values, 0.05)),
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
        }

    return {
        "accuracy_delta": _ci(acc_delta),
        "macro_f1_delta": _ci(macro_delta),
        "bot_f1_delta": _ci(bot_delta),
    }


def _empty_gain_cost_report():
    return {
        "peak_gpu_memory": None,
        "wall_clock_time": None,
        "trainable_parameter_count": None,
        "per_triggered_node_latency": None,
        "embedding_cache_size": None,
        "precompute_cost": None,
    }
