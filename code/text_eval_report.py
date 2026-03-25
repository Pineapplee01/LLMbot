"""
text_eval_report.py
Aggregate multi-seed text-only matrix outputs into calibration/risk tables
and branch-selection reports.

Primary reading convention:
- `g1_plain` is the main probability baseline
- `g6_vib_edl` is the main uncertainty-aware baseline for risk analysis
- other groups are auxiliary or appendix comparisons derived from the same
  text-only experiment surface
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

# Report selection first decides which post-hoc probability view represents a
# group (raw / TS / Platt / Beta), then decides which uncertainty-like source
# best supports selective-risk analysis inside that chosen family.
PRIMARY_SELECTION_OUTPUTS = ("raw", "ts", "platt", "beta")
RISK_INTERNAL_SOURCES = ("conf_unc", "u_text", "combined_unc_legacy", "combined_unc_monotone")


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _t_critical_95(df: int) -> float:
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
        18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    if df <= 0:
        return float("nan")
    if df in table:
        return table[df]
    return 1.96


def _mean_var_std_ci(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "var": float("nan"), "std": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    mean = float(np.mean(arr))
    if n == 1:
        return {"n": 1, "mean": mean, "var": 0.0, "std": 0.0, "ci95_low": mean, "ci95_high": mean}
    var = float(np.var(arr, ddof=1))
    std = float(np.sqrt(var))
    tcrit = _t_critical_95(n - 1)
    half = float(tcrit * std / math.sqrt(n))
    return {"n": n, "mean": mean, "var": var, "std": std, "ci95_low": mean - half, "ci95_high": mean + half}


def _aggregate_rows(rows: List[Dict], key_fields: List[str], metric_fields: List[str]) -> List[Dict]:
    buckets: Dict[Tuple, List[Dict]] = defaultdict(list)
    for r in rows:
        key = tuple(r[k] for k in key_fields)
        buckets[key].append(r)

    out = []
    for key, items in buckets.items():
        row = {k: v for k, v in zip(key_fields, key)}
        row["n_seeds"] = int(len({int(x["seed"]) for x in items}))
        for m in metric_fields:
            stats = _mean_var_std_ci([float(x[m]) for x in items if m in x])
            row[f"{m}_mean"] = stats["mean"]
            row[f"{m}_var"] = stats["var"]
            row[f"{m}_std"] = stats["std"]
            row[f"{m}_ci95_low"] = stats["ci95_low"]
            row[f"{m}_ci95_high"] = stats["ci95_high"]
        out.append(row)
    return out


def _ranknorm(x: np.ndarray) -> np.ndarray:
    n = x.size
    if n <= 1:
        return np.zeros_like(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return ranks / float(n - 1)


def _fit_monotone_combiner(conf_unc: np.ndarray, u_text: np.ndarray, err: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
    n = int(err.size)
    if n < 2:
        return None
    x1 = _ranknorm(conf_unc)
    x2 = _ranknorm(u_text)
    x3 = x1 * x2
    x = np.stack([x1, x2, x3], axis=1).astype(np.float32)
    y = err.astype(np.float32).reshape(-1, 1)

    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y)
    weight_raw = torch.nn.Parameter(torch.zeros(3, 1))
    bias = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([weight_raw, bias], lr=0.5, max_iter=100)

    def closure():
        optimizer.zero_grad()
        weights = torch.nn.functional.softplus(weight_raw)
        logits = x_t @ weights + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        weights = torch.nn.functional.softplus(weight_raw).view(-1).cpu().numpy()
        bias_np = bias.view(-1).cpu().numpy()
    return {"weights": weights, "bias": bias_np}


def _apply_monotone_combiner(conf_unc: np.ndarray, u_text: np.ndarray, state: Optional[Dict[str, np.ndarray]]) -> np.ndarray:
    if state is None:
        return _ranknorm(conf_unc) + _ranknorm(u_text)
    x1 = _ranknorm(conf_unc)
    x2 = _ranknorm(u_text)
    x3 = x1 * x2
    x = np.stack([x1, x2, x3], axis=1).astype(np.float64)
    weights = np.asarray(state["weights"], dtype=np.float64).reshape(-1, 1)
    bias = float(np.asarray(state["bias"], dtype=np.float64).reshape(-1)[0])
    logits = x @ weights
    logits = logits.reshape(-1) + bias
    probs = 1.0 / (1.0 + np.exp(-logits))
    return probs.astype(np.float64)


def _aurc_from_uncertainty(error: np.ndarray, uncertainty: np.ndarray) -> float:
    if error.size == 0:
        return float("nan")
    order = np.argsort(uncertainty)  # low-uncertainty first
    sorted_err = error[order].astype(np.float64)
    cum_err = np.cumsum(sorted_err)
    cover = np.arange(1, error.size + 1, dtype=np.float64)
    risk = cum_err / cover
    return float(np.mean(risk))


def _risk_at_coverage(error: np.ndarray, uncertainty: np.ndarray, coverage: float) -> float:
    n = error.size
    if n == 0:
        return float("nan")
    k = max(1, int(math.ceil(float(coverage) * n)))
    order = np.argsort(uncertainty)
    keep = order[:k]
    return float(np.mean(error[keep]))


def _coverage_at_risk(error: np.ndarray, uncertainty: np.ndarray, risk_target: float) -> float:
    n = error.size
    if n == 0:
        return 0.0
    order = np.argsort(uncertainty)
    sorted_err = error[order].astype(np.float64)
    cum_err = np.cumsum(sorted_err)
    cover = np.arange(1, n + 1, dtype=np.float64)
    risk = cum_err / cover
    valid = np.where(risk <= float(risk_target))[0]
    if valid.size == 0:
        return 0.0
    k = int(valid[-1] + 1)
    return float(k / n)


def _safe_auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int64)
    score = np.asarray(score).astype(np.float64)
    if y_true.size == 0 or np.min(y_true) == np.max(y_true):
        return float("nan")
    return float(roc_auc_score(y_true, score))


def _build_slices(conf: np.ndarray, u_text: np.ndarray, total_log_degree: np.ndarray, graph_missing: np.ndarray) -> Dict[str, np.ndarray]:
    n = conf.size
    all_mask = np.ones(n, dtype=bool)
    try:
        q20_deg = float(np.nanquantile(total_log_degree, 0.2))
    except Exception:
        q20_deg = float("nan")
    low_degree = total_log_degree <= q20_deg if np.isfinite(q20_deg) else np.zeros(n, dtype=bool)
    missing = graph_missing >= 0.5

    q80_conf = float(np.nanquantile(conf, 0.8))
    q20_conf = float(np.nanquantile(conf, 0.2))
    q80_u = float(np.nanquantile(u_text, 0.8))
    q20_u = float(np.nanquantile(u_text, 0.2))
    conflict = ((conf >= q80_conf) & (u_text >= q80_u)) | ((conf <= q20_conf) & (u_text <= q20_u))
    return {
        "all": all_mask,
        "low_degree": low_degree,
        "graph_missing": missing,
        "uncertainty_conflict": conflict,
    }


def _per_node_path(run_dir: Path, output_name: str, seed: int, split: str = "test") -> Path:
    tag = f"_seed{seed}"
    prefix = "per_node_test_text" if split == "test" else "per_node_val_cal_text"
    if output_name == "raw":
        return run_dir / f"{prefix}{tag}.jsonl"
    return run_dir / f"{prefix}_cal_{output_name}{tag}.jsonl"


def _group_outputs_for_split(group_row: Dict, split: str) -> Dict[str, Dict]:
    if split == "val":
        outputs = group_row.get("val_outputs")
        if outputs:
            return dict(outputs)
    return dict(group_row.get("outputs", {}))


def _build_calibration_long(seed_runs: List[Dict], split: str = "test") -> List[Dict]:
    rows = []
    for record in seed_runs:
        seed = int(record["seed"])
        for g in record["rows"]:
            outputs = _group_outputs_for_split(g, split=split)
            for output_name, m in outputs.items():
                row = {
                    "seed": seed,
                    "group_id": g["group_id"],
                    "method": g.get("method", g["group_id"]),
                    "loss_mode": g.get("loss_mode", ""),
                    "output_name": output_name,
                    "metric_split": split,
                    "acc": float(m.get("acc", float("nan"))),
                    "f1": float(m.get("f1", float("nan"))),
                    "nll": float(m.get("nll", float("nan"))),
                    "brier": float(m.get("brier", float("nan"))),
                    "ece_10": float(m.get("ece_10", m.get("ece", float("nan")))),
                    "ece_15": float(m.get("ece_15", m.get("ece", float("nan")))),
                    "ece_20": float(m.get("ece_20", m.get("ece", float("nan")))),
                }
                rows.append(row)
    return rows


def _build_risk_and_slice_long(seed_runs: List[Dict], report_slices: bool, min_n_slice: int) -> Tuple[List[Dict], List[Dict]]:
    risk_rows = []
    slice_rows = []
    for record in seed_runs:
        seed = int(record["seed"])
        for g in record["rows"]:
            run_dir = Path(g.get("run_dir", Path(g["ckpt_path"]).parent))
            outputs = list(dict(g.get("outputs", {})).keys())
            for output_name in outputs:
                node_path = _per_node_path(run_dir, output_name, seed, split="test")
                val_node_path = _per_node_path(run_dir, output_name, seed, split="val_cal")
                if not node_path.exists():
                    continue
                nodes = _load_jsonl(node_path)
                if not nodes:
                    continue
                val_nodes = _load_jsonl(val_node_path) if val_node_path.exists() else []
                y = np.array([int(x["y"]) for x in nodes], dtype=np.int64)
                pred = np.array([int(x["pred_text"]) for x in nodes], dtype=np.int64)
                conf = np.array([float(x["conf_text"]) for x in nodes], dtype=np.float64)
                u_text = np.array([float(x.get("u_text", 0.0)) for x in nodes], dtype=np.float64)
                total_deg = np.array([float(x.get("total_log_degree", np.nan)) for x in nodes], dtype=np.float64)
                graph_missing = np.array([float(x.get("graph_missing", 0.0)) for x in nodes], dtype=np.float64)
                err = (pred != y).astype(np.int64)

                conf_unc = 1.0 - conf
                combined = _ranknorm(conf_unc) + _ranknorm(u_text)
                monotone_state = None
                if val_nodes:
                    val_y = np.array([int(x["y"]) for x in val_nodes], dtype=np.int64)
                    val_pred = np.array([int(x["pred_text"]) for x in val_nodes], dtype=np.int64)
                    val_conf = np.array([float(x["conf_text"]) for x in val_nodes], dtype=np.float64)
                    val_u = np.array([float(x.get("u_text", 0.0)) for x in val_nodes], dtype=np.float64)
                    val_err = (val_pred != val_y).astype(np.int64)
                    monotone_state = _fit_monotone_combiner(1.0 - val_conf, val_u, val_err)
                combined_monotone = _apply_monotone_combiner(conf_unc, u_text, monotone_state)
                src_map = {
                    "conf_unc": conf_unc,
                    "u_text": u_text,
                    "combined_unc": combined,
                    "combined_unc_legacy": combined,
                    "combined_unc_monotone": combined_monotone,
                }

                slices = _build_slices(conf=conf, u_text=u_text, total_log_degree=total_deg, graph_missing=graph_missing)
                for src_name, unc in src_map.items():
                    # all-slice risk table
                    all_mask = slices["all"]
                    e = err[all_mask]
                    u = unc[all_mask]
                    risk_rows.append({
                        "seed": seed,
                        "group_id": g["group_id"],
                        "method": g.get("method", g["group_id"]),
                        "loss_mode": g.get("loss_mode", ""),
                        "output_name": output_name,
                        "uncertainty_source": src_name,
                        "error_auroc": _safe_auroc(e, u),
                        "aurc": _aurc_from_uncertainty(e, u),
                        "risk_cov_100": _risk_at_coverage(e, u, 1.00),
                        "risk_cov_95": _risk_at_coverage(e, u, 0.95),
                        "risk_cov_90": _risk_at_coverage(e, u, 0.90),
                        "risk_cov_80": _risk_at_coverage(e, u, 0.80),
                        "cov_risk_05": _coverage_at_risk(e, u, 0.05),
                        "cov_risk_10": _coverage_at_risk(e, u, 0.10),
                    })

                    if not report_slices:
                        continue
                    for slice_name, mask in slices.items():
                        n_slice = int(mask.sum())
                        if n_slice == 0:
                            continue
                        support_status = "ok" if n_slice >= int(min_n_slice) else "insufficient_support"
                        row = {
                            "seed": seed,
                            "group_id": g["group_id"],
                            "method": g.get("method", g["group_id"]),
                            "loss_mode": g.get("loss_mode", ""),
                            "output_name": output_name,
                            "uncertainty_source": src_name,
                            "slice": slice_name,
                            "n_slice": n_slice,
                            "slice_ratio": float(mask.mean()),
                            "support_status": support_status,
                        }
                        if support_status == "ok":
                            e_s = err[mask]
                            u_s = unc[mask]
                            row.update({
                                "err_rate": float(np.mean(e_s)),
                                "error_auroc": _safe_auroc(e_s, u_s),
                                "aurc": _aurc_from_uncertainty(e_s, u_s),
                                "risk_cov_95": _risk_at_coverage(e_s, u_s, 0.95),
                                "risk_cov_90": _risk_at_coverage(e_s, u_s, 0.90),
                                "risk_cov_80": _risk_at_coverage(e_s, u_s, 0.80),
                                "cov_risk_10": _coverage_at_risk(e_s, u_s, 0.10),
                            })
                        else:
                            row.update({
                                "err_rate": float("nan"),
                                "error_auroc": float("nan"),
                                "aurc": float("nan"),
                                "risk_cov_95": float("nan"),
                                "risk_cov_90": float("nan"),
                                "risk_cov_80": float("nan"),
                                "cov_risk_10": float("nan"),
                            })
                        slice_rows.append(row)
    return risk_rows, slice_rows


def _output_tie_rank(output_name: str) -> int:
    order = ["raw", "ts", "platt", "beta", "isotonic"]
    return order.index(output_name) if output_name in order else len(order)


def _pick_best_output_per_seed(
    rows: List[Dict],
    group_id: str,
    allowed_output_names: Optional[Tuple[str, ...]] = None,
) -> Dict[int, str]:
    by_seed: Dict[int, List[Dict]] = defaultdict(list)
    for r in rows:
        if r["group_id"] != group_id:
            continue
        if allowed_output_names is not None and str(r["output_name"]) not in allowed_output_names:
            continue
        by_seed[int(r["seed"])].append(r)
    selected: Dict[int, str] = {}
    for seed, items in by_seed.items():
        best = min(
            items,
            key=lambda r: (
                float(r["nll"]),
                float(r["ece_15"]),
                float(r["brier"]),
                _output_tie_rank(str(r["output_name"])),
            ),
        )
        selected[int(seed)] = str(best["output_name"])
    return selected


def _resolve_seedwise_outputs(
    val_rows: List[Dict],
    test_rows: List[Dict],
    group_id: str,
    allowed_output_names: Optional[Tuple[str, ...]] = None,
) -> Tuple[Dict[int, str], str]:
    val_selected = _pick_best_output_per_seed(val_rows, group_id, allowed_output_names=allowed_output_names)
    test_selected = _pick_best_output_per_seed(test_rows, group_id, allowed_output_names=allowed_output_names)
    if not val_selected and not test_selected:
        return {}, "missing"
    merged: Dict[int, str] = {}
    used_fallback = False
    for seed in sorted(set(val_selected.keys()) | set(test_selected.keys())):
        if seed in val_selected:
            merged[int(seed)] = val_selected[seed]
        elif seed in test_selected:
            merged[int(seed)] = test_selected[seed]
            used_fallback = True
    return merged, ("val_with_test_fallback" if used_fallback else "val")


def _paired_delta_ci(seed_to_a: Dict[int, float], seed_to_b: Dict[int, float]) -> Dict[str, float]:
    common = sorted(set(seed_to_a.keys()) & set(seed_to_b.keys()))
    if not common:
        return {"n": 0, "mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "significant": False}
    diffs = np.array([float(seed_to_b[s] - seed_to_a[s]) for s in common], dtype=np.float64)
    stats = _mean_var_std_ci(diffs.tolist())
    if int(stats["n"]) < 2:
        return {
            "n": int(stats["n"]),
            "mean": float(stats["mean"]),
            "var": float(stats["var"]),
            "std": float(stats["std"]),
            "ci95_low": float(stats["ci95_low"]),
            "ci95_high": float(stats["ci95_high"]),
            "significant": False,
        }
    return {
        "n": int(stats["n"]),
        "mean": float(stats["mean"]),
        "var": float(stats["var"]),
        "std": float(stats["std"]),
        "ci95_low": float(stats["ci95_low"]),
        "ci95_high": float(stats["ci95_high"]),
        "significant": bool((stats["ci95_low"] > 0) or (stats["ci95_high"] < 0)),
    }


def _selected_rows_for_group(cal_rows: List[Dict], group_id: str, selected_outputs: Dict[int, str]) -> List[Dict]:
    by_key = {
        (int(r["seed"]), str(r["output_name"])): r
        for r in cal_rows
        if r["group_id"] == group_id
    }
    rows = []
    for seed, output_name in sorted(selected_outputs.items()):
        row = by_key.get((int(seed), str(output_name)))
        if row is not None:
            rows.append(row)
    return rows


def _summarize_candidate(
    cal_rows: List[Dict],
    group_id: str,
    selected_outputs: Dict[int, str],
    metrics: List[str],
) -> Dict[str, Any]:
    rows = _selected_rows_for_group(cal_rows, group_id, selected_outputs)
    out: Dict[str, Dict[str, float]] = {}
    for m in metrics:
        out[m] = _mean_var_std_ci([float(r[m]) for r in rows])
    counts = defaultdict(int)
    for output_name in selected_outputs.values():
        counts[str(output_name)] += 1
    return {
        "metrics": out,
        "selected_outputs_by_seed": [
            {"seed": int(seed), "output_name": str(output_name)}
            for seed, output_name in sorted(selected_outputs.items())
        ],
        "selected_output_counts": dict(sorted(counts.items())),
    }


def _annotate_slice_support(slice_rows: List[Dict], min_seed_support: int) -> List[Dict]:
    support_counts: Dict[Tuple[str, str, str, str], int] = defaultdict(int)
    for row in slice_rows:
        if row.get("support_status") == "ok":
            key = (
                str(row["group_id"]),
                str(row["output_name"]),
                str(row["uncertainty_source"]),
                str(row["slice"]),
            )
            support_counts[key] += 1
    out = []
    for row in slice_rows:
        key = (
            str(row["group_id"]),
            str(row["output_name"]),
            str(row["uncertainty_source"]),
            str(row["slice"]),
        )
        seed_support = int(support_counts.get(key, 0))
        row_copy = dict(row)
        row_copy["seed_support"] = seed_support
        row_copy["min_seed_support"] = int(min_seed_support)
        row_copy["included_in_main"] = int(
            row_copy.get("support_status") == "ok" and seed_support >= int(min_seed_support)
        )
        out.append(row_copy)
    return out


def _build_incremental_risk_rows(risk_rows: List[Dict]) -> List[Dict]:
    by_key: Dict[Tuple[int, str, str], Dict[str, Dict]] = defaultdict(dict)
    for row in risk_rows:
        key = (int(row["seed"]), str(row["group_id"]), str(row["output_name"]))
        by_key[key][str(row["uncertainty_source"])] = row

    delta_rows: List[Dict] = []
    for (seed, group_id, output_name), src_map in by_key.items():
        baseline = src_map.get("conf_unc", None)
        if baseline is None:
            continue
        for src_name in ("u_text", "combined_unc_legacy", "combined_unc_monotone"):
            row = src_map.get(src_name, None)
            if row is None:
                continue
            delta_rows.append({
                "seed": seed,
                "group_id": group_id,
                "method": row.get("method", group_id),
                "loss_mode": row.get("loss_mode", ""),
                "output_name": output_name,
                "comparison_source": src_name,
                "baseline_source": "conf_unc",
                "source_error_auroc": float(row["error_auroc"]),
                "baseline_error_auroc": float(baseline["error_auroc"]),
                "delta_error_auroc": float(row["error_auroc"]) - float(baseline["error_auroc"]),
                "source_aurc": float(row["aurc"]),
                "baseline_aurc": float(baseline["aurc"]),
                "delta_aurc": float(row["aurc"]) - float(baseline["aurc"]),
            })
    return delta_rows


def _aggregate_incremental_risk(delta_rows: List[Dict]) -> List[Dict]:
    key_fields = ["group_id", "method", "loss_mode", "output_name", "comparison_source", "baseline_source"]
    metric_fields = [
        "source_error_auroc",
        "baseline_error_auroc",
        "delta_error_auroc",
        "source_aurc",
        "baseline_aurc",
        "delta_aurc",
    ]
    return _aggregate_rows(delta_rows, key_fields=key_fields, metric_fields=metric_fields)


def _extract_group_rows(seed_runs: List[Dict], group_id: str) -> List[Dict]:
    rows = []
    for record in seed_runs:
        seed = int(record["seed"])
        for row in record["rows"]:
            if row["group_id"] == group_id:
                row_copy = dict(row)
                row_copy["seed"] = seed
                rows.append(row_copy)
                break
    return rows


def _build_branch_selection_report(
    cal_val_rows: List[Dict],
    cal_test_rows: List[Dict],
    risk_rows: List[Dict],
    seed_runs: List[Dict],
) -> Dict:
    # Probability branch:
    # Prefer the simpler `g1_plain` unless the classwise alternative shows a
    # seed-paired test NLL win after per-seed validation-based output selection.
    g_prob_a, g_prob_b = "g1_plain", "g9_classwise_decoupled"
    out_a, src_a = _resolve_seedwise_outputs(
        cal_val_rows,
        cal_test_rows,
        g_prob_a,
        allowed_output_names=PRIMARY_SELECTION_OUTPUTS,
    )
    out_b, src_b = _resolve_seedwise_outputs(
        cal_val_rows,
        cal_test_rows,
        g_prob_b,
        allowed_output_names=PRIMARY_SELECTION_OUTPUTS,
    )
    prob_metrics = ["nll", "ece_15", "brier"]
    prob_a_stats = _summarize_candidate(cal_test_rows, g_prob_a, out_a, prob_metrics)
    prob_b_stats = _summarize_candidate(cal_test_rows, g_prob_b, out_b, prob_metrics)

    seed_to_a_nll = {int(r["seed"]): float(r["nll"]) for r in _selected_rows_for_group(cal_test_rows, g_prob_a, out_a)}
    seed_to_b_nll = {int(r["seed"]): float(r["nll"]) for r in _selected_rows_for_group(cal_test_rows, g_prob_b, out_b)}
    delta_prob_primary = _paired_delta_ci(seed_to_a_nll, seed_to_b_nll)  # b-a; negative favors g4
    if delta_prob_primary["significant"]:
        selected_prob = g_prob_b if delta_prob_primary["mean"] < 0 else g_prob_a
    else:
        selected_prob = g_prob_a  # simplicity tie-break

    # Risk branch:
    # Hold the backbone fixed at `g6_vib_edl/raw`, then ask which uncertainty
    # source best supports selective-risk ranking.
    risk_backbone = "g6_vib_edl"
    risk_output = "raw"
    source_rows = [
        r for r in risk_rows
        if r["group_id"] == risk_backbone
        and r["output_name"] == risk_output
        and r["uncertainty_source"] in RISK_INTERNAL_SOURCES
    ]
    source_stats = {}
    source_seed_maps: Dict[str, Dict[str, Dict[int, float]]] = {}
    for src_name in RISK_INTERNAL_SOURCES:
        src_subset = [r for r in source_rows if r["uncertainty_source"] == src_name]
        seed_to_aurc = {int(r["seed"]): float(r["aurc"]) for r in src_subset}
        seed_to_auroc = {int(r["seed"]): float(r["error_auroc"]) for r in src_subset}
        source_seed_maps[src_name] = {"aurc": seed_to_aurc, "error_auroc": seed_to_auroc}
        source_stats[src_name] = {
            "aurc": _mean_var_std_ci(seed_to_aurc.values()),
            "error_auroc": _mean_var_std_ci(seed_to_auroc.values()),
        }
    baseline_source = "conf_unc"
    incremental_deltas = {}
    for src_name in ("u_text", "combined_unc_legacy", "combined_unc_monotone"):
        incremental_deltas[src_name] = {
            "delta_aurc_vs_conf_unc": _paired_delta_ci(
                source_seed_maps[baseline_source]["aurc"],
                source_seed_maps[src_name]["aurc"],
            ),
            "delta_error_auroc_vs_conf_unc": _paired_delta_ci(
                source_seed_maps[baseline_source]["error_auroc"],
                source_seed_maps[src_name]["error_auroc"],
            ),
        }
    selected_risk_source = baseline_source
    if incremental_deltas["u_text"]["delta_aurc_vs_conf_unc"]["significant"] and incremental_deltas["u_text"]["delta_aurc_vs_conf_unc"]["mean"] < 0:
        selected_risk_source = "u_text"
    elif incremental_deltas["combined_unc_monotone"]["delta_aurc_vs_conf_unc"]["significant"] and incremental_deltas["combined_unc_monotone"]["delta_aurc_vs_conf_unc"]["mean"] < 0:
        selected_risk_source = "combined_unc_monotone"
    elif incremental_deltas["combined_unc_legacy"]["delta_aurc_vs_conf_unc"]["significant"] and incremental_deltas["combined_unc_legacy"]["delta_aurc_vs_conf_unc"]["mean"] < 0:
        selected_risk_source = "combined_unc_legacy"

    # Auxiliary post-hoc view check:
    # raw g6 and calibrated g7 are two views of the same backbone, so this is
    # reported as a consistency check rather than a backbone comparison.
    g_risk_view_a, g_risk_view_b = "g6_vib_edl", "g7_vib_edl_ts"
    risk_group_rows_a = _extract_group_rows(seed_runs, g_risk_view_a)
    risk_group_rows_b = _extract_group_rows(seed_runs, g_risk_view_b)
    out_risk_a = {int(r["seed"]): str(r.get("selected_output", "raw")) for r in risk_group_rows_a}
    out_risk_b = {int(r["seed"]): str(r.get("selected_output", "ts")) for r in risk_group_rows_b}
    risk_view_filt = [
        r for r in risk_rows
        if r["uncertainty_source"] == "combined_unc_legacy"
        and (
            (r["group_id"] == g_risk_view_a and out_risk_a.get(int(r["seed"])) == r["output_name"])
            or (r["group_id"] == g_risk_view_b and out_risk_b.get(int(r["seed"])) == r["output_name"])
        )
    ]
    seed_to_view_a_aurc = {int(r["seed"]): float(r["aurc"]) for r in risk_view_filt if r["group_id"] == g_risk_view_a}
    seed_to_view_b_aurc = {int(r["seed"]): float(r["aurc"]) for r in risk_view_filt if r["group_id"] == g_risk_view_b}
    seed_to_view_a_auroc = {int(r["seed"]): float(r["error_auroc"]) for r in risk_view_filt if r["group_id"] == g_risk_view_a}
    seed_to_view_b_auroc = {int(r["seed"]): float(r["error_auroc"]) for r in risk_view_filt if r["group_id"] == g_risk_view_b}
    posthoc_view_check = {
        "comparison_type": "raw_vs_posthoc_view_of_same_backbone",
        "shared_backbone": True,
        "delta_aurc": {"metric": "aurc", **_paired_delta_ci(seed_to_view_a_aurc, seed_to_view_b_aurc)},
        "delta_error_auroc": {"metric": "error_auroc", **_paired_delta_ci(seed_to_view_a_auroc, seed_to_view_b_auroc)},
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "probability_primary_metric": "nll",
            "probability_secondary_metrics": ["ece_15", "brier"],
            "risk_primary_metric": "aurc",
            "risk_secondary_metric": "error_auroc",
            "risk_backbone": risk_backbone,
            "risk_output_name": risk_output,
            "risk_uncertainty_sources": list(RISK_INTERNAL_SOURCES),
            "ci_level": 0.95,
            "primary_selection_outputs": list(PRIMARY_SELECTION_OUTPUTS),
        },
        "probability_branch": {
            "candidates": {
                g_prob_a: {
                    "selection_source": src_a,
                    **prob_a_stats,
                },
                g_prob_b: {
                    "selection_source": src_b,
                    **prob_b_stats,
                },
            },
            "delta_b_minus_a": {"metric": "nll", **delta_prob_primary},
            "selected": selected_prob,
            "selection_rule": "Select output per seed on validation NLL/ECE/Brier over raw/ts/platt/beta, then compare test NLL with paired 95% CI; if CI crosses 0, choose simpler g1_plain.",
        },
        "risk_branch": {
            "backbone": risk_backbone,
            "output_name": risk_output,
            "baseline_source": baseline_source,
            "candidate_sources": {
                src_name: {"metrics": source_stats[src_name]}
                for src_name in RISK_INTERNAL_SOURCES
            },
            "incremental_deltas_vs_conf_unc": incremental_deltas,
            "selected_source": selected_risk_source,
            "selection_rule": "Within g6_vib_edl/raw, compare conf_unc, u_text, combined_unc_legacy, and combined_unc_monotone on AURC first and error_AUROC second; keep conf_unc unless another source shows a significant AURC improvement.",
            "posthoc_view_check": {
                **posthoc_view_check,
                "note": "g7_vib_edl_ts remains a post-hoc calibrated view of g6_vib_edl and is reported only as an auxiliary raw-vs-posthoc consistency check.",
            },
        },
    }


def generate_text_eval_report(
    ckpt_root: Path,
    seeds: List[int],
    report_slices: bool = True,
    min_n_slice: int = 30,
    min_seed_support: int = 3,
) -> Dict:
    seed_runs = []
    for seed in seeds:
        full_path = ckpt_root / f"text_matrix_seed{seed}_full.json"
        if not full_path.exists():
            continue
        seed_runs.append({"seed": int(seed), "rows": _load_json(full_path)})
    if not seed_runs:
        raise FileNotFoundError(f"No *_full matrix files found under {ckpt_root}")

    cal_test_long = _build_calibration_long(seed_runs, split="test")
    cal_val_long = _build_calibration_long(seed_runs, split="val")
    cal_agg = _aggregate_rows(
        cal_test_long,
        key_fields=["group_id", "method", "loss_mode", "output_name"],
        metric_fields=["acc", "f1", "nll", "brier", "ece_10", "ece_15", "ece_20"],
    )

    risk_long, slice_long = _build_risk_and_slice_long(
        seed_runs,
        report_slices=report_slices,
        min_n_slice=min_n_slice,
    )
    slice_long = _annotate_slice_support(slice_long, min_seed_support=min_seed_support)
    risk_agg = _aggregate_rows(
        risk_long,
        key_fields=["group_id", "method", "loss_mode", "output_name", "uncertainty_source"],
        metric_fields=[
            "error_auroc", "aurc", "risk_cov_100", "risk_cov_95", "risk_cov_90",
            "risk_cov_80", "cov_risk_05", "cov_risk_10",
        ],
    )
    incremental_risk_long = _build_incremental_risk_rows(risk_long)
    incremental_risk_agg = _aggregate_incremental_risk(incremental_risk_long)

    branch_report = _build_branch_selection_report(cal_val_long, cal_test_long, risk_long, seed_runs)

    out_cal = ckpt_root / "text_calibration_table.csv"
    out_risk = ckpt_root / "text_risk_table.csv"
    out_inc_risk = ckpt_root / "incremental_risk_table.csv"
    out_slice = ckpt_root / "text_slice_report.csv"
    out_sel = ckpt_root / "text_branch_selection_report.json"
    _write_csv(out_cal, cal_agg)
    _write_csv(out_risk, risk_agg)
    _write_csv(out_inc_risk, incremental_risk_agg)
    _write_csv(out_slice, slice_long)
    with open(out_sel, "w", encoding="utf-8") as f:
        json.dump(branch_report, f, indent=2, ensure_ascii=False)

    return {
        "calibration_table": str(out_cal),
        "risk_table": str(out_risk),
        "incremental_risk_table": str(out_inc_risk),
        "slice_report": str(out_slice),
        "selection_report": str(out_sel),
        "n_seeds": int(len(seed_runs)),
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate text-only matrix reports")
    parser.add_argument("--ckpt_root", type=str, required=True)
    parser.add_argument("--seed_start", type=int, default=42)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--report_slices", action="store_true")
    parser.add_argument("--min_n_slice", type=int, default=30)
    parser.add_argument("--min_seed_support", type=int, default=3)
    args = parser.parse_args()

    seeds = [args.seed_start + i for i in range(max(1, args.num_seeds))]
    report = generate_text_eval_report(
        ckpt_root=Path(args.ckpt_root),
        seeds=seeds,
        report_slices=args.report_slices,
        min_n_slice=args.min_n_slice,
        min_seed_support=args.min_seed_support,
    )
    print("Generated report artifacts:")
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
