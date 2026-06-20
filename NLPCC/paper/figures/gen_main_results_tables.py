from __future__ import annotations

import csv
from pathlib import Path


FIG_DIR = Path(__file__).resolve().parent
DATA_DIR = FIG_DIR / "data"
INPUT_CSV = DATA_DIR / "main_results.csv"

COMPACT_COLUMNS = [
    ("acc", "Accuracy"),
    ("macro_p", "Precision"),
    ("macro_f1", "F1-score"),
]

FULL_COLUMNS = [
    ("acc", "Accuracy"),
    ("macro_f1", "F1-score"),
    ("binary_f1", "Bot-F1"),
    ("bot_p", "Bot-Precision"),
    ("bot_r", "Bot-Recall"),
]

METHOD_GROUPS = [
    ("Feature-based", ["BotHunter", "FriendBot"]),
    ("Graph/multimodal", ["BotRGCN", "RGT", "BIC", "BotMoE", "SEBot"]),
    ("Reliability-oriented", ["BotBR"]),
    ("Graph distillation", ["LMBot-GNN", "LMBot-LM"]),
    ("Ours", ["Ours"]),
]

METHOD_LABELS = {
    "BotHunter": r"BotHunter~\cite{bot_hunter2018}",
    "FriendBot": r"FriendBot~\cite{friendbot2020}",
    "BotRGCN": r"BotRGCN~\cite{feng2021botrgcn}",
    "RGT": r"RGT~\cite{feng2022rgt}",
    "BIC": r"BIC~\cite{lei2023bic}",
    "BotMoE": r"BotMoE~\cite{liu2023botmoe}",
    "SEBot": r"SEBot~\cite{yang2024sebot}",
    "BotBR": r"BotBR~\cite{lin2025botbr}",
    "LMBot-GNN": r"LMBot-GNN~\cite{cai2024lmbot}",
    "LMBot-LM": r"LMBot-LM~\cite{cai2024lmbot}",
    "Ours": r"\textbf{BotRHG}",
}


def _load_rows() -> list[dict[str, str]]:
    with INPUT_CSV.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _ordered_groups(rows: list[dict[str, str]]) -> list[tuple[str, list[str]]]:
    available = {row["method"] for row in rows}
    grouped: list[tuple[str, list[str]]] = []
    grouped_methods: set[str] = set()
    for category, methods in METHOD_GROUPS:
        present = [method for method in methods if method in available]
        if present:
            grouped.append((category, present))
            grouped_methods.update(present)
    unknown = []
    for row in rows:
        method = row["method"]
        if method not in grouped_methods and method not in unknown:
            unknown.append(method)
    if unknown:
        grouped.append(("Other", unknown))
    return grouped


def _fmt(mean: str, std: str, rank: str | None = None) -> str:
    if not mean:
        return "--"
    body = f"{float(mean) * 100:.2f}"
    if std:
        body = rf"{body} $\pm$ {float(std) * 100:.2f}"
    if rank == "best":
        return rf"\textbf{{{body}}}"
    if rank == "second":
        return rf"\underline{{{body}}}"
    return body


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _rank_by_dataset(rows: list[dict[str, str]], dataset: str, metric: str) -> dict[str, str]:
    values: list[tuple[float, int, str]] = []
    dataset_rows = [row for row in rows if row["dataset"] == dataset]
    for index, row in enumerate(dataset_rows):
        value = row[f"{metric}_mean"]
        if value:
            values.append((float(value), index, row["method"]))
    values.sort(key=lambda item: (-item[0], item[1]))
    ranks: dict[str, str] = {}
    if values:
        ranks[values[0][2]] = "best"
    if len(values) > 1:
        ranks[values[1][2]] = "second"
    return ranks


def _rank_maps(rows: list[dict[str, str]], datasets: list[str], columns: list[tuple[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {
        (dataset, metric): _rank_by_dataset(rows, dataset, metric)
        for dataset in datasets
        for metric, _ in columns
    }


def _best_by_dataset(rows: list[dict[str, str]], dataset: str, metric: str) -> float | None:
    values: list[float] = []
    for row in rows:
        if row["dataset"] != dataset:
            continue
        value = row[f"{metric}_mean"]
        if value:
            values.append(float(value))
    return max(values) if values else None


def _write_combined_main_table() -> None:
    rows = _load_rows()
    datasets = ["TwiBot-20", "TwiBot-22 sampled"]
    groups = _ordered_groups(rows)

    by_dataset_method = {(row["dataset"], row["method"]): row for row in rows}
    ranks = _rank_maps(rows, datasets, COMPACT_COLUMNS)
    lines = [
        r"\begin{table}[t]",
        r"\caption{Main results on TwiBot-20 and sampled TwiBot-22. Values are reported as percentages. Bold and underlined values indicate the best and second-best results, respectively. A dash denotes unavailable compatible results.}",
        r"\label{tab:main-results}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{m{2.2cm}|cccccc}",
        r"\toprule",
        r"Method & \multicolumn{3}{c}{TwiBot-20} & \multicolumn{3}{c}{TwiBot-22} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        " & " + " & ".join([label for _, label in COMPACT_COLUMNS] * len(datasets)) + r" \\",
        r"\midrule",
    ]
    for group_index, (category, methods) in enumerate(groups):
        if group_index:
            lines.append(r"\midrule")
        for method_index, method in enumerate(methods):
            method_rows = [row for row in rows if row["method"] == method]
            name = _method_label(method)
            cells = []
            for dataset in datasets:
                row = by_dataset_method.get((dataset, method))
                for metric, _ in COMPACT_COLUMNS:
                    if row is None:
                        cells.append("--")
                        continue
                    value = row[f"{metric}_mean"]
                    rank = ranks[(dataset, metric)].get(method)
                    cells.append(_fmt(value, row[f"{metric}_std"], rank=rank))
            lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""])
    (DATA_DIR / "main_results_combined_table.tex").write_text("\n".join(lines), encoding="utf-8")


def _write_compact_dataset_table(dataset: str, label_suffix: str, caption: str) -> None:
    rows = [row for row in _load_rows() if row["dataset"] == dataset]
    groups = _ordered_groups(rows)
    ranks = _rank_maps(rows, [dataset], COMPACT_COLUMNS)
    by_method = {row["method"]: row for row in rows}
    lines = [
        r"\begin{table}[t]",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:main-results-{label_suffix}}}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{m{2.2cm}|ccc}",
        r"\toprule",
        "Method & " + " & ".join(label for _, label in COMPACT_COLUMNS) + r" \\",
        r"\midrule",
    ]
    for group_index, (category, methods) in enumerate(groups):
        if group_index:
            lines.append(r"\midrule")
        for method_index, method in enumerate(methods):
            row = by_method[method]
            metric_cells = []
            for metric, _ in COMPACT_COLUMNS:
                value = row[f"{metric}_mean"]
                rank = ranks[(dataset, metric)].get(row["method"])
                metric_cells.append(_fmt(value, row[f"{metric}_std"], rank=rank))
            name = _method_label(row["method"])
            lines.append(f"{name} & " + " & ".join(metric_cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""])
    (DATA_DIR / f"main_results_{label_suffix}_compact_table.tex").write_text("\n".join(lines), encoding="utf-8")


def _write_full_twibot20_table() -> None:
    dataset = "TwiBot-20"
    rows = [row for row in _load_rows() if row["dataset"] == dataset]
    groups = _ordered_groups(rows)
    ranks = _rank_maps(rows, [dataset], FULL_COLUMNS)
    by_method = {row["method"]: row for row in rows}
    lines = [
        r"\begin{table}[t]",
        r"\caption{Detailed main results on TwiBot-20. Values are reported as percentages. Bold and underlined values indicate the best and second-best results, respectively.}",
        r"\label{tab:main-results-twibot20-full}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{m{2.2cm}|ccccc}",
        r"\toprule",
        "Method & " + " & ".join(label for _, label in FULL_COLUMNS) + r" \\",
        r"\midrule",
    ]
    for group_index, (category, methods) in enumerate(groups):
        if group_index:
            lines.append(r"\midrule")
        for method_index, method in enumerate(methods):
            row = by_method[method]
            metric_cells = []
            for metric, _ in FULL_COLUMNS:
                value = row[f"{metric}_mean"]
                rank = ranks[(dataset, metric)].get(row["method"])
                metric_cells.append(_fmt(value, row[f"{metric}_std"], rank=rank))
            name = _method_label(row["method"])
            lines.append(f"{name} & " + " & ".join(metric_cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""])
    (DATA_DIR / "main_results_twibot20_full_table.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _write_combined_main_table()
    _write_compact_dataset_table(
        "TwiBot-20",
        "twibot20",
        "Main results on TwiBot-20. Values are reported as percentages. Bold and underlined values indicate the best and second-best results, respectively.",
    )
    _write_compact_dataset_table(
        "TwiBot-22 sampled",
        "twibot22",
        "Main results on sampled TwiBot-22. Values are reported as percentages. Bold and underlined values indicate the best and second-best results, respectively.",
    )
    _write_full_twibot20_table()
    print(f"Read {INPUT_CSV}")
    print(f"Wrote {DATA_DIR / 'main_results_combined_table.tex'}")
    print(f"Wrote {DATA_DIR / 'main_results_twibot20_compact_table.tex'}")
    print(f"Wrote {DATA_DIR / 'main_results_twibot22_compact_table.tex'}")
    print(f"Wrote {DATA_DIR / 'main_results_twibot20_full_table.tex'}")


if __name__ == "__main__":
    main()
