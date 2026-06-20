from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import apply_publication_style, save_fig_bundle


FIG_DIR = Path(__file__).resolve().parent
DATA_DIR = FIG_DIR / "data"
OUTPUT_CSV = DATA_DIR / "twibot20_component_ablation.csv"
OUTPUT_TEX = DATA_DIR / "twibot20_component_ablation_table.tex"


def _load_csv_rows() -> list[dict[str, str]]:
    with OUTPUT_CSV.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _reported_rows() -> list[dict[str, str]]:
    return [row for row in _load_csv_rows() if row["reported"].lower() == "yes"]


def _tex(text: str) -> str:
    return text.replace("->", r"$\rightarrow$").replace("%", r"\%").replace("w/o", r"w/o")


def write_latex_table() -> None:
    rows = _load_csv_rows()
    lines = [
        r"\begin{table}[t]",
        r"\caption{Ablation study on TwiBot-20. The full model uses $K=8$, fanout 64, and a routed budget of 10\%.}",
        r"\label{tab:component-ablation}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Variant & Acc. & F1 \\",
        r"\midrule",
    ]
    for row in rows:
        acc = row["acc"] if row["acc"] else "--"
        f1 = row["f1"] if row["f1"] else "--"
        name = _tex(row["variant"])
        if row["is_full"].lower() == "yes":
            acc = rf"\textbf{{{acc}}}"
            f1 = rf"\textbf{{{f1}}}"
        lines.append(f"{name} & {acc} & {f1} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    OUTPUT_TEX.write_text("\n".join(lines), encoding="utf-8")


def plot() -> None:
    apply_publication_style()
    rows = _reported_rows()
    labels = [row["figure_label"] for row in rows]
    acc_values = np.array([float(row["acc"]) for row in rows])
    f1_values = np.array([float(row["f1"]) for row in rows])

    y = np.arange(len(rows))
    height = 0.34
    colors_acc = ["#2F6F9F" if row["is_full"].lower() == "yes" else "#A9C3D9" for row in rows]
    colors_f1 = ["#C75D2C" if row["is_full"].lower() == "yes" else "#E6B292" for row in rows]

    fig, ax = plt.subplots(figsize=(7.3, 3.05))
    bars_acc = ax.barh(
        y + height / 2,
        acc_values,
        height,
        label="Acc.",
        color=colors_acc,
        edgecolor="#2C3E50",
        linewidth=0.65,
    )
    bars_f1 = ax.barh(
        y - height / 2,
        f1_values,
        height,
        label="F1",
        color=colors_f1,
        edgecolor="#2C3E50",
        linewidth=0.65,
    )

    ax.set_xlabel("Score")
    ax.set_xlim(0.858, 0.895)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#D9E2EC", linewidth=0.8, alpha=0.75)
    ax.legend(frameon=False, ncol=2, loc="lower right", bbox_to_anchor=(0.99, 0.02))

    for bars in (bars_acc, bars_f1):
        for bar in bars:
            ax.text(
                bar.get_width() + 0.0004,
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.4f}",
                ha="left",
                va="center",
                fontsize=7.2,
            )

    save_fig_bundle(
        fig,
        [
            FIG_DIR / "twibot20_component_ablation.pdf",
            FIG_DIR / "twibot20_component_ablation.png",
        ],
    )


def main() -> None:
    write_latex_table()
    plot()
    print(f"Read {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_TEX}")
    print(f"Wrote {FIG_DIR / 'twibot20_component_ablation.pdf'}")


if __name__ == "__main__":
    main()
