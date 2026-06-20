from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import apply_publication_style, save_fig_bundle


FIG_DIR = Path(__file__).resolve().parent
DATA_DIR = FIG_DIR / "data"
OUTPUT_CSV = DATA_DIR / "twibot20_hparam_sensitivity.csv"
OUTPUT_TEX = DATA_DIR / "twibot20_hparam_sensitivity_table.tex"

PANELS = [
    ("K", "K"),
    ("fanout", "Fanout"),
    ("budget", "Budget"),
]


def _tex(text: str) -> str:
    return text.replace("%", r"\%")


def _load_csv_rows() -> list[dict[str, str]]:
    with OUTPUT_CSV.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_latex_table() -> None:
    rows = _load_csv_rows()
    lines = [
        r"\begin{table}[t]",
        r"\caption{Hyperparameter sensitivity on TwiBot-20. The full model uses $K=8$, fanout 64, and a routed budget of 10\%.}",
        r"\label{tab:hparam-sensitivity}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Hyperparameter & Setting & Acc. & F1 \\",
        r"\midrule",
    ]
    for row in rows:
        acc = row["acc"]
        f1 = row["f1"]
        if row["is_full"].lower() == "yes":
            acc = rf"\textbf{{{acc}}}"
            f1 = rf"\textbf{{{f1}}}"
        lines.append(f"{_tex(row['hyperparameter'])} & {_tex(row['setting'])} & {acc} & {f1} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    OUTPUT_TEX.write_text("\n".join(lines), encoding="utf-8")


def _panel_rows(rows: list[dict[str, str]], hyperparameter: str) -> list[dict[str, str]]:
    return [row for row in rows if row["hyperparameter"] == hyperparameter]


def plot() -> None:
    apply_publication_style()
    rows = _load_csv_rows()
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.65), gridspec_kw={"wspace": 0.52})

    for panel_idx, (hyperparameter, xlabel) in enumerate(PANELS):
        ax = axes[panel_idx]
        panel_rows = _panel_rows(rows, hyperparameter)
        x = np.arange(len(panel_rows))
        labels = [row["setting_label"] for row in panel_rows]
        acc_values = np.array([float(row["acc"]) for row in panel_rows])
        f1_values = np.array([float(row["f1"]) for row in panel_rows])
        full_idx = next((idx for idx, row in enumerate(panel_rows) if row["is_full"].lower() == "yes"), None)

        ax.plot(
            x,
            acc_values,
            marker="o",
            markersize=4.0,
            linewidth=1.5,
            color="#4C78A8",
            label="Acc." if panel_idx == 0 else None,
        )
        ax.plot(
            x,
            f1_values,
            marker="s",
            markersize=4.0,
            linewidth=1.5,
            color="#F58518",
            label="F1" if panel_idx == 0 else None,
        )
        if full_idx is not None:
            ax.axvline(full_idx, color="#525252", linestyle="--", linewidth=0.85, alpha=0.75)

        ax.set_xlabel(xlabel)
        if panel_idx == 0:
            ax.set_ylabel("Score")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.872, 0.892)
        ax.grid(axis="y", color="#D9E2EC", linewidth=0.8, alpha=0.75)
        ax.text(
            0.02,
            0.98,
            chr(ord("A") + panel_idx),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

    axes[0].legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0.00, 0.01), fontsize=8)
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.21, top=0.96)

    save_fig_bundle(
        fig,
        [
            FIG_DIR / "twibot20_seed1_hparam_sensitivity.pdf",
            FIG_DIR / "twibot20_seed1_hparam_sensitivity.png",
        ],
    )


def main() -> None:
    write_latex_table()
    plot()
    print(f"Read {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_TEX}")
    print(f"Wrote {FIG_DIR / 'twibot20_seed1_hparam_sensitivity.pdf'}")


if __name__ == "__main__":
    main()
