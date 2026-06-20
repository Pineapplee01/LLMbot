from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches

from paper_plot_style import apply_publication_style, save_fig_bundle


FIG_DIR = Path(__file__).resolve().parent


COLORS = {
    "feature": "#DCEBFA",
    "hybrid": "#F5E6CC",
    "detector": "#DFF1E3",
    "risk": "#F8D7DA",
    "high": "#E7DAF3",
    "low": "#DCEBFA",
    "edge": "#344054",
    "muted": "#667085",
    "routed": "#C84C4C",
    "nonrouted": "#4C78A8",
}


def _box(ax, xy, width, height, text, facecolor, edgecolor=None, fontsize=9, weight="normal", linestyle="-"):
    edgecolor = edgecolor or COLORS["edge"]
    patch = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.2,
        linestyle=linestyle,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color="#182230",
    )
    return patch


def _arrow(ax, start, end, color=None, linewidth=1.5, linestyle="-", alpha=1.0):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "->",
            "color": color or COLORS["edge"],
            "lw": linewidth,
            "linestyle": linestyle,
            "alpha": alpha,
            "shrinkA": 2,
            "shrinkB": 2,
        },
    )


def _section(ax, xy, width, height, title, facecolor):
    ax.add_patch(
        patches.FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.014,rounding_size=0.026",
            facecolor=facecolor,
            edgecolor="#98A2B3",
            linewidth=1.2,
        )
    )
    ax.text(xy[0] + 0.02, xy[1] + height - 0.055, title, ha="left", va="center", fontsize=11, fontweight="bold")


def _mini_nodes(ax, center_x, center_y, labels, colors):
    xs = [center_x - 0.055, center_x - 0.015, center_x + 0.025, center_x + 0.065]
    ys = [center_y, center_y + 0.032, center_y - 0.020, center_y + 0.010]
    for i in range(len(xs) - 1):
        ax.plot([xs[i], xs[i + 1]], [ys[i], ys[i + 1]], color="#98A2B3", lw=1.0)
    for x, y, label, color in zip(xs, ys, labels, colors):
        ax.add_patch(patches.Circle((x, y), 0.018, facecolor=color, edgecolor="white", linewidth=0.9))
        ax.text(x, y, label, ha="center", va="center", fontsize=7, color="white", fontweight="bold")


def build() -> None:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(10.4, 4.05))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _section(ax, (0.025, 0.13), 0.25, 0.74, "Feature Encoder", COLORS["feature"])
    _section(ax, (0.315, 0.13), 0.39, 0.74, "Hybrid-Order Relation Learner", COLORS["hybrid"])
    _section(ax, (0.745, 0.13), 0.23, 0.74, "Bot Detector", COLORS["detector"])

    _box(ax, (0.055, 0.67), 0.19, 0.11, "node input $x$\nsemantic/profile evidence", "#FFFFFF", fontsize=8.8)
    _box(ax, (0.055, 0.47), 0.19, 0.11, "low-order relation\nbackbone", COLORS["low"], fontsize=8.8)
    _box(ax, (0.055, 0.27), 0.19, 0.11, "$x_{low}$\nrelation representation", "#FFFFFF", fontsize=8.8)

    _arrow(ax, (0.15, 0.67), (0.15, 0.58))
    _arrow(ax, (0.15, 0.47), (0.15, 0.38))

    _box(ax, (0.345, 0.66), 0.15, 0.12, "$x_{new}=[x_{low};x]$\nconstruction space", "#FFFFFF", fontsize=8.7)
    _box(ax, (0.530, 0.66), 0.14, 0.12, "Risk Aggregator\nerror-risk ranking", COLORS["risk"], fontsize=8.7)
    _box(ax, (0.352, 0.38), 0.15, 0.13, "High-Order Learner\nKNN hypergraph / HGNN", COLORS["high"], fontsize=8.7)
    _box(ax, (0.535, 0.38), 0.13, 0.13, "$x_{high}$\nsupport evidence", "#FFFFFF", fontsize=8.7)

    _mini_nodes(
        ax,
        0.437,
        0.255,
        ["R", "B", "B", "H"],
        [COLORS["routed"], COLORS["routed"], COLORS["routed"], COLORS["nonrouted"]],
    )
    ax.text(0.437, 0.185, "routed centers with\nsupport members", ha="center", va="center", fontsize=8.0, color=COLORS["muted"])

    _arrow(ax, (0.245, 0.725), (0.345, 0.725))
    _arrow(ax, (0.245, 0.325), (0.345, 0.690))
    _arrow(ax, (0.495, 0.720), (0.530, 0.720))
    _arrow(ax, (0.420, 0.660), (0.420, 0.510))
    _arrow(ax, (0.502, 0.445), (0.535, 0.445))
    _arrow(ax, (0.600, 0.660), (0.600, 0.510), color=COLORS["routed"], linewidth=1.4)

    _box(ax, (0.775, 0.62), 0.17, 0.12, "routed nodes\nactivate residual", "#FFFFFF", edgecolor=COLORS["routed"], fontsize=8.8)
    _box(ax, (0.775, 0.41), 0.17, 0.12, "$h_i=x_{low,i}+\\gamma_i\\Delta_i$\nif $i\\in R_B$", "#FFF7ED", edgecolor="#D9822B", fontsize=8.5)
    _box(ax, (0.775, 0.22), 0.17, 0.12, "non-routed nodes\nbypass high-order branch", "#FFFFFF", edgecolor=COLORS["nonrouted"], fontsize=8.8)

    _arrow(ax, (0.670, 0.720), (0.775, 0.680), color=COLORS["routed"], linewidth=1.6)
    _arrow(ax, (0.668, 0.445), (0.775, 0.470), color="#D9822B", linewidth=1.6)
    _arrow(ax, (0.245, 0.325), (0.775, 0.280), color=COLORS["nonrouted"], linewidth=1.1, linestyle="--", alpha=0.85)
    _arrow(ax, (0.860, 0.620), (0.860, 0.530), color=COLORS["routed"], linewidth=1.4)

    ax.text(0.600, 0.620, "hard-node set", ha="center", va="center", fontsize=7.8, color=COLORS["routed"])
    ax.text(0.338, 0.292, "bypass path", ha="left", va="center", fontsize=7.8, color=COLORS["nonrouted"])

    save_fig_bundle(
        fig,
        [
            FIG_DIR / "Architecture_fit.pdf",
            FIG_DIR / "Architecture_fit.png",
            FIG_DIR / "Architecture_fit.svg",
        ],
    )


if __name__ == "__main__":
    build()
    print(f"Wrote {FIG_DIR / 'Architecture_fit.pdf'}")
