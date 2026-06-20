from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

FONT_FAMILY = "DejaVu Serif"
COLORS = {
    "low": "#3B6FB6",
    "high": "#D9822B",
    "input": "#5C8D76",
    "risk": "#C84C4C",
    "gate": "#2B8A6E",
    "neutral": "#E8EDF3",
    "neutral_edge": "#5A6472",
    "routed": "#C84C4C",
    "non_routed": "#3B6FB6",
    "support_good": "#4C956C",
    "gray_text": "#495057",
}


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": FONT_FAMILY,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "mathtext.fontset": "stix",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_fig(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_format = output_path.suffix.lower().lstrip(".") or "pdf"
    fig.savefig(output_path, format=output_format)
    plt.close(fig)


def save_fig_bundle(fig: plt.Figure, output_paths) -> None:
    output_paths = list(output_paths)
    if not output_paths:
        plt.close(fig)
        return
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_format = output_path.suffix.lower().lstrip(".") or "pdf"
        fig.savefig(output_path, format=output_format)
    plt.close(fig)


def _box(ax, xy, width, height, text, fc, ec="#425466", fontsize=9, weight="normal", ls="-"):
    patch = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.2,
        facecolor=fc,
        edgecolor=ec,
        linestyle=ls,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2.0,
        xy[1] + height / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color="#1F2933",
    )
    return patch


def _arrow(ax, start, end, color="#425466", lw=1.5, ls="-", alpha=1.0, shrink_a=0.0, shrink_b=0.0):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="->",
            color=color,
            lw=lw,
            linestyle=ls,
            alpha=alpha,
            shrinkA=shrink_a,
            shrinkB=shrink_b,
        ),
    )


def _polyline_arrow(ax, points, color="#425466", lw=1.5, ls="-", alpha=1.0):
    if len(points) < 2:
        return
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            lw=lw,
            linestyle=ls,
            alpha=alpha,
        )
    _arrow(ax, points[-2], points[-1], color=color, lw=lw, ls=ls, alpha=alpha)


def _callout(ax, xy, text, fc="#FFFFFF", ec="#7A8797", fontsize=8.5):
    _box(ax, xy, 0.16, 0.09, text, fc=fc, ec=ec, fontsize=fontsize)


def _panel_box(ax, xy, width, height):
    patch = patches.Rectangle(
        xy,
        width,
        height,
        linewidth=1.2,
        facecolor="none",
        edgecolor="#111111",
        linestyle=(0, (3.2, 3.2)),
    )
    ax.add_patch(patch)
    return patch


def _ribbon(ax, x_center, y_center, width, height, text, fc, ec):
    left = x_center - width / 2.0
    right = x_center + width / 2.0
    top = y_center + height / 2.0
    bottom = y_center - height / 2.0
    notch = height * 0.42
    tail = height * 0.55

    body = patches.Rectangle((left, bottom), width, height, facecolor=fc, edgecolor=ec, linewidth=1.0)
    ax.add_patch(body)
    left_tail = patches.Polygon(
        [
            (left, bottom + 0.02 * height),
            (left - tail, bottom + 0.02 * height),
            (left - tail * 0.55, y_center),
            (left - tail, top - 0.02 * height),
            (left, top - 0.02 * height),
            (left + notch, y_center),
        ],
        closed=True,
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.0,
    )
    right_tail = patches.Polygon(
        [
            (right, bottom + 0.02 * height),
            (right + tail, bottom + 0.02 * height),
            (right + tail * 0.55, y_center),
            (right + tail, top - 0.02 * height),
            (right, top - 0.02 * height),
            (right - notch, y_center),
        ],
        closed=True,
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.0,
    )
    ax.add_patch(left_tail)
    ax.add_patch(right_tail)
    ax.text(x_center, y_center, text, ha="center", va="center", fontsize=12.5, fontweight="bold", color="#111111")


def _stacked_cards_icon(ax, x, y, width, height, fc="#D8E7F9", ec="#222222"):
    offsets = [(0.018, 0.018), (0.009, 0.009), (0.0, 0.0)]
    for dx, dy in offsets:
        card = patches.FancyBboxPatch(
            (x - dx, y - dy),
            width,
            height,
            boxstyle="round,pad=0.003,rounding_size=0.01",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.2,
        )
        ax.add_patch(card)
    tail = patches.Polygon(
        [(x + width * 0.16, y), (x + width * 0.26, y), (x + width * 0.19, y - height * 0.12)],
        closed=True,
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.0,
    )
    ax.add_patch(tail)
    for yy in (0.72, 0.50, 0.30):
        ax.plot([x + width * 0.18, x + width * 0.82], [y + height * yy, y + height * yy], color="#4B5D73", lw=1.1)


def _profile_icon(ax, x, y, width, height, fc="#DCEFCF", ec="#222222"):
    card = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.003,rounding_size=0.01",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.2,
    )
    ax.add_patch(card)
    fold = patches.Polygon(
        [(x + width * 0.78, y + height), (x + width, y + height), (x + width, y + height * 0.78)],
        closed=True,
        facecolor="#EEF6E8",
        edgecolor=ec,
        linewidth=1.0,
    )
    ax.add_patch(fold)
    ax.add_patch(patches.Circle((x + width * 0.45, y + height * 0.62), width * 0.13, fill=False, edgecolor="#39543A", linewidth=1.4))
    ax.add_patch(
        patches.Arc(
            (x + width * 0.45, y + height * 0.33),
            width * 0.42,
            height * 0.34,
            theta1=0,
            theta2=180,
            edgecolor="#39543A",
            linewidth=1.4,
        )
    )


def _draw_social_graph_icon(ax, center_x, center_y, scale=1.0):
    coords = [
        (-0.045, 0.00),
        (-0.012, 0.038),
        (-0.005, -0.05),
        (0.034, 0.03),
        (0.056, -0.03),
        (0.00, 0.00),
    ]
    edges = [(0, 1), (0, 2), (0, 5), (1, 3), (1, 5), (2, 5), (5, 3), (5, 4), (3, 4)]
    for a, b in edges:
        ax.plot(
            [center_x + coords[a][0] * scale, center_x + coords[b][0] * scale],
            [center_y + coords[a][1] * scale, center_y + coords[b][1] * scale],
            color="#5B5B5B",
            lw=1.3,
            zorder=1,
        )
    node_colors = ["#7FB2E5", "#F3C0BE", "#7FB2E5", "#7FB2E5", "#F3C0BE", "#CFC6E8"]
    for (dx, dy), color in zip(coords, node_colors):
        ax.add_patch(
            patches.Circle(
                (center_x + dx * scale, center_y + dy * scale),
                0.0095 * scale,
                facecolor=color,
                edgecolor="#222222",
                linewidth=1.0,
                zorder=2,
            )
        )


def _trapezoid_block(ax, xy, width, height, text, fc, ec="#222222", fontsize=10):
    x, y = xy
    pts = [
        (x, y),
        (x + width * 0.83, y + height * 0.10),
        (x + width, y + height / 2.0),
        (x + width * 0.83, y + height * 0.90),
        (x, y + height),
    ]
    poly = patches.Polygon(pts, closed=True, facecolor=fc, edgecolor=ec, linewidth=1.2)
    ax.add_patch(poly)
    ax.text(x + width * 0.43, y + height / 2.0, text, ha="center", va="center", fontsize=fontsize, color="#111111")
    return poly


def _embedding_bar(ax, x, y, width, height, n_segments, palette, label, label_pos="top"):
    segment_h = height / n_segments
    for idx in range(n_segments):
        color = palette[idx % len(palette)]
        ax.add_patch(
            patches.Rectangle(
                (x, y + idx * segment_h),
                width,
                segment_h,
                facecolor=color,
                edgecolor="#222222",
                linewidth=0.8,
            )
        )
    if label_pos == "top":
        ax.text(x + width / 2.0, y + height + 0.012, label, ha="center", va="bottom", fontsize=11)
    elif label_pos == "right":
        ax.text(x + width + 0.012, y + height / 2.0, label, ha="left", va="center", fontsize=11)
    else:
        ax.text(x + width / 2.0, y - 0.02, label, ha="center", va="top", fontsize=11)


def _vertical_pill(ax, xy, width, height, text, fc, ec="#222222", fontsize=10):
    patch = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.01,rounding_size=0.01",
        linewidth=1.0,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2.0, xy[1] + height / 2.0, text, rotation=90, ha="center", va="center", fontsize=fontsize)
    return patch


def _support_group_card(ax, xy, size):
    x, y = xy
    ax.add_patch(patches.Rectangle((x, y), size, size, facecolor="#F8F8F8", edgecolor="#B9B9B9", linewidth=1.0))
    groups = [
        ((x + size * 0.26, y + size * 0.70), "#E8D39A"),
        ((x + size * 0.71, y + size * 0.63), "#D8B1AF"),
        ((x + size * 0.31, y + size * 0.32), "#A8C6E6"),
        ((x + size * 0.58, y + size * 0.28), "#B9D7A8"),
    ]
    for (cx, cy), color in groups:
        ax.add_patch(patches.Circle((cx, cy), size * 0.18, facecolor=color, edgecolor="#777777", linewidth=1.0, alpha=0.65))
        mini_offsets = [(-0.05, 0.02), (0.03, 0.05), (0.04, -0.04)]
        for dx, dy in mini_offsets:
            ax.add_patch(
                patches.Circle(
                    (cx + dx * size, cy + dy * size),
                    size * 0.03,
                    facecolor="#F8F0D3",
                    edgecolor="#9A7C2A",
                    linewidth=0.6,
                )
            )


def _hypergraph_illustration(ax, center_x, center_y, radius):
    ax.add_patch(patches.Circle((center_x, center_y), radius, facecolor="#F7F7F7", edgecolor="#222222", linewidth=1.0))
    ax.add_patch(patches.Circle((center_x, center_y), radius * 0.88, facecolor="none", edgecolor="#222222", linewidth=0.9))
    groups = [
        ((center_x - radius * 0.38, center_y - radius * 0.12), "#D8A7A7"),
        ((center_x + radius * 0.18, center_y + radius * 0.18), "#B3C6EE"),
        ((center_x + radius * 0.30, center_y - radius * 0.22), "#B3C6EE"),
    ]
    offsets = [(-0.28, 0.00), (0.00, 0.26), (0.21, -0.18), (0.12, 0.10)]
    for (gx, gy), color in groups:
        ax.add_patch(patches.Circle((gx, gy), radius * 0.38, facecolor=color, edgecolor="#222222", linewidth=1.0, alpha=0.75))
        nodes = []
        for dx, dy in offsets[:3]:
            px = gx + dx * radius
            py = gy + dy * radius
            nodes.append((px, py))
            ax.add_patch(patches.Circle((px, py), radius * 0.065, facecolor=color, edgecolor="#222222", linewidth=0.9))
        for a in range(len(nodes)):
            for b in range(a + 1, len(nodes)):
                ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]], color="#4B4B4B", lw=1.0)


def _human_bot_icons(ax, human_xy, bot_xy, scale=1.0):
    hx, hy = human_xy
    bx, by = bot_xy
    ax.add_patch(patches.Circle((hx, hy + 0.022 * scale), 0.011 * scale, facecolor="#C7DDA8", edgecolor="#333333", linewidth=1.0))
    ax.add_patch(
        patches.Arc(
            (hx, hy - 0.002 * scale),
            0.042 * scale,
            0.035 * scale,
            theta1=0,
            theta2=180,
            edgecolor="#5B7E37",
            linewidth=1.5,
        )
    )
    ax.add_patch(
        patches.FancyBboxPatch(
            (bx - 0.017 * scale, by + 0.002 * scale),
            0.034 * scale,
            0.028 * scale,
            boxstyle="round,pad=0.004,rounding_size=0.006",
            facecolor="#E8C0C0",
            edgecolor="#333333",
            linewidth=1.0,
        )
    )
    ax.plot([bx - 0.010 * scale, bx + 0.010 * scale], [by + 0.031 * scale, by + 0.031 * scale], color="#333333", lw=1.0)
    ax.plot([bx, bx], [by + 0.031 * scale, by + 0.040 * scale], color="#333333", lw=1.0)
    ax.add_patch(patches.Circle((bx - 0.007 * scale, by + 0.015 * scale), 0.0026 * scale, color="#333333"))
    ax.add_patch(patches.Circle((bx + 0.007 * scale, by + 0.015 * scale), 0.0026 * scale, color="#333333"))
    ax.plot([bx - 0.006 * scale, bx + 0.006 * scale], [by + 0.008 * scale, by + 0.008 * scale], color="#333333", lw=0.9)


def _arrow_label(ax, text, xy, fontsize=10):
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=fontsize)


def _read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _draw_selective_panel(ax, title: str, routed: bool, caption: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.94, title, ha="center", va="top", fontsize=10, fontweight="bold")

    _box(ax, (0.08, 0.62), 0.22, 0.16, r"$x_{new}$ support", COLORS["neutral"], ec=COLORS["neutral_edge"])
    for x, y in ((0.14, 0.70), (0.20, 0.74), (0.24, 0.66), (0.27, 0.72)):
        ax.add_patch(patches.Circle((x, y), 0.014, facecolor=COLORS["non_routed"], edgecolor="white", linewidth=0.8))
    center_color = COLORS["routed"] if routed else COLORS["support_good"]
    ax.add_patch(patches.Circle((0.19, 0.68), 0.018, facecolor=center_color, edgecolor="white", linewidth=0.9))
    ax.text(0.19, 0.58, "same $x_{new}$ construction", ha="center", va="center", fontsize=8.5, color=COLORS["gray_text"])

    _box(ax, (0.40, 0.65), 0.18, 0.10, r"$x_{low}$", COLORS["low"], ec=COLORS["low"], fontsize=10, weight="bold")
    if routed:
        _box(ax, (0.40, 0.44), 0.18, 0.10, r"$\gamma_i \cdot \Delta_i$", "#F9E1D7", ec=COLORS["high"], fontsize=10, weight="bold")
        ax.text(0.49, 0.37, r"$i \in R_B$", ha="center", va="center", fontsize=9, color=COLORS["risk"])
    else:
        _box(ax, (0.40, 0.44), 0.18, 0.10, r"$0 \cdot \Delta_i$", "#F5F5F5", ec=COLORS["neutral_edge"], fontsize=10, weight="bold")
        ax.text(0.49, 0.37, r"$i \notin R_B$", ha="center", va="center", fontsize=9, color=COLORS["gray_text"])

    _arrow(ax, (0.58, 0.70), (0.77, 0.70), color=COLORS["low"], lw=2.8)
    if routed:
        _arrow(ax, (0.58, 0.49), (0.77, 0.49), color=COLORS["high"], lw=3.8, alpha=0.95)
    else:
        _arrow(ax, (0.58, 0.49), (0.77, 0.49), color="#A0AEC0", lw=1.2, ls="--", alpha=0.9)

    ax.text(0.67, 0.58, "+", ha="center", va="center", fontsize=15, color="#3D4852")
    _box(ax, (0.80, 0.56), 0.15, 0.14, "hidden", "#EDF7F3", ec=COLORS["gate"], fontsize=10, weight="bold")
    ax.text(0.50, 0.18, caption, ha="center", va="center", fontsize=8.4, color=COLORS["gray_text"])


def draw_conflict_panel(ax, rows) -> None:
    metrics = []
    grouped = {}
    for row in rows:
        metric = row["metric_label"]
        group = row["group"]
        value = float(row["value"])
        if metric not in grouped:
            grouped[metric] = {}
            metrics.append(metric)
        grouped[metric][group] = value

    x = range(len(metrics))
    width = 0.32
    routed_values = [grouped[m]["routed_test_5pct"] for m in metrics]
    non_routed_values = [grouped[m]["non_routed_test_complement"] for m in metrics]

    routed_bars = ax.bar(
        [i - width / 2 for i in x],
        routed_values,
        width=width,
        color=COLORS["routed"],
        label="Routed test 5%",
    )
    non_routed_bars = ax.bar(
        [i + width / 2 for i in x],
        non_routed_values,
        width=width,
        color=COLORS["non_routed"],
        label="Non-routed test",
    )

    ax.set_ylabel("Mean value")
    ax.set_ylim(0.0, 1.08)
    ax.set_xticks(list(x))
    ax.set_xticklabels(["Local\ndisagreement", "High-risk\nsupport mass", "Stable\nsupport mass"])
    ax.grid(axis="y", color="#D9E2EC", linewidth=0.8, alpha=0.7)
    ax.legend(frameon=False, loc="upper right")

    for bars in (routed_bars, non_routed_bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.022,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.3,
            )


def draw_support_ablation_panel(ax, rows) -> None:
    slices = ["routed_test_5pct", "non_routed_test_complement"]
    labels = {
        "routed_test_5pct": "Routed test 5%",
        "non_routed_test_complement": "Non-routed test",
    }
    values = {}
    for row in rows:
        values[(row["slice"], row["policy"])] = float(row["macro_f1"])

    x = range(len(slices))
    width = 0.32
    default_vals = [values[(s, "default")] for s in slices]
    filtered_vals = [values[(s, "post_topk_exclude_routed")] for s in slices]

    bars_default = ax.bar(
        [i - width / 2 for i in x],
        default_vals,
        width=width,
        color=COLORS["non_routed"],
        label="Default support",
    )
    bars_filtered = ax.bar(
        [i + width / 2 for i in x],
        filtered_vals,
        width=width,
        color=COLORS["high"],
        label="Post-topK exclude-routed",
    )

    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0.58, 0.90)
    ax.set_xticks(list(x))
    ax.set_xticklabels([labels[s] for s in slices])
    ax.grid(axis="y", color="#D9E2EC", linewidth=0.8, alpha=0.7)
    ax.legend(frameon=False, loc="upper left")

    for bars in (bars_default, bars_filtered):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.006,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=8.2,
            )


def build_framework_overview(output_path: Path | None = None) -> Path:
    apply_publication_style()
    if output_path is None:
        output_path = ROOT / "framework_overview_risk_gated_residual.pdf"

    svg_output_path = output_path.with_suffix(".svg")

    fig, ax = plt.subplots(figsize=(12.8, 6.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    panel_y = 0.105
    panel_h = 0.79
    panel_a = (0.025, panel_y, 0.315, panel_h)
    panel_b = (0.345, panel_y, 0.395, panel_h)
    panel_c = (0.745, panel_y, 0.235, panel_h)

    for panel in (panel_a, panel_b, panel_c):
        _panel_box(ax, (panel[0], panel[1]), panel[2], panel[3])

    _ribbon(ax, 0.182, 0.92, 0.255, 0.075, "(a) Node Representation\nEncoder", "#D8E7F9", "#7C92A9")
    _ribbon(ax, 0.542, 0.92, 0.292, 0.075, "(b) Risk-Guided High-Order\nRelation Modeling", "#E9DDF3", "#9D8CAB")
    _ribbon(ax, 0.862, 0.92, 0.145, 0.075, "(c) Routed Residual\nConsumer", "#F4D7D7", "#C28F8F")

    _stacked_cards_icon(ax, 0.032, 0.669, 0.055, 0.070)
    _profile_icon(ax, 0.040, 0.448, 0.048, 0.067)
    _draw_social_graph_icon(ax, 0.050, 0.308, scale=1.15)
    ax.text(0.057, 0.629, "User Tweets", ha="center", va="top", fontsize=10)
    ax.text(0.058, 0.408, "User Profile", ha="center", va="top", fontsize=10)
    ax.text(0.060, 0.228, "Social Graph", ha="center", va="top", fontsize=10)

    _trapezoid_block(ax, (0.115, 0.596), 0.095, 0.158, "Semantic\nEncoder", "#DBEBD5", fontsize=10.6)
    _trapezoid_block(ax, (0.115, 0.288), 0.095, 0.158, "Relation\nEncoder", "#E5D9EB", fontsize=10.6)

    _embedding_bar(ax, 0.255, 0.568, 0.018, 0.168, 5, ["#B9D1DB"], "$x$")
    _embedding_bar(ax, 0.255, 0.260, 0.018, 0.168, 5, ["#DABDC2"], "$x_{low}$")
    ax.text(0.257, 0.545, "Semantic\nRepresentation", ha="center", va="top", fontsize=10)
    ax.text(0.258, 0.236, "Low-Order\nRepresentation", ha="center", va="top", fontsize=10)

    _arrow(ax, (0.087, 0.672), (0.115, 0.672))
    _polyline_arrow(ax, [(0.084, 0.482), (0.084, 0.626), (0.115, 0.626)])
    _arrow(ax, (0.088, 0.366), (0.115, 0.366))
    _arrow(ax, (0.210, 0.675), (0.255, 0.675))
    _arrow(ax, (0.210, 0.367), (0.255, 0.367))

    _vertical_pill(ax, (0.377, 0.403), 0.028, 0.135, "concat", "#F7DEC8", fontsize=10)
    ax.text(0.412, 0.742, r"$x_{new} =$", ha="right", va="center", fontsize=16)
    ax.text(0.480, 0.745, "[", ha="center", va="center", fontsize=26)
    ax.text(0.508, 0.768, r"$x_{low}$", ha="center", va="center", fontsize=16)
    ax.text(0.508, 0.722, r"$x$", ha="center", va="center", fontsize=16)
    ax.text(0.536, 0.745, "]", ha="center", va="center", fontsize=26)

    _support_group_card(ax, (0.447, 0.535), 0.084)
    ax.text(0.489, 0.512, "target-centered\nsemantic support group", ha="center", va="top", fontsize=10)

    _hypergraph_illustration(ax, 0.500, 0.332, 0.073)
    ax.text(0.500, 0.205, "hypergraph/group\nrelation illustration", ha="center", va="top", fontsize=10)

    _trapezoid_block(ax, (0.585, 0.307), 0.073, 0.165, "HGNN\nLayer", "#D8D1ED", fontsize=10.6)
    _embedding_bar(ax, 0.680, 0.281, 0.018, 0.167, 5, ["#C9BDE7"], "$x_{high}$")
    ax.text(0.697, 0.257, "High-Order\nSupport View", ha="center", va="top", fontsize=10)
    ax.text(0.693, 0.360, r"$x_{low}$", ha="left", va="center", fontsize=12)

    _box(
        ax,
        (0.570, 0.605),
        0.112,
        0.145,
        "Conformal\nRisk\nEstimator",
        "#F4D0CD",
        ec="#8D4D4D",
        fontsize=12,
    )
    ax.text(0.628, 0.577, u"\u2022 node-wise\n  error risk", ha="center", va="top", fontsize=10)
    ax.text(0.718, 0.678, r"$risk_i$", ha="left", va="center", fontsize=18)

    ax.plot([0.291, 0.291], [0.344, 0.675], color="#425466", lw=1.5)
    _arrow(ax, (0.273, 0.652), (0.291, 0.652))
    _arrow(ax, (0.273, 0.344), (0.291, 0.344))
    _arrow(ax, (0.291, 0.470), (0.377, 0.470))
    _polyline_arrow(ax, [(0.405, 0.470), (0.445, 0.470), (0.445, 0.678), (0.570, 0.678)])
    _polyline_arrow(ax, [(0.489, 0.535), (0.489, 0.410), (0.500, 0.410)])
    _arrow(ax, (0.573, 0.390), (0.585, 0.390))
    _arrow(ax, (0.658, 0.390), (0.680, 0.364))

    _box(
        ax,
        (0.790, 0.622),
        0.154,
        0.108,
        "Top-$\\beta$ routing\n$R_{\\beta}=\\mathrm{TopB}(\\{r_i\\},\\beta|V|)$",
        "#DFEFD0",
        ec="#69815C",
        fontsize=11,
    )
    fusion_box = patches.FancyBboxPatch(
        (0.805, 0.378),
        0.135,
        0.168,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.2,
        facecolor="#F7E6A4",
        edgecolor="#8F7A32",
    )
    ax.add_patch(fusion_box)
    ax.plot([0.805, 0.940], [0.477, 0.477], color="#8F7A32", lw=1.0)
    ax.text(0.873, 0.507, "Residual Fusion", ha="center", va="center", fontsize=12)
    ax.text(0.873, 0.450, r"$i \in R_{\beta}$", ha="center", va="center", fontsize=12)
    inner = patches.FancyBboxPatch(
        (0.818, 0.406),
        0.112,
        0.047,
        boxstyle="round,pad=0.005,rounding_size=0.008",
        linewidth=1.0,
        facecolor="#FFF6E9",
        edgecolor="#9A8E7A",
    )
    ax.add_patch(inner)
    ax.text(0.874, 0.429, r"$h_i = x_{low,i} + \mathbf{1}(i\in R_{\beta})\,\gamma_i \delta_i$", ha="center", va="center", fontsize=9.2)

    _box(ax, (0.847, 0.241), 0.050, 0.083, "Classifier", "#F7C08A", ec="#9A6E45", fontsize=11)
    _human_bot_icons(ax, (0.837, 0.155), (0.905, 0.155), scale=1.0)
    ax.text(0.837, 0.122, "Human", ha="center", va="top", fontsize=10)
    ax.text(0.905, 0.122, "Bot", ha="center", va="top", fontsize=10)

    ax.plot([0.725, 0.725], [0.318, 0.676], color="#425466", lw=1.5)
    _arrow(ax, (0.682, 0.676), (0.790, 0.676))
    _arrow(ax, (0.698, 0.364), (0.725, 0.364))
    _arrow(ax, (0.693, 0.318), (0.725, 0.318))
    _arrow(ax, (0.725, 0.455), (0.805, 0.455))
    _arrow(ax, (0.725, 0.415), (0.805, 0.415))
    _arrow(ax, (0.867, 0.622), (0.867, 0.546))
    _arrow(ax, (0.874, 0.378), (0.874, 0.324))
    _polyline_arrow(ax, [(0.873, 0.241), (0.873, 0.190), (0.837, 0.190), (0.837, 0.182)])
    _polyline_arrow(ax, [(0.873, 0.241), (0.873, 0.190), (0.905, 0.190), (0.905, 0.182)])

    ax.text(0.182, 0.065, "(a) Node Representation Encoder", ha="center", va="center", fontsize=12)
    ax.text(0.542, 0.065, "(b) Risk-Guided High-Order Relation Modeling", ha="center", va="center", fontsize=12)
    ax.text(0.862, 0.065, "(c) Routed Residual Consumer", ha="center", va="center", fontsize=12)

    save_fig_bundle(fig, [output_path, svg_output_path])
    return output_path


def build_fig2a_gate_mechanism(output_path: Path | None = None) -> Path:
    apply_publication_style()
    if output_path is None:
        output_path = ROOT / "fig2a_gate_mechanism.pdf"

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.3))
    _draw_selective_panel(
        axes[0],
        "non-routed node",
        routed=False,
        caption="keep low-order prediction",
    )
    _draw_selective_panel(
        axes[1],
        "routed hard node",
        routed=True,
        caption="activate high-order residual correction",
    )
    _draw_selective_panel(
        axes[2],
        "routed + filtered support",
        routed=True,
        caption="prefer stable non-routed support members",
    )

    fig.text(0.5, 0.02, "same $x_{new}$ construction, selective routed-only residual consumption", ha="center", va="bottom", fontsize=9.5, color=COLORS["gray_text"])
    save_fig(fig, output_path)
    return output_path


def build_composite_figure2(conflict_csv: Path | None = None, support_csv: Path | None = None, output_path: Path | None = None) -> Path:
    apply_publication_style()
    if conflict_csv is None:
        conflict_csv = DATA_DIR / "fig2_conflict_summary.csv"
    if support_csv is None:
        support_csv = DATA_DIR / "fig2_support_ablation.csv"
    if output_path is None:
        output_path = ROOT / "risk_gate_mechanism_and_evidence.pdf"

    conflict_rows = _read_csv_rows(conflict_csv)
    support_rows = _read_csv_rows(support_csv)

    fig = plt.figure(figsize=(12.8, 7.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.34, wspace=0.26)

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    ax_a.axis("off")
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.text(0.00, 1.02, "A", ha="left", va="bottom", fontsize=12, fontweight="bold")
    ax_b.text(-0.14, 1.02, "B", transform=ax_b.transAxes, ha="left", va="bottom", fontsize=12, fontweight="bold")
    ax_c.text(-0.14, 1.02, "C", transform=ax_c.transAxes, ha="left", va="bottom", fontsize=12, fontweight="bold")

    left = ax_a.inset_axes([0.02, 0.06, 0.30, 0.86])
    center = ax_a.inset_axes([0.35, 0.06, 0.30, 0.86])
    right = ax_a.inset_axes([0.68, 0.06, 0.30, 0.86])
    _draw_selective_panel(left, "non-routed node", False, "keep low-order prediction")
    _draw_selective_panel(center, "routed hard node", True, "activate high-order residual correction")
    _draw_selective_panel(right, "routed + filtered support", True, "prefer stable non-routed support members")
    ax_a.text(0.5, 0.98, "same $x_{new}$ construction, different residual use by routing", ha="center", va="top", fontsize=10, color=COLORS["gray_text"])

    draw_conflict_panel(ax_b, conflict_rows)
    draw_support_ablation_panel(ax_c, support_rows)

    save_fig(fig, output_path)
    return output_path
