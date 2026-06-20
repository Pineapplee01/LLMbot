from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches

from paper_plot_style import apply_publication_style, save_fig_bundle


FIG_DIR = Path(__file__).resolve().parent
DATA_JSON = FIG_DIR / "data" / "casestudy_node11654_evidence.json"

BOT = "#C84C4C"
HUMAN = "#4C78A8"
GREEN = "#2F855A"
ORANGE = "#D9822B"
GRAY = "#5A6472"
PANEL_EDGE = "#9AA5B1"


def _wrap(text: str, width: int, max_lines: int | None = None) -> str:
    lines = textwrap.wrap(text, width=width, break_long_words=False, replace_whitespace=False)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(". ") + "..."
    return "\n".join(lines)


def _label_color(label: str) -> str:
    return BOT if label == "Bot" else HUMAN


def _round_box(ax, xy, width, height, text="", fc="#FFFFFF", ec=PANEL_EDGE, lw=1.0, fontsize=8, bold=False):
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(box)
    if text:
        ax.text(
            xy[0] + 0.012,
            xy[1] + height - 0.025,
            text,
            ha="left",
            va="top",
            fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            color="#182230",
        )
    return box


def _node(ax, x, y, label, color, r=0.035, sublabel=None):
    ax.add_patch(patches.Circle((x, y), r, facecolor=color, edgecolor="white", linewidth=1.0, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=8, color="white", fontweight="bold", zorder=4)
    if sublabel:
        ax.text(x, y - r - 0.018, sublabel, ha="center", va="top", fontsize=6.5, color="#475467")


def _confidence_block(ax, x, y, w, h, branch):
    pred = branch["prediction"]
    correct = bool(branch["correct"])
    edge = GREEN if correct else BOT
    _round_box(ax, (x, y), w, h, fc="#FFFFFF", ec=edge, lw=1.4)
    ax.text(x + 0.014, y + h - 0.024, branch["name"], ha="left", va="top", fontsize=7.5, fontweight="bold")
    ax.text(x + w - 0.014, y + h - 0.024, f"conf {float(branch['true_confidence']):.3f}", ha="right", va="top", fontsize=6.7, color="#667085")
    ax.text(x + 0.014, y + h - 0.060, "prediction", ha="left", va="top", fontsize=6.6, color="#667085")
    ax.text(x + 0.102, y + h - 0.060, pred, ha="left", va="top", fontsize=7.4, color=_label_color(pred), fontweight="bold")
    ax.text(x + w - 0.014, y + h - 0.060, "correct" if correct else "wrong", ha="right", va="top", fontsize=6.9, color=edge)
    ax.text(
        x + 0.014,
        y + 0.023,
        f"margin true-other {float(branch['margin_true_minus_other']):+.3f}",
        ha="left",
        va="bottom",
        fontsize=6.5,
        color="#667085",
    )


def build() -> None:
    apply_publication_style()
    data = json.loads(DATA_JSON.read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(11.4, 6.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Panel containers.
    _round_box(ax, (0.025, 0.05), 0.330, 0.90, fc="#F7FBFF", ec=PANEL_EDGE, lw=1.2)
    _round_box(ax, (0.380, 0.05), 0.300, 0.90, fc="#FAFAF7", ec=PANEL_EDGE, lw=1.2)
    _round_box(ax, (0.705, 0.05), 0.270, 0.90, fc="#F8FBF8", ec=PANEL_EDGE, lw=1.2)
    ax.text(0.045, 0.925, "Target Evidence", fontsize=9.6, fontweight="bold", ha="left", va="top")
    ax.text(0.400, 0.925, "Local / Support Neighborhood", fontsize=9.6, fontweight="bold", ha="left", va="top")
    ax.text(0.725, 0.925, "Branch Evidence", fontsize=9.6, fontweight="bold", ha="left", va="top")

    target = data["target"]
    case = data["case"]
    header = (
        f"node {case['node_index']}  |  {target['user_id']}  |  @{target['username']}\n"
        f"gold: {case['gold_name']}    risk: {case['risk_score']:.3f}    routed"
    )
    _round_box(ax, (0.045, 0.805), 0.285, 0.085, header, fc="#FFFFFF", ec=BOT, lw=1.2, fontsize=7.0, bold=True)

    desc = f"Description:\n{_wrap(target['description'], 42, max_lines=3)}"
    _round_box(ax, (0.045, 0.650), 0.285, 0.130, desc, fc="#FFF5D9", ec="#C59F3F", fontsize=6.7, bold=True)

    meta = (
        "Metadata:\n"
        f"created: {target['created_at'][:16]}\n"
        f"followers/following: {target['followers_count']} / {target['following_count']}\n"
        f"tweets/listed: {target['tweet_count']} / {target['listed_count']}\n"
        f"verified/protected: {target['verified']} / {target['protected']}"
    )
    _round_box(ax, (0.045, 0.455), 0.285, 0.165, meta, fc="#FBE7D9", ec="#C98962", fontsize=6.9, bold=True)

    tweets = "\n".join([f"{idx + 1}. {_wrap(tweet, 44, max_lines=2)}" for idx, tweet in enumerate(target["tweets"][:4])])
    _round_box(ax, (0.045, 0.085), 0.285, 0.335, "Tweets:\n" + tweets, fc="#DFF1FF", ec="#6A9EC3", fontsize=6.4, bold=True)

    # Neighborhood graph.
    cx, cy = 0.530, 0.535
    _node(ax, cx, cy, "T", BOT, r=0.045, sublabel="11654")
    ax.text(cx, cy + 0.067, "target", ha="center", va="bottom", fontsize=7.2, color="#475467")

    rel_positions = [(0.455, 0.700), (0.610, 0.700)]
    for pos, nb in zip(rel_positions, data["relation_neighbors"]):
        color = _label_color(nb["label_name"])
        ax.plot([cx, pos[0]], [cy, pos[1]], color="#344054", linewidth=1.1, linestyle="-", zorder=1)
        _node(ax, pos[0], pos[1], nb["label_name"][0], color, r=0.031, sublabel=nb["node_index"])
    ax.text(0.525, 0.760, f"relation bot ratio = {case['relation_bot_ratio']:.3f}", ha="center", va="center", fontsize=7.6)

    support_positions = [
        (0.430, 0.390),
        (0.465, 0.315),
        (0.520, 0.280),
        (0.585, 0.300),
        (0.630, 0.365),
        (0.620, 0.455),
        (0.565, 0.405),
        (0.475, 0.455),
    ]
    for pos, nb in zip(support_positions, data["support_neighbors"]):
        color = _label_color(nb["label_name"])
        ax.plot([cx, pos[0]], [cy, pos[1]], color="#667085", linewidth=0.9, linestyle=(0, (3, 3)), zorder=1)
        _node(ax, pos[0], pos[1], nb["label_name"][0], color, r=0.026, sublabel=nb["rank"])
    ax.text(0.525, 0.210, f"KNN support bot ratio = {case['support_bot_ratio']:.3f}", ha="center", va="center", fontsize=7.6)
    ax.text(0.525, 0.175, "support labels: B B B B B H B B", ha="center", va="center", fontsize=7.4, color="#475467")

    # Branch evidence.
    y = 0.760
    for branch in data["branch_evidence"]:
        _confidence_block(ax, 0.725, y, 0.225, 0.115, branch)
        y -= 0.145
    _round_box(
        ax,
        (0.725, 0.125),
        0.225,
        0.110,
        "Final decision:\nRouted-only residual -> Bot\nLow-order error is corrected.",
        fc="#FFFFFF",
        ec=GREEN,
        lw=1.5,
        fontsize=6.8,
        bold=True,
    )

    save_fig_bundle(fig, [FIG_DIR / "casestudy.pdf", FIG_DIR / "casestudy.png", FIG_DIR / "casestudy.svg"])


if __name__ == "__main__":
    build()
    print(f"Wrote {FIG_DIR / 'casestudy.pdf'}")
    print(f"Wrote {FIG_DIR / 'casestudy.png'}")
    print(f"Wrote {FIG_DIR / 'casestudy.svg'}")
