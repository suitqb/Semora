"""Plotting module — benchmark result visualisations for Semora."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor": "#1a1d27",
    "axes.edgecolor": "#3a3d4d",
    "axes.labelcolor": "#c8ccd8",
    "xtick.color": "#c8ccd8",
    "ytick.color": "#c8ccd8",
    "text.color": "#e0e3f0",
    "grid.color": "#2a2d3d",
    "grid.linewidth": 0.6,
    "legend.facecolor": "#1a1d27",
    "legend.edgecolor": "#3a3d4d",
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
})

# ── Color palette (one color per model, two shades per N) ────────────────────
_PALETTE = [
    "#6c8ef5",  # blue
    "#f57c6c",  # coral
    "#6cf5a8",  # teal
    "#f5d06c",  # gold
    "#c86cf5",  # purple
    "#6cdff5",  # cyan
]

_UNRELIABLE = {"age"}  # fields known to be unreliable (flagged with ⚠)


def _load(run_dir: Path) -> list[dict]:
    path = run_dir / "raw" / "scores.json"
    if not path.exists():
        raise FileNotFoundError(f"scores.json not found in {run_dir / 'raw'}")
    with open(path) as f:
        return json.load(f)


def _label(s: dict) -> str:
    return f"{s['model_name']} N={s['window_size']}"


def _model_colors(summaries: list[dict]) -> dict[str, str]:
    """Assign one base color per model name."""
    models = sorted({s["model_name"] for s in summaries})
    return {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(models)}


def _darken(hex_color: str, factor: float = 0.6) -> str:
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return f"#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}"


# ── Plot 1 — F1 overview bar chart ───────────────────────────────────────────
def plot_f1_overview(summaries: list[dict], output: Path) -> None:
    labels  = [_label(s) for s in summaries]
    f1_ctx  = [s["f1_context"]      for s in summaries]
    f1_ped  = [s["f1_pedestrians"]  for s in summaries]
    f1_veh  = [s["f1_vehicles"]     for s in summaries]
    parse   = [s["parse_success_rate"] for s in summaries]

    x = np.arange(len(labels))
    width = 0.22

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(max(10, len(labels) * 1.4), 9),
                                   gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("F1 Scores Overview", fontsize=15, fontweight="bold", y=0.98)

    mc = _model_colors(summaries)
    colors_ctx = [mc[s["model_name"]] for s in summaries]
    colors_ped = [_darken(mc[s["model_name"]], 0.75) for s in summaries]
    colors_veh = [_darken(mc[s["model_name"]], 0.5) for s in summaries]

    bars_ctx = ax.bar(x - width, f1_ctx, width, label="Context (simple_context)", color=colors_ctx, alpha=0.9)
    bars_ped = ax.bar(x,         f1_ped, width, label="Pedestrians (avg)",        color=colors_ped, alpha=0.9)
    bars_veh = ax.bar(x + width, f1_veh, width, label="Vehicles (avg)",           color=colors_veh, alpha=0.9)

    for bars in (bars_ctx, bars_ped, bars_veh):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7, color="#c8ccd8")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.1)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)

    # Parse success rate subplot
    ax2.bar(x, parse, color=[mc[s["model_name"]] for s in summaries], alpha=0.85)
    for i, v in enumerate(parse):
        ax2.text(i, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=8, color="#c8ccd8")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Parse rate")
    ax2.set_ylim(0, 1.15)
    ax2.yaxis.grid(True)
    ax2.set_axisbelow(True)
    ax2.set_title("Parse Success Rate", fontsize=11)

    plt.tight_layout()
    path = output / "f1_overview.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ── Plot 2 — Pedestrian field detail ─────────────────────────────────────────
def plot_person_fields(summaries: list[dict], output: Path) -> None:
    fields = ["atomic_action", "simple_context", "communicative", "transporting", "age"]
    labels = [_label(s) for s in summaries]
    mc = _model_colors(summaries)

    fig, axes = plt.subplots(1, len(fields), figsize=(len(fields) * 3.2, 6), sharey=True)
    fig.suptitle("Pedestrian Fields — Precision / Recall / F1", fontsize=13, fontweight="bold")

    x = np.arange(len(labels))
    w = 0.26

    for ax, field in zip(axes, fields):
        prec = [s["person_fields"][field]["precision"] for s in summaries]
        rec  = [s["person_fields"][field]["recall"]    for s in summaries]
        f1   = [s["person_fields"][field]["f1"]        for s in summaries]
        colors = [mc[s["model_name"]] for s in summaries]

        ax.bar(x - w, prec, w, label="Precision", color=colors, alpha=0.55)
        ax.bar(x,     rec,  w, label="Recall",    color=colors, alpha=0.75)
        ax.bar(x + w, f1,   w, label="F1",        color=colors, alpha=0.95)

        unreliable_mark = " ⚠" if field in _UNRELIABLE else ""
        ax.set_title(f"{field.replace('_', ' ')}{unreliable_mark}", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1.1)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Score")
    # Shared legend
    handles = [
        mpatches.Patch(facecolor="#888", alpha=0.55, label="Precision"),
        mpatches.Patch(facecolor="#888", alpha=0.75, label="Recall"),
        mpatches.Patch(facecolor="#888", alpha=0.95, label="F1"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=9, bbox_to_anchor=(1.0, 0.98))
    plt.tight_layout()
    path = output / "person_fields.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ── Plot 3 — Vehicle field detail ────────────────────────────────────────────
def plot_vehicle_fields(summaries: list[dict], output: Path) -> None:
    fields = ["motion_status", "trunk_open", "doors_open"]
    labels = [_label(s) for s in summaries]
    mc = _model_colors(summaries)

    fig, axes = plt.subplots(1, len(fields), figsize=(len(fields) * 3.5, 6), sharey=True)
    fig.suptitle("Vehicle Fields — Precision / Recall / F1", fontsize=13, fontweight="bold")

    x = np.arange(len(labels))
    w = 0.26

    for ax, field in zip(axes, fields):
        prec = [s["vehicle_fields"][field]["precision"] for s in summaries]
        rec  = [s["vehicle_fields"][field]["recall"]    for s in summaries]
        f1   = [s["vehicle_fields"][field]["f1"]        for s in summaries]
        colors = [mc[s["model_name"]] for s in summaries]

        ax.bar(x - w, prec, w, label="Precision", color=colors, alpha=0.55)
        ax.bar(x,     rec,  w, label="Recall",    color=colors, alpha=0.75)
        ax.bar(x + w, f1,   w, label="F1",        color=colors, alpha=0.95)

        ax.set_title(field.replace("_", " "), fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1.1)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Score")
    handles = [
        mpatches.Patch(facecolor="#888", alpha=0.55, label="Precision"),
        mpatches.Patch(facecolor="#888", alpha=0.75, label="Recall"),
        mpatches.Patch(facecolor="#888", alpha=0.95, label="F1"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=9, bbox_to_anchor=(1.0, 0.98))
    plt.tight_layout()
    path = output / "vehicle_fields.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ── Plot 4 — Radar chart per model ───────────────────────────────────────────
def plot_radar(summaries: list[dict], output: Path) -> None:
    """One radar per (model, N). All fields on the same chart."""
    all_fields = [
        "atomic_action", "simple_context", "communicative", "transporting", "age",
        "motion_status", "trunk_open", "doors_open",
    ]
    field_labels = [f.replace("_", "\n") for f in all_fields]

    n_fields = len(all_fields)
    angles = np.linspace(0, 2 * np.pi, n_fields, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    mc = _model_colors(summaries)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(field_labels, fontsize=8, color="#c8ccd8")
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7, color="#888")
    ax.set_ylim(0, 1)
    ax.grid(color="#2a2d3d", linewidth=0.8)
    ax.spines["polar"].set_color("#3a3d4d")

    for s in summaries:
        vals = []
        for f in all_fields:
            if f in s["person_fields"]:
                vals.append(s["person_fields"][f]["f1"])
            else:
                vals.append(s["vehicle_fields"][f]["f1"])
        vals += vals[:1]

        color = mc[s["model_name"]]
        if s["window_size"] > min(x["window_size"] for x in summaries):
            color = _darken(color, 0.7)
            ls = "--"
        else:
            ls = "-"

        ax.plot(angles, vals, ls, linewidth=1.8, color=color, label=_label(s))
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_title("Per-field F1 — All models", fontsize=13, fontweight="bold",
                 pad=20, color="#e0e3f0")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    plt.tight_layout()
    path = output / "radar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ── Plot 5 — LLM Judge scores ─────────────────────────────────────────────────
def plot_judge(summaries: list[dict], output: Path) -> None:
    judge_summaries = [s for s in summaries if s.get("avg_judge_overall") is not None]
    if not judge_summaries:
        print("  no judge scores found — skipping judge plot")
        return

    labels     = [_label(s) for s in judge_summaries]
    comp       = [s["avg_judge_completeness"]      for s in judge_summaries]
    richness   = [s["avg_judge_semantic_richness"] for s in judge_summaries]
    spatial    = [s["avg_judge_spatial_relations"] for s in judge_summaries]
    overall    = [s["avg_judge_overall"]           for s in judge_summaries]

    x = np.arange(len(labels))
    w = 0.18
    mc = _model_colors(judge_summaries)

    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.6), 6))
    fig.suptitle("LLM Judge Scores", fontsize=14, fontweight="bold")

    colors = [mc[s["model_name"]] for s in judge_summaries]

    ax.bar(x - 1.5 * w, comp,    w, label="Completeness",     color="#6c8ef5", alpha=0.9)
    ax.bar(x - 0.5 * w, richness, w, label="Semantic richness", color="#6cf5a8", alpha=0.9)
    ax.bar(x + 0.5 * w, spatial,  w, label="Spatial relations", color="#f5d06c", alpha=0.9)
    ax.bar(x + 1.5 * w, overall,  w, label="Overall",           color="#f57c6c", alpha=0.9)

    for i, v in enumerate(overall):
        ax.text(i + 1.5 * w, v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8, color="#e0e3f0", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score (0–1)")
    ax.set_ylim(0, 1.1)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = output / "judge_scores.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ── Plot 6 — Latency comparison ───────────────────────────────────────────────
def plot_latency(summaries: list[dict], output: Path) -> None:
    labels  = [_label(s) for s in summaries]
    latency = [s["avg_latency_s"] for s in summaries]
    mc = _model_colors(summaries)
    colors = [mc[s["model_name"]] for s in summaries]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.4), 5))
    fig.suptitle("Average Latency per Window", fontsize=13, fontweight="bold")

    x = np.arange(len(labels))
    bars = ax.bar(x, latency, color=colors, alpha=0.9)
    for bar, v in zip(bars, latency):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3, f"{v:.1f}s",
                ha="center", va="bottom", fontsize=9, color="#c8ccd8")

    ax.set_ylabel("Seconds")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = output / "latency.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────
def run_all_plots(run_dir: Path, output_dir: Path | None = None) -> Path:
    """Generate all plots for a run. Returns the output directory."""
    run_dir = run_dir.resolve()
    output  = (output_dir or run_dir / "report" / "plots").resolve()
    output.mkdir(parents=True, exist_ok=True)

    summaries = _load(run_dir)
    print(f"  [{run_dir.name}] loaded {len(summaries)} configs → {output}/")

    plot_f1_overview(summaries, output)
    plot_person_fields(summaries, output)
    plot_vehicle_fields(summaries, output)
    plot_radar(summaries, output)
    plot_judge(summaries, output)
    plot_latency(summaries, output)

    print(f"  Done — {len(list(output.glob('*.png')))} plots saved")
    return output
