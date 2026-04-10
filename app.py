"""Semora — Streamlit analysis interface.

Launch: streamlit run app.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Semora",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Custom CSS — dark theme aligned with plot palette (#0f1117)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Base ── */
.stApp, section.main, .block-container {
    background-color: #0f1117 !important;
    color: #c8ccd8 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #1a1d27 !important;
    border-right: 1px solid #2a2d3d;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #c8ccd8 !important;
}
[data-testid="stSidebar"] .stSelectbox > div,
[data-testid="stSidebar"] .stMultiSelect > div {
    background-color: #131620 !important;
    border-color: #2a2d3d !important;
}

/* ── Headers ── */
h1, h2, h3, h4 { color: #e0e3f0 !important; }
h1 { border-bottom: 1px solid #2a2d3d; padding-bottom: 0.4rem; }

/* ── Tabs ── */
[data-testid="stTabs"] button {
    color: #888 !important;
    border-bottom: 2px solid transparent;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #6c8ef5 !important;
    border-bottom-color: #6c8ef5 !important;
    background-color: transparent !important;
}
[data-testid="stTabs"] button:hover { color: #c8ccd8 !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background-color: #1a1d27;
    border: 1px solid #2a2d3d;
    border-radius: 10px;
    padding: 16px 20px;
}
[data-testid="metric-container"] label {
    color: #888 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e0e3f0 !important;
    font-size: 1.6rem !important;
    font-weight: 700;
}
[data-testid="stMetricDelta"] svg { display: none; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid #2a2d3d; border-radius: 8px; }

/* ── Info / warning / error ── */
[data-testid="stAlert"] {
    background-color: #1a1d27 !important;
    border-color: #2a2d3d !important;
    color: #c8ccd8 !important;
}

/* ── Divider ── */
hr { border-color: #2a2d3d !important; }

/* ── Caption ── */
.stCaption { color: #555 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

RUNS_ROOT      = Path("runs")
PERSON_FIELDS  = ["atomic_action", "simple_context", "communicative", "transporting", "age"]
VEHICLE_FIELDS = ["motion_status", "trunk_open", "doors_open"]
ALL_FIELDS     = PERSON_FIELDS + VEHICLE_FIELDS
PALETTE        = px.colors.qualitative.Plotly

# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def list_runs() -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for mode_dir in sorted(RUNS_ROOT.iterdir()) if RUNS_ROOT.exists() else []:
        if not mode_dir.is_dir():
            continue
        runs = sorted(
            [r for r in mode_dir.iterdir() if r.is_dir() and (r / "raw").exists()],
            reverse=True,
        )
        if runs:
            result[mode_dir.name] = runs
    return result


@st.cache_data
def load_scores(run_dir: Path) -> list[dict]:
    path = run_dir / "raw" / "scores.json"
    return json.loads(path.read_text()) if path.exists() else []


@st.cache_data
def load_jsonl(run_dir: Path, pattern: str) -> list[dict]:
    records = []
    for path in sorted((run_dir / "raw").glob(pattern)):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _color_f1(val: float | None) -> str:
    if val is None:
        return ""
    if val >= 0.6:
        return "background-color: #1a3a1a; color: #6cf5a8"
    if val >= 0.35:
        return "background-color: #3a3a1a; color: #f5d06c"
    return "background-color: #3a1a1a; color: #f57c6c"


def _style_f1_cols(df: pd.DataFrame, cols: list[str]) -> pd.io.formats.style.Styler:
    def _row(row):
        styles = [""] * len(row)
        for col in cols:
            if col in df.columns:
                styles[df.columns.get_loc(col)] = _color_f1(row.get(col))
        return styles
    return df.style.apply(_row, axis=1)


def _plotly_cfg() -> dict:
    return dict(
        paper_bgcolor="#0f1117",
        plot_bgcolor="#1a1d27",
        font_color="#c8ccd8",
        margin=dict(t=40, b=20, l=20, r=20),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Metric cards
# ─────────────────────────────────────────────────────────────────────────────

def _metric_cards(summaries: list[dict], ref_summaries: list[dict] | None = None) -> None:
    """Show key metrics as st.metric cards for each config.

    If ref_summaries provided, delta = value − ref value (first ref config).
    """
    if not summaries:
        return

    metrics = [
        ("Parse %",       "parse_success_rate",          100,   "%"),
        ("F1 Ped",        "f1_pedestrians",               1,    ""),
        ("F1 Veh",        "f1_vehicles",                  1,    ""),
        ("Judge Overall", "avg_judge_overall",             1,    ""),
        ("Latency (s)",   "avg_latency_s",                1,    "s"),
    ]

    ref_map: dict[str, float | None] = {}
    if ref_summaries:
        r = ref_summaries[0]
        for label, key, scale, _ in metrics:
            v = r.get(key)
            ref_map[label] = v * scale if v is not None else None

    for s in summaries:
        label_cfg = f"**{s['model_name']}** · N={s['window_size']}"
        st.markdown(label_cfg)
        cols = st.columns(len(metrics))
        for col, (label, key, scale, unit) in zip(cols, metrics):
            val = s.get(key)
            display = f"{val * scale:.2f}{unit}" if val is not None else "—"
            delta = None
            if ref_summaries and label in ref_map and ref_map[label] is not None and val is not None:
                delta = round(val * scale - ref_map[label], 3)
            col.metric(label, display, delta=f"{delta:+.3f}" if delta is not None else None)
        st.markdown("")


# ─────────────────────────────────────────────────────────────────────────────
# Extraction views
# ─────────────────────────────────────────────────────────────────────────────

def view_scores_overview(summaries: list[dict]) -> None:
    st.subheader("Overview")
    _metric_cards(summaries)
    st.divider()

    rows = []
    for s in summaries:
        parse = s.get("parse_success_rate")
        rows.append({
            "Model":         s["model_name"],
            "N":             s["window_size"],
            "Parse %":       round(parse * 100, 1) if parse is not None else None,
            "F1 Context":    s.get("f1_context"),
            "F1 Ped":        s.get("f1_pedestrians"),
            "F1 Veh":        s.get("f1_vehicles"),
            "Completeness":  s.get("avg_judge_completeness"),
            "Sem. Richness": s.get("avg_judge_semantic_richness"),
            "Spatial":       s.get("avg_judge_spatial_relations"),
            "Judge Overall": s.get("avg_judge_overall"),
            "Latency (s)":   round(s["avg_latency_s"], 2),
            "Prompt tok.":   s.get("total_prompt_tokens"),
            "Compl. tok.":   s.get("total_completion_tokens"),
        })
    df = pd.DataFrame(rows)
    f1_cols = ["F1 Context", "F1 Ped", "F1 Veh", "Completeness",
               "Sem. Richness", "Spatial", "Judge Overall"]
    st.dataframe(
        _style_f1_cols(df, f1_cols).format(
            {c: "{:.3f}" for c in f1_cols if c in df.columns}, na_rep="—"
        ),
        width="stretch",
        height=min(400, 80 + len(rows) * 38),
    )


def view_field_detail(summaries: list[dict]) -> None:
    st.subheader("Per-field Precision / Recall / F1")
    tab_ped, tab_veh, tab_radar = st.tabs(["Pedestrians", "Vehicles", "Radar"])

    for tab, fields, key in [
        (tab_ped, PERSON_FIELDS, "person_fields"),
        (tab_veh, VEHICLE_FIELDS, "vehicle_fields"),
    ]:
        with tab:
            rows = []
            for s in summaries:
                label = f"{s['model_name']} N={s['window_size']}"
                for field, vals in s.get(key, {}).items():
                    rows.append({
                        "Config": label, "Field": field,
                        "Precision": vals["precision"],
                        "Recall":    vals["recall"],
                        "F1":        vals["f1"],
                    })
            if not rows:
                st.info("No data.")
                continue
            df = pd.DataFrame(rows)
            fig = px.bar(
                df.melt(id_vars=["Config", "Field"],
                        value_vars=["Precision", "Recall", "F1"],
                        var_name="Metric", value_name="Score"),
                x="Field", y="Score", color="Config", barmode="group",
                facet_col="Metric", range_y=[0, 1.05],
                template="plotly_dark", height=400,
            )
            fig.update_layout(**{**_plotly_cfg(), "margin": dict(t=40, b=20)})
            st.plotly_chart(fig, width="stretch")
            st.dataframe(
                df.pivot_table(index=["Config", "Field"],
                               values=["Precision", "Recall", "F1"]).round(3),
                width="stretch",
            )

    # 3. Radar chart
    with tab_radar:
        fig = go.Figure()
        angles = ALL_FIELDS + [ALL_FIELDS[0]]  # close the polygon
        for i, s in enumerate(summaries):
            label = f"{s['model_name']} N={s['window_size']}"
            vals = []
            for f in ALL_FIELDS:
                if f in s.get("person_fields", {}):
                    vals.append(s["person_fields"][f]["f1"])
                elif f in s.get("vehicle_fields", {}):
                    vals.append(s["vehicle_fields"][f]["f1"])
                else:
                    vals.append(0.0)
            vals += [vals[0]]
            color = PALETTE[i % len(PALETTE)]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=angles, name=label,
                fill="toself", fillcolor=color,
                line=dict(color=color, width=2),
                opacity=0.6,
            ))
        fig.update_layout(
            polar=dict(
                bgcolor="#1a1d27",
                radialaxis=dict(visible=True, range=[0, 1],
                                color="#555", gridcolor="#2a2d3d"),
                angularaxis=dict(color="#c8ccd8", gridcolor="#2a2d3d"),
            ),
            paper_bgcolor="#0f1117",
            font_color="#c8ccd8",
            height=500,
            legend=dict(bgcolor="#1a1d27", bordercolor="#2a2d3d"),
            margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig, width="stretch")


def view_judge_scores(summaries: list[dict]) -> None:
    judge_sums = [s for s in summaries if s.get("avg_judge_overall") is not None]
    if not judge_sums:
        st.info("No LLM judge scores in this run.")
        return
    st.subheader("LLM Judge Scores")
    rows = []
    for s in judge_sums:
        label = f"{s['model_name']} N={s['window_size']}"
        rows.append({
            "Config":        label,
            "Completeness":  s["avg_judge_completeness"],
            "Sem. Richness": s["avg_judge_semantic_richness"],
            "Spatial":       s["avg_judge_spatial_relations"],
            "Overall":       s["avg_judge_overall"],
        })
    df = pd.DataFrame(rows)
    fig = px.bar(
        df.melt(id_vars="Config", var_name="Criterion", value_name="Score"),
        x="Config", y="Score", color="Criterion", barmode="group",
        range_y=[0, 1.05], template="plotly_dark", height=380,
    )
    fig.update_layout(**_plotly_cfg())
    st.plotly_chart(fig, width="stretch")


def view_latency(summaries: list[dict]) -> None:
    st.subheader("Average Latency per Window")
    rows = [{"Config": f"{s['model_name']} N={s['window_size']}",
             "Latency (s)": s["avg_latency_s"]} for s in summaries]
    df = pd.DataFrame(rows)
    fig = px.bar(df, x="Config", y="Latency (s)", color="Config",
                 template="plotly_dark", height=320)
    fig.update_layout(**_plotly_cfg(), showlegend=False)
    st.plotly_chart(fig, width="stretch")


def view_temporal_consistency(run_dir: Path) -> None:
    st.subheader("Inter-window Temporal Consistency")
    records = load_jsonl(run_dir, "*_parsed_outputs.jsonl")
    if not records:
        st.info("No parsed_outputs.jsonl — enable save_parsed_outputs in benchmark.yaml.")
        return

    _MATCH_THRESHOLD = 0.45

    def _jaccard(a: str, b: str) -> float:
        sa, sb = set(a.lower().split()), set(b.lower().split())
        return len(sa & sb) / len(sa | sb) if sa and sb else 0.0

    def _center_peds(frames: list[dict], N: int) -> list[dict]:
        if not frames:
            return []
        ci = min(N // 2 if N % 2 == 1 else N - 1, len(frames) - 1)
        return frames[ci].get("pedestrians", [])

    group_windows: dict[tuple, list] = defaultdict(list)
    for rec in records:
        if not rec.get("parse_success"):
            continue
        peds = _center_peds(rec["parsed"].get("frames", []), rec["N"])
        if peds:
            group_windows[(rec["model"], rec["clip_id"], rec["N"])].append(
                {"center_frame": rec["center_frame"], "pedestrians": peds}
            )

    agg: dict[tuple, dict] = defaultdict(lambda: {
        "n_pairs": 0, "n_matches": 0, "n_candidates": 0,
        "field_agrees": defaultdict(list),
    })
    for (model, clip_id, N), windows in group_windows.items():
        windows = sorted(windows, key=lambda w: w["center_frame"])
        key = (model, N)
        for i in range(len(windows)):
            for j in range(i + 1, len(windows)):
                pi, pj = windows[i]["pedestrians"], windows[j]["pedestrians"]
                if not pi or not pj:
                    continue
                used, matches = set(), []
                for pa in pi:
                    hint_a = pa.get("track_hint", "").strip()
                    if not hint_a:
                        continue
                    best_sim, best_j = 0.0, -1
                    for jj, pb in enumerate(pj):
                        if jj in used:
                            continue
                        sim = _jaccard(hint_a, pb.get("track_hint", "").strip())
                        if sim > best_sim:
                            best_sim, best_j = sim, jj
                    if best_sim >= _MATCH_THRESHOLD and best_j >= 0:
                        matches.append((pa, pj[best_j]))
                        used.add(best_j)
                agg[key]["n_pairs"]      += 1
                agg[key]["n_matches"]    += len(matches)
                agg[key]["n_candidates"] += len(pi)
                for pa, pb in matches:
                    for field in PERSON_FIELDS:
                        va, vb = pa.get(field), pb.get(field)
                        if va and vb:
                            agg[key]["field_agrees"][field].append(va == vb)

    if not agg:
        st.warning("Not enough windows per clip for pairwise comparison.")
        return

    rows = []
    for (model, N), data in sorted(agg.items()):
        if data["n_pairs"] == 0:
            continue
        reid = data["n_matches"] / data["n_candidates"] if data["n_candidates"] else 0.0
        row = {"Model": model, "N": N, "Re-ID rate": round(reid, 3),
               "Pairs": data["n_pairs"]}
        for field in PERSON_FIELDS:
            agrees = data["field_agrees"].get(field, [])
            row[field] = round(sum(agrees) / len(agrees), 3) if agrees else None
        rows.append(row)

    if not rows:
        st.info("No valid pairs found.")
        return

    df = pd.DataFrame(rows)
    agree_cols = PERSON_FIELDS + ["Re-ID rate"]
    st.dataframe(
        _style_f1_cols(df, agree_cols).format(
            {c: "{:.1%}" for c in agree_cols if c in df.columns}, na_rep="—"
        ),
        width="stretch",
    )
    st.caption(
        f"Re-ID threshold: Jaccard ≥ {_MATCH_THRESHOLD:.0%}. "
        "Dynamic fields (action, context): lower agreement is normal. "
        "**age** should be stable across windows."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Complexity views
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def _load_complexity_data(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame_records = load_jsonl(run_dir, "*_frame_scores.jsonl")
    judge_records = load_jsonl(run_dir, "*_judge_outputs.jsonl")

    frame_rows = []
    for rec in frame_records:
        n_ent = rec.get("n_entities_gt") or (
            rec.get("n_persons_gt", 0) + rec.get("n_vehicles_gt", 0)
        )
        ped_f1s = [s["f1"] for s in rec.get("person_scores", {}).values()] if rec.get("parse_success") else []
        veh_f1s = [s["f1"] for s in rec.get("vehicle_scores", {}).values()] if rec.get("parse_success") else []
        n_pp, n_pg = rec.get("n_persons_pred"), rec.get("n_persons_gt", 0)
        n_vp, n_vg = rec.get("n_vehicles_pred"), rec.get("n_vehicles_gt", 0)
        frame_rows.append({
            "model":         rec["model_name"],
            "window_size":   rec["window_size"],
            "n_entities_gt": n_ent,
            "n_persons_gt":  n_pg,
            "n_vehicles_gt": n_vg,
            "f1_ped":  sum(ped_f1s) / len(ped_f1s) if ped_f1s else None,
            "f1_veh":  sum(veh_f1s) / len(veh_f1s) if veh_f1s else None,
            "det_ped": min(n_pp / n_pg, 1.0) if n_pg and n_pp is not None else None,
            "det_veh": min(n_vp / n_vg, 1.0) if n_vg and n_vp is not None else None,
        })

    judge_rows = []
    for rec in judge_records:
        n_ent = rec.get("n_entities_gt") or (
            rec.get("n_persons_gt", 0) + rec.get("n_vehicles_gt", 0)
        )
        scores = rec.get("scores", {})
        judge_rows.append({
            "model":         rec["model"],
            "window_size":   rec["N"],
            "n_entities_gt": n_ent,
            "completeness":  scores.get("completeness"),
            "overall":       scores.get("overall"),
        })

    return (
        pd.DataFrame(frame_rows) if frame_rows else pd.DataFrame(),
        pd.DataFrame(judge_rows) if judge_rows else pd.DataFrame(),
    )


def view_complexity_table(frame_df: pd.DataFrame, judge_df: pd.DataFrame) -> None:
    st.subheader("Complexity Degradation by Entity Count")
    if frame_df.empty:
        st.info("No frame_scores data.")
        return
    rows = []
    for (model, ws, n_ent), grp in frame_df.groupby(["model", "window_size", "n_entities_gt"]):
        f1_ped = grp["f1_ped"].dropna().mean()
        f1_veh = grp["f1_veh"].dropna().mean()
        row = {"Model": model, "N": ws, "Entities (GT)": n_ent, "Frames": len(grp),
               "F1 Ped": round(f1_ped, 3) if not pd.isna(f1_ped) else None,
               "F1 Veh": round(f1_veh, 3) if not pd.isna(f1_veh) else None}
        if not judge_df.empty:
            jg = judge_df[
                (judge_df["model"] == model) &
                (judge_df["window_size"] == ws) &
                (judge_df["n_entities_gt"] == n_ent)
            ]
            row["Completeness ★"] = round(jg["completeness"].dropna().mean(), 3) if not jg.empty else None
            row["Judge Overall"]  = round(jg["overall"].dropna().mean(), 3) if not jg.empty else None
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["Model", "N", "Entities (GT)"])
    score_cols = [c for c in ["Completeness ★", "Judge Overall", "F1 Ped", "F1 Veh"] if c in df.columns]
    st.dataframe(
        _style_f1_cols(df, score_cols).format(
            {c: "{:.3f}" for c in score_cols}, na_rep="—"
        ),
        width="stretch",
        height=min(600, 80 + len(df) * 38),
    )


def _scatter_complexity(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str) -> None:
    d = df[[x, y, "model"]].dropna()
    if d.empty:
        st.info(f"No data for {title}.")
        return
    fig = px.scatter(d, x=x, y=y, color="model", trendline="ols",
                     range_y=[0, 1.05],
                     labels={x: "Entities in scene (GT)", y: ylabel},
                     title=title, template="plotly_dark", height=380)
    fig.update_layout(**_plotly_cfg())
    if y in ("det_ped", "det_veh", "completeness"):
        fig.add_hline(y=1.0, line_dash="dash", line_color="#444", opacity=0.6)
    st.plotly_chart(fig, width="stretch")


def view_detection_rate(frame_df: pd.DataFrame) -> None:
    if frame_df.empty:
        return
    st.subheader("Entity Detection Rate vs Scene Density")
    c1, c2 = st.columns(2)
    with c1:
        _scatter_complexity(frame_df, "n_persons_gt", "det_ped",
                            "Pedestrian detection rate", "Detection rate")
    with c2:
        _scatter_complexity(frame_df, "n_vehicles_gt", "det_veh",
                            "Vehicle detection rate", "Detection rate")


def view_f1_vs_entities(frame_df: pd.DataFrame) -> None:
    if frame_df.empty:
        return
    st.subheader("Attribute F1 vs Entity Count")
    c1, c2 = st.columns(2)
    with c1:
        _scatter_complexity(frame_df, "n_persons_gt", "f1_ped",
                            "Pedestrian attribute F1", "F1 Score")
    with c2:
        _scatter_complexity(frame_df, "n_vehicles_gt", "f1_veh",
                            "Vehicle attribute F1", "F1 Score")


def view_completeness_vs_entities(judge_df: pd.DataFrame) -> None:
    if judge_df.empty or "completeness" not in judge_df.columns:
        return
    st.subheader("Completeness (LLM Judge) vs Entity Count")
    _scatter_complexity(judge_df, "n_entities_gt", "completeness",
                        "Completeness vs Scene Density", "Completeness")


# ─────────────────────────────────────────────────────────────────────────────
# 4 & 5. Multi-run comparison page
# ─────────────────────────────────────────────────────────────────────────────

def _load_multi_scores(run_dirs: list[Path]) -> pd.DataFrame:
    """Load scores.json from multiple runs, tag with run_id."""
    rows = []
    for rd in run_dirs:
        for s in load_scores(rd):
            rows.append({**s, "run_id": rd.name})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def page_compare(all_runs: dict[str, list[Path]]) -> None:
    st.header("Compare Runs")

    mode = st.selectbox("Mode", list(all_runs.keys()), key="cmp_mode")
    runs = all_runs[mode]
    run_labels = [r.name for r in runs]

    sel_labels = st.multiselect(
        "Select runs to compare", run_labels,
        default=run_labels[:min(3, len(run_labels))],
        key="cmp_runs",
    )
    if len(sel_labels) < 2:
        st.info("Select at least 2 runs to compare.")
        return

    sel_dirs = [runs[run_labels.index(l)] for l in sel_labels]

    if mode == "extraction":
        df = _load_multi_scores(sel_dirs)
        if df.empty:
            st.error("No scores.json found in selected runs.")
            return

        # ── Metric comparison table ──
        st.subheader("Scores Overview — All Runs")
        display_cols = ["run_id", "model_name", "window_size", "parse_success_rate",
                        "f1_context", "f1_pedestrians", "f1_vehicles",
                        "avg_judge_overall", "avg_latency_s"]
        existing = [c for c in display_cols if c in df.columns]
        tbl = df[existing].copy()
        f1_cols = ["f1_context", "f1_pedestrians", "f1_vehicles", "avg_judge_overall"]
        fmt = {c: "{:.3f}" for c in f1_cols if c in tbl.columns}
        if "parse_success_rate" in tbl.columns:
            fmt["parse_success_rate"] = "{:.1%}"
        st.dataframe(
            _style_f1_cols(tbl, [c for c in f1_cols if c in tbl.columns]).format(
                fmt, na_rep="—",
            ),
            width="stretch",
        )

        # ── Overlay F1 bar chart ──
        st.subheader("F1 Comparison")
        melt_cols = [c for c in ["f1_pedestrians", "f1_vehicles", "f1_context"] if c in df.columns]
        if melt_cols:
            df["config"] = df["run_id"] + " · " + df["model_name"] + " N=" + df["window_size"].astype(str)
            melted = df[["config"] + melt_cols].melt(
                id_vars="config", var_name="Metric", value_name="F1"
            )
            fig = px.bar(melted, x="config", y="F1", color="Metric", barmode="group",
                         range_y=[0, 1.05], template="plotly_dark", height=420)
            fig.update_layout(**_plotly_cfg())
            st.plotly_chart(fig, width="stretch")

        # ── Radar overlay per run ──
        st.subheader("Radar — Per-field F1")
        fig = go.Figure()
        for i, rd in enumerate(sel_dirs):
            for s in load_scores(rd):
                label = f"{rd.name} · {s['model_name']} N={s['window_size']}"
                vals = []
                for f in ALL_FIELDS:
                    if f in s.get("person_fields", {}):
                        vals.append(s["person_fields"][f]["f1"])
                    elif f in s.get("vehicle_fields", {}):
                        vals.append(s["vehicle_fields"][f]["f1"])
                    else:
                        vals.append(0.0)
                vals += [vals[0]]
                color = PALETTE[i % len(PALETTE)]
                fig.add_trace(go.Scatterpolar(
                    r=vals, theta=ALL_FIELDS + [ALL_FIELDS[0]],
                    name=label, fill="toself", fillcolor=color,
                    line=dict(color=color, width=2), opacity=0.55,
                ))
        fig.update_layout(
            polar=dict(
                bgcolor="#1a1d27",
                radialaxis=dict(visible=True, range=[0, 1],
                                color="#555", gridcolor="#2a2d3d"),
                angularaxis=dict(color="#c8ccd8", gridcolor="#2a2d3d"),
            ),
            paper_bgcolor="#0f1117", font_color="#c8ccd8",
            height=520,
            legend=dict(bgcolor="#1a1d27", bordercolor="#2a2d3d"),
            margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig, width="stretch")

    elif mode == "complexity":
        # Collect frame data from all selected runs
        all_frame_dfs = []
        for i, rd in enumerate(sel_dirs):
            frame_df, _ = _load_complexity_data(rd)
            if not frame_df.empty:
                frame_df = frame_df.copy()
                frame_df["run_label"] = rd.name + " · " + frame_df["model"].astype(str)
                frame_df["_color_idx"] = i
                all_frame_dfs.append(frame_df)

        if not all_frame_dfs:
            st.warning("No frame_scores data found in the selected runs.")
            return

        combined = pd.concat(all_frame_dfs, ignore_index=True)

        def _scatter_multi(x: str, y: str, title: str, xlabel: str, ylabel: str) -> None:
            d = combined[[x, y, "run_label", "_color_idx"]].dropna()
            if d.empty:
                st.info(f"No data for {title}.")
                return
            fig = go.Figure()
            for (label, cidx), grp in d.groupby(["run_label", "_color_idx"]):
                color = PALETTE[int(cidx) % len(PALETTE)]
                fig.add_trace(go.Scatter(
                    x=grp[x], y=grp[y], mode="markers",
                    name=label,
                    marker=dict(color=color, opacity=0.5, size=7),
                ))
            fig.update_layout(
                **_plotly_cfg(),
                title=title,
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                yaxis_range=[0, 1.05],
                template="plotly_dark",
                height=400,
            )
            st.plotly_chart(fig, width="stretch")

        st.subheader("F1 vs Entity Count")
        c1, c2 = st.columns(2)
        with c1:
            _scatter_multi("n_persons_gt", "f1_ped",
                           "Pedestrian F1 vs Person Count", "Persons (GT)", "F1")
        with c2:
            _scatter_multi("n_vehicles_gt", "f1_veh",
                           "Vehicle F1 vs Vehicle Count", "Vehicles (GT)", "F1")

        st.subheader("Detection Rate vs Entity Count")
        c1, c2 = st.columns(2)
        with c1:
            _scatter_multi("n_persons_gt", "det_ped",
                           "Pedestrian Detection Rate", "Persons (GT)", "Detection rate")
        with c2:
            _scatter_multi("n_vehicles_gt", "det_veh",
                           "Vehicle Detection Rate", "Vehicles (GT)", "Detection rate")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    all_runs = list_runs()

    with st.sidebar:
        st.markdown("## 🚗 Semora")
        st.divider()

        page = st.radio("Navigation", ["Analysis", "Compare Runs"],
                        label_visibility="collapsed")
        st.divider()

    if not all_runs:
        st.warning(f"No runs found in `{RUNS_ROOT}/`. Run `python run_benchmark.py` first.")
        return

    # ── Compare page ──────────────────────────────────────────────────────────
    if page == "Compare Runs":
        page_compare(all_runs)
        return

    # ── Analysis page ─────────────────────────────────────────────────────────
    st.title("🚗 Semora — Benchmark Analysis")

    with st.sidebar:
        mode = st.selectbox("Mode", list(all_runs.keys()))
        runs = all_runs[mode]
        run_labels = [r.name for r in runs]
        selected_label = st.selectbox("Run", run_labels)
        run_dir = runs[run_labels.index(selected_label)]
        st.caption(f"`{run_dir}`")

    if mode == "extraction":
        summaries = load_scores(run_dir)
        if not summaries:
            st.error("No scores.json found in this run.")
            return

        models = sorted({s["model_name"] for s in summaries})
        ns     = sorted({s["window_size"]  for s in summaries})

        with st.sidebar:
            st.divider()
            st.subheader("Filters")
            sel_models = st.multiselect("Models", models, default=models)
            sel_ns     = st.multiselect("Window size (N)", ns, default=ns)

        filtered = [s for s in summaries
                    if s["model_name"] in sel_models and s["window_size"] in sel_ns]

        tabs = st.tabs(["Overview", "Per-field", "Judge Scores",
                        "Temporal Consistency", "Latency"])
        with tabs[0]: view_scores_overview(filtered)
        with tabs[1]: view_field_detail(filtered)
        with tabs[2]: view_judge_scores(filtered)
        with tabs[3]: view_temporal_consistency(run_dir)
        with tabs[4]: view_latency(filtered)

    elif mode == "complexity":
        frame_df, judge_df = _load_complexity_data(run_dir)
        if frame_df.empty:
            st.error("No frame_scores.jsonl found in this run.")
            return

        models = sorted(frame_df["model"].unique())
        with st.sidebar:
            st.divider()
            st.subheader("Filters")
            sel_models = st.multiselect("Models", models, default=models)

        frame_df = frame_df[frame_df["model"].isin(sel_models)]
        if not judge_df.empty:
            judge_df = judge_df[judge_df["model"].isin(sel_models)]

        tabs = st.tabs(["Complexity Table", "Detection Rate", "F1 vs Entities"])
        with tabs[0]: view_complexity_table(frame_df, judge_df)
        with tabs[1]: view_detection_rate(frame_df)
        with tabs[2]: view_f1_vs_entities(frame_df)

    else:
        st.info(f"No view implemented for mode '{mode}'.")


if __name__ == "__main__":
    main()
