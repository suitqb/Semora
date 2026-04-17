"""Semora — Streamlit analysis interface.

Launch: streamlit run app.py
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Suppress statsmodels R² divide-by-zero warnings from OLS trendlines
# (triggered when a model group has only one unique x value)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Semora",
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
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return []          # ancien format non migré — ignoré
    return data.get("results", [])


@st.cache_data
def load_run_meta(run_dir: Path) -> dict:
    path = run_dir / "raw" / "scores.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return {}          # ancien format non migré — ignoré
    return data.get("meta", {})


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
                styles[df.columns.get_loc(col)] = _color_f1(row.get(col))  # type: ignore[index]
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
            "Frames":        s.get("n_frames"),
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

    meta = load_run_meta(run_dir)
    if meta.get("tracking"):
        st.caption("Matching par `track_id` exact — fenêtres adjacentes uniquement.")
    else:
        st.warning(
            "**Tracking inactif** — matching par similarité textuelle (Jaccard). "
            "Deux piétons différents mais décrits de façon similaire peuvent être matchés. "
            "Les scores sont **potentiellement surestimés** et non comparables aux runs avec tracking.",
            icon="⚠",
        )

    _MATCH_THRESHOLD = 0.10

    def _jaccard(a: str, b: str) -> float:
        sa, sb = set(a.lower().split()), set(b.lower().split())
        return len(sa & sb) / len(sa | sb) if sa and sb else 0.0

    def _center_peds(frames: list[dict], N: int) -> list[dict]:
        if not frames:
            return []
        ci = min(N // 2 if N % 2 == 1 else N - 1, len(frames) - 1)
        return frames[ci].get("pedestrians", [])

    def _match_pedestrians(pi: list[dict], pj: list[dict]) -> list[tuple]:
        """Match pedestrians between two windows.

        Uses track_id (integer, stable across windows) when available,
        falls back to Jaccard similarity on track_hint strings.
        """
        # track_id matching: both peds have a positive track_id
        has_track_ids = any(
            isinstance(p.get("track_id"), int) and p["track_id"] > 0
            for p in pi + pj
        )
        if has_track_ids:
            id_map = {p["track_id"]: p for p in pj
                      if isinstance(p.get("track_id"), int) and p["track_id"] > 0}
            return [
                (pa, id_map[pa["track_id"]])
                for pa in pi
                if isinstance(pa.get("track_id"), int)
                and pa["track_id"] > 0
                and pa["track_id"] in id_map
            ]
        # Fallback: Jaccard on track_hint
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
        return matches

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
        "has_tracking": False,
        "field_agrees": defaultdict(list),
    })
    for (model, clip_id, N), windows in group_windows.items():
        windows = sorted(windows, key=lambda w: w["center_frame"])
        key = (model, N)
        # Comparer uniquement les fenêtres adjacentes : même entité présente → même personne
        for i in range(len(windows) - 1):
            pi, pj = windows[i]["pedestrians"], windows[i + 1]["pedestrians"]
            if not pi or not pj:
                continue
            # Determine if this pair uses track_id matching
            has_ids = any(
                isinstance(p.get("track_id"), int) and p["track_id"] > 0
                for p in pi + pj
            )
            matches = _match_pedestrians(pi, pj)
            agg[key]["n_pairs"] += 1
            agg[key]["n_matches"] += len(matches)
            if has_ids:
                # Only count pedestrians that can actually be matched (positive track_ids)
                agg[key]["n_candidates"] += len([
                    p for p in pi
                    if isinstance(p.get("track_id"), int) and p["track_id"] > 0
                ])
                agg[key]["has_tracking"] = True
            else:
                # Jaccard mode: count peds with non-empty track_hint
                agg[key]["n_candidates"] += len([
                    p for p in pi if p.get("track_hint", "").strip()
                ])
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
        if data["has_tracking"] and data["n_candidates"]:
            reid = round(data["n_matches"] / data["n_candidates"], 3)
        else:
            reid = None  # Jaccard sur texte libre = inutilisable comme Re-ID
        row = {"Model": model, "N": N, "Re-ID rate": reid, "Pairs": data["n_pairs"]}
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
        "Comparaison fenêtres **adjacentes** uniquement. "
        "**Re-ID** : track_id exact (tracking requis) — `—` sans tracking. "
        "**Accords de champs** : Jaccard ≥ "
        f"{_MATCH_THRESHOLD:.0%} sur track_hint (indicatif sans tracking — cohérence stylistique, pas identité). "
        "**age** devrait être stable · action/context naturellement plus variables."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Complexity views
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def _load_complexity_data(run_dir: Path) -> pd.DataFrame:
    records = load_jsonl(run_dir, "*_frame_scores.jsonl")
    rows = []
    for rec in records:
        n_pg = rec.get("n_persons_gt", 0)
        n_vg = rec.get("n_vehicles_gt", 0)
        n_pp = rec.get("n_persons_pred")
        n_vp = rec.get("n_vehicles_pred")
        rows.append({
            "model":         rec["model_name"],
            "window_size":   rec["window_size"],
            "n_entities_gt": rec.get("n_entities_gt") or (n_pg + n_vg),
            "n_persons_gt":  n_pg,
            "n_vehicles_gt": n_vg,
            "det_ped": min(n_pp / n_pg, 1.0) if n_pg and n_pp is not None else None,
            "det_veh": min(n_vp / n_vg, 1.0) if n_vg and n_vp is not None else None,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def view_pdr_table(frame_df: pd.DataFrame) -> None:
    st.subheader("PDR par densité de scène")
    if frame_df.empty:
        st.info("No frame_scores data.")
        return
    rows = []
    for (model, ws, n_ent), grp in frame_df.groupby(["model", "window_size", "n_entities_gt"]):
        det_ped = grp["det_ped"].dropna().mean()
        det_veh = grp["det_veh"].dropna().mean()
        rows.append({
            "Model": model, "N": ws, "Entities (GT)": n_ent, "Frames": len(grp),
            "PDR piétons": round(det_ped, 3) if not pd.isna(det_ped) else None,
            "PDR véhicules": round(det_veh, 3) if not pd.isna(det_veh) else None,
        })
    df = pd.DataFrame(rows).sort_values(["Model", "N", "Entities (GT)"])
    pdr_cols = ["PDR piétons", "PDR véhicules"]
    st.dataframe(
        _style_f1_cols(df, pdr_cols).format(
            {c: "{:.3f}" for c in pdr_cols}, na_rep="—"
        ),
        width="stretch",
        height=min(600, 80 + len(df) * 38),
    )


def _scatter_complexity(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str) -> None:
    d = df[[x, y, "model"]].dropna()
    if d.empty:
        st.info(f"No data for {title}.")
        return
    trendline = "ols" if d[x].nunique() >= 2 else None
    fig = px.scatter(d, x=x, y=y, color="model", trendline=trendline,
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


# ─────────────────────────────────────────────────────────────────────────────
# Compare Runs tab (inside extraction analysis)
# ─────────────────────────────────────────────────────────────────────────────

def view_compare_runs_tab(extraction_runs: list[Path]) -> None:
    """Diff table between two extraction runs, with Δ columns colour-coded."""
    if not extraction_runs:
        st.info("No extraction runs available.")
        return

    def _run_label(run_dir: Path) -> str:
        meta = load_run_meta(run_dir)
        tracking = "on" if meta.get("tracking") else "off"
        return f"{run_dir.name} | tracking={tracking}"

    run_options = {_run_label(r): r for r in extraction_runs}
    labels = list(run_options.keys())

    col_a, col_b = st.columns(2)
    with col_a:
        label_a = st.selectbox("Run A", labels, key="cmp_tab_run_a")
    with col_b:
        label_b = st.selectbox("Run B", labels, index=min(1, len(labels) - 1), key="cmp_tab_run_b")

    if label_a == label_b:
        st.info("Select two different runs to compare.")
        return

    scores_a = {(s["model_name"], s["window_size"]): s for s in load_scores(run_options[label_a])}
    scores_b = {(s["model_name"], s["window_size"]): s for s in load_scores(run_options[label_b])}

    common_keys = sorted(set(scores_a) & set(scores_b))
    if not common_keys:
        st.warning("No common (model, window_size) pairs found between the two runs.")
        return

    def _delta(va, vb):
        return round(vb - va, 4) if va is not None and vb is not None else None

    rows = []
    for model, ws in common_keys:
        sa, sb = scores_a[(model, ws)], scores_b[(model, ws)]
        rows.append({
            "model":       model,
            "window_size": ws,
            "F1_ped (A)":  sa.get("f1_pedestrians"),
            "F1_ped (B)":  sb.get("f1_pedestrians"),
            "ΔF1_ped":     _delta(sa.get("f1_pedestrians"), sb.get("f1_pedestrians")),
            "F1_veh (A)":  sa.get("f1_vehicles"),
            "F1_veh (B)":  sb.get("f1_vehicles"),
            "ΔF1_veh":     _delta(sa.get("f1_vehicles"), sb.get("f1_vehicles")),
            "F1_ctx (A)":  sa.get("f1_context"),
            "F1_ctx (B)":  sb.get("f1_context"),
            "ΔF1_ctx":     _delta(sa.get("f1_context"), sb.get("f1_context")),
        })

    df = pd.DataFrame(rows)
    delta_cols = ["ΔF1_ped", "ΔF1_veh", "ΔF1_ctx"]
    score_cols = ["F1_ped (A)", "F1_ped (B)", "F1_veh (A)", "F1_veh (B)", "F1_ctx (A)", "F1_ctx (B)"]

    def _style_row(row):
        styles = [""] * len(row)
        for col in delta_cols:
            if col not in df.columns:
                continue
            idx = df.columns.get_loc(col)
            val = row.get(col)
            if val is None or val == 0:
                styles[idx] = "color: #888"  # type: ignore[index]
            elif val > 0:
                styles[idx] = "color: #6cf5a8"  # type: ignore[index]
            else:
                styles[idx] = "color: #f57c6c"  # type: ignore[index]
        return styles

    fmt = {c: "{:.4f}" for c in delta_cols + score_cols if c in df.columns}
    st.dataframe(
        df.style.apply(_style_row, axis=1).format(fmt, na_rep="—"),  # type: ignore[arg-type]
        width="stretch",
        height=min(600, 80 + len(rows) * 38),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4 & 5. Multi-run comparison page
# ─────────────────────────────────────────────────────────────────────────────

def _load_multi_scores(run_dirs: list[Path]) -> pd.DataFrame:
    """Load scores.json from multiple runs, tag with run_id and tracking flag."""
    rows = []
    for rd in run_dirs:
        meta = load_run_meta(rd)
        tracking = bool(meta.get("tracking"))
        for s in load_scores(rd):
            rows.append({**s, "run_id": rd.name, "tracking": tracking})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _run_label_with_tracking(run_dir: Path) -> str:
    meta = load_run_meta(run_dir)
    mode = meta.get("mode", "extraction")
    if mode == "complexity":
        badge = "✓ detection" if meta.get("tracking") else "✗ no detection"
    else:
        badge = "✓ tracking" if meta.get("tracking") else "✗ no tracking"
    return f"{run_dir.name}  [{badge}]"


def _partition_by_tracking(runs: list[Path]) -> tuple[list[Path], list[Path]]:
    """Return (runs_with_tracking, runs_without_tracking)."""
    with_tr, without_tr = [], []
    for r in runs:
        if load_run_meta(r).get("tracking"):
            with_tr.append(r)
        else:
            without_tr.append(r)
    return with_tr, without_tr


def _render_general_comparison(sel_dirs: list[Path], mode: str) -> None:
    """Render the general multi-run comparison (extraction or complexity)."""
    if mode == "extraction":
        df = _load_multi_scores(sel_dirs)
        if df.empty:
            st.error("No scores.json found in selected runs.")
            return

        st.subheader("Scores Overview — All Runs")
        display_cols = ["run_id", "tracking", "model_name", "window_size", "n_frames",
                        "parse_success_rate", "f1_context", "f1_pedestrians", "f1_vehicles",
                        "avg_judge_overall", "avg_latency_s"]
        existing = [c for c in display_cols if c in df.columns]
        tbl = df[existing].copy()
        f1_cols = ["f1_context", "f1_pedestrians", "f1_vehicles", "avg_judge_overall"]
        fmt = {c: "{:.3f}" for c in f1_cols if c in tbl.columns}
        if "parse_success_rate" in tbl.columns:
            fmt["parse_success_rate"] = "{:.1%}"
        st.dataframe(
            _style_f1_cols(tbl, [c for c in f1_cols if c in tbl.columns]).format(
                fmt, na_rep="—",  # type: ignore[arg-type]
            ),
            width="stretch",
        )

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
        # ── PDR summary from scores.json ──────────────────────────────────────
        pdr_rows = []
        for rd in sel_dirs:
            meta = load_run_meta(rd)
            tracking = "✓" if meta.get("tracking") else "✗"
            for s in load_scores(rd):
                pdr_rows.append({
                    "Run":              rd.name,
                    "Tracking":         tracking,
                    "Model":            s.get("model_name", ""),
                    "N":                s.get("window_size"),
                    "Frames":           s.get("n_frames"),
                    "Parse %":          round(s["parse_success_rate"] * 100, 1) if s.get("parse_success_rate") is not None else None,
                    "PDR ped (mean)":   s.get("mean_pdr_ped"),
                    "PDR ped (std)":    s.get("std_pdr_ped"),
                    "PDR veh (mean)":   s.get("mean_pdr_veh"),
                    "PDR veh (std)":    s.get("std_pdr_veh"),
                    "Latency (s)":      round(s["avg_latency_s"], 2) if s.get("avg_latency_s") is not None else None,
                })
        if pdr_rows:
            st.subheader("PDR Scores — All Runs")
            pdr_df = pd.DataFrame(pdr_rows)
            pdr_score_cols = ["PDR ped (mean)", "PDR ped (std)", "PDR veh (mean)", "PDR veh (std)"]
            st.dataframe(
                _style_f1_cols(pdr_df, ["PDR ped (mean)", "PDR veh (mean)"]).format(
                    {c: "{:.3f}" for c in pdr_score_cols if c in pdr_df.columns},
                    na_rep="—",  # type: ignore[arg-type]
                ),
                width="stretch",
                height=min(500, 80 + len(pdr_rows) * 38),
            )

        all_frame_dfs = []
        for i, rd in enumerate(sel_dirs):
            frame_df = _load_complexity_data(rd)
            if not frame_df.empty:
                frame_df = frame_df.copy()
                frame_df["run_id"]    = rd.name          # run seul, sans modèle
                frame_df["run_label"] = rd.name + " · " + frame_df["model"].astype(str)
                frame_df["_color_idx"] = i
                all_frame_dfs.append(frame_df)

        if not all_frame_dfs:
            st.warning("No frame_scores data found in the selected runs.")
            return

        combined = pd.concat(all_frame_dfs, ignore_index=True)

        # ── Bucketing n_persons_gt ────────────────────────────────────────
        _BUCKET_DEFS = [
            ("1",     lambda n: n == 1),
            ("2",     lambda n: n == 2),
            ("3-4",   lambda n: 3 <= n <= 4),
            ("5-6",   lambda n: 5 <= n <= 6),
            ("7-10",  lambda n: 7 <= n <= 10),
            ("11-15", lambda n: 11 <= n <= 15),
            ("16-20", lambda n: 16 <= n <= 20),
            ("21-30", lambda n: 21 <= n <= 30),
            ("31-40", lambda n: 31 <= n <= 40),
            ("41+",   lambda n: n >= 41),
        ]
        _BUCKET_NAMES = [b[0] for b in _BUCKET_DEFS]
        _BUCKET_ORDER = {b: i for i, b in enumerate(_BUCKET_NAMES)}

        def _assign_bucket(n):
            for name, fn in _BUCKET_DEFS:
                if fn(int(n)): return name
            return "41+"

        if "n_persons_gt" in combined.columns:
            combined["bucket"] = combined["n_persons_gt"].dropna().apply(_assign_bucket)
            combined["bucket_order"] = combined["bucket"].map(_BUCKET_ORDER)

        # Colors per model (consistent across panels)
        _MODEL_PALETTE = {
            "gpt-4o-mini":   "#5b9cf6",
            "gpt-5-mini":    "#4ecb8d",
            "mistral-large": "#f5844c",
            "mistral-medium":"#b57bee",
        }
        _LAYOUT_CLEAN = dict(
            plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
            font=dict(color="#c8ccd8"),
            yaxis_gridcolor="#2a2d3d", yaxis_linecolor="#2a2d3d",
            xaxis_linecolor="#2a2d3d",
            legend=dict(bgcolor="#1a1d27", bordercolor="#2a2d3d", borderwidth=1),
            margin=dict(t=40, b=40),
        )

        # ── 1. Lignes : 2 panneaux côte à côte, 4 courbes par panneau ──────
        st.subheader("Pedestrian Detection Rate par densité de scène")
        run_ids = list(combined["run_id"].unique())
        c1, c2 = st.columns(2)
        for col, run_id in zip([c1, c2], run_ids[:2]):
            with col:
                d: pd.DataFrame = combined[combined["run_id"] == run_id].dropna(
                    subset=["bucket", "det_ped"])
                if d.empty:
                    st.info("No data.")
                    continue
                fig = go.Figure()
                for model, grp in d.groupby("model"):
                    bkt_stats = (
                        grp.groupby(["bucket", "bucket_order"])["det_ped"]
                        .mean().reset_index()
                        .sort_values("bucket_order")
                    )
                    color = _MODEL_PALETTE.get(model, "#888888")
                    fig.add_trace(go.Scatter(
                        x=bkt_stats["bucket"], y=bkt_stats["det_ped"],
                        mode="lines+markers", name=model,
                        line=dict(color=color, width=2),
                        marker=dict(color=color, size=6),
                    ))
                fig.update_layout(
                    **_LAYOUT_CLEAN,
                    title=run_id,
                    xaxis_title="Nb piétons GT",
                    yaxis_title="Pedestrian Detection Rate",
                    yaxis_range=[0, 1.05],
                    height=400,
                    xaxis=dict(
                        categoryorder="array",
                        categoryarray=_BUCKET_NAMES,
                        gridcolor="#2a2d3d", tickangle=30,
                        color="#c8ccd8",
                    ),
                )
                st.plotly_chart(fig, width="stretch")

        # ── 2. Barres par modèle : PDR vs densité, une barre par run ──────
        st.subheader("PDR par densité — détail par modèle")
        models_present = [m for m in _MODEL_PALETTE if m in combined["model"].unique()]
        cols2 = st.columns(2)
        for i, model in enumerate(models_present):
            with cols2[i % 2]:
                d = combined[combined["model"] == model].dropna(
                    subset=["bucket", "det_ped"])
                if d.empty:
                    continue
                fig = go.Figure()
                for cidx, (run_id, grp) in enumerate(d.groupby("run_id")):
                    bkt_stats = (
                        grp.groupby(["bucket", "bucket_order"])["det_ped"]
                        .mean().reset_index()
                        .sort_values("bucket_order")
                    )
                    color = PALETTE[cidx % len(PALETTE)]
                    fig.add_trace(go.Bar(
                        x=bkt_stats["bucket"], y=bkt_stats["det_ped"],
                        name=run_id,
                        marker_color=color, opacity=0.85,
                        text=bkt_stats["det_ped"].map("{:.2f}".format),
                        textposition="outside",
                        textfont=dict(size=9, color="#c8ccd8"),
                    ))
                fig.update_layout(
                    **_LAYOUT_CLEAN,
                    title=model,
                    xaxis_title="Nb piétons GT",
                    yaxis_title="PDR",
                    yaxis_range=[0, 1.2],
                    barmode="group",
                    height=360,
                    xaxis=dict(
                        categoryorder="array",
                        categoryarray=_BUCKET_NAMES,
                        gridcolor="#2a2d3d", tickangle=30,
                        color="#c8ccd8",
                    ),
                )
                st.plotly_chart(fig, width="stretch")


def _render_tracking_delta(run_no_tr: Path, run_with_tr: Path) -> None:
    """Render delta table + bar chart between a no-tracking and a tracking run."""
    scores_no  = {(s["model_name"], s["window_size"]): s for s in load_scores(run_no_tr)}
    scores_yes = {(s["model_name"], s["window_size"]): s for s in load_scores(run_with_tr)}

    common_keys = sorted(set(scores_no) & set(scores_yes))
    if not common_keys:
        st.warning("Aucune paire (model, window_size) commune entre les deux runs.")
        return

    def _delta(va, vb):
        return round(vb - va, 4) if va is not None and vb is not None else None

    rows = []
    for model, ws in common_keys:
        sn, sy = scores_no[(model, ws)], scores_yes[(model, ws)]
        rows.append({
            "Model":              model,
            "N":                  ws,
            "F1_ped (no tr.)":   sn.get("f1_pedestrians"),
            "F1_ped (tracking)": sy.get("f1_pedestrians"),
            "ΔF1_ped":           _delta(sn.get("f1_pedestrians"), sy.get("f1_pedestrians")),
            "F1_veh (no tr.)":   sn.get("f1_vehicles"),
            "F1_veh (tracking)": sy.get("f1_vehicles"),
            "ΔF1_veh":           _delta(sn.get("f1_vehicles"), sy.get("f1_vehicles")),
            "F1_ctx (no tr.)":   sn.get("f1_context"),
            "F1_ctx (tracking)": sy.get("f1_context"),
            "ΔF1_ctx":           _delta(sn.get("f1_context"), sy.get("f1_context")),
        })

    df = pd.DataFrame(rows)
    delta_cols = ["ΔF1_ped", "ΔF1_veh", "ΔF1_ctx"]
    score_cols = [c for c in df.columns if c not in ["Model", "N"] + delta_cols]

    def _style_delta(row):
        styles = [""] * len(row)
        for col in delta_cols:
            if col not in df.columns:
                continue
            idx = df.columns.get_loc(col)
            val = row.get(col)
            if val is None or val == 0:
                styles[idx] = "color: #888"  # type: ignore[index]
            elif val > 0:
                styles[idx] = "color: #6cf5a8"  # type: ignore[index]
            else:
                styles[idx] = "color: #f57c6c"  # type: ignore[index]
        return styles

    fmt = {c: "{:.4f}" for c in delta_cols + score_cols if c in df.columns}
    st.subheader("Table de delta (B − A)")
    st.dataframe(
        df.style.apply(_style_delta, axis=1).format(fmt, na_rep="—"),  # type: ignore[arg-type]
        width="stretch",
        height=min(600, 80 + len(rows) * 38),
    )

    # Bar chart side-by-side
    bar_rows = []
    for r in rows:
        cfg = f"{r['Model']} N={r['N']}"
        for metric, key_no, key_yes in [
            ("F1 Ped", "F1_ped (no tr.)", "F1_ped (tracking)"),
            ("F1 Veh", "F1_veh (no tr.)", "F1_veh (tracking)"),
            ("F1 Ctx", "F1_ctx (no tr.)", "F1_ctx (tracking)"),
        ]:
            bar_rows.append({"Config": cfg, "Metric": metric,
                              "Score": r.get(key_no),  "Mode": "Sans tracking"})
            bar_rows.append({"Config": cfg, "Metric": metric,
                              "Score": r.get(key_yes), "Mode": "Avec tracking"})
    bar_df = pd.DataFrame(bar_rows).dropna(subset=["Score"])
    if not bar_df.empty:
        st.subheader("Comparaison F1 — Avec vs Sans Tracking")
        fig = px.bar(
            bar_df, x="Config", y="Score", color="Mode",
            facet_col="Metric", barmode="group",
            range_y=[0, 1.05], template="plotly_dark", height=420,
            color_discrete_map={"Sans tracking": "#6c8ef5", "Avec tracking": "#6cf5a8"},
        )
        fig.update_layout(**_plotly_cfg())
        st.plotly_chart(fig, width="stretch")


def page_compare(all_runs: dict[str, list[Path]]) -> None:
    st.header("Compare Runs")

    mode = st.selectbox("Mode", list(all_runs.keys()), key="cmp_mode")
    runs = all_runs[mode]

    tab_general, tab_tracking = st.tabs(["Comparaison générale", "Avec vs Sans Tracking"])

    # ── Onglet 1 : Comparaison générale ──────────────────────────────────────
    with tab_general:
        filtered_runs = runs

        if not filtered_runs:
            st.info("Aucun run disponible pour ce filtre.")
        else:
            run_labels = [_run_label_with_tracking(r) for r in filtered_runs]
            sel_labels = st.multiselect(
                "Sélectionner les runs à comparer", run_labels,
                default=run_labels[:min(3, len(run_labels))],
                key="cmp_runs_general",
            )
            if len(sel_labels) < 2:
                st.info("Sélectionner au moins 2 runs.")
            else:
                sel_dirs = [filtered_runs[run_labels.index(l)] for l in sel_labels]
                _render_general_comparison(sel_dirs, mode)

    # ── Onglet 2 : Avec vs Sans Tracking ─────────────────────────────────────
    with tab_tracking:
        if mode != "extraction":
            st.info("La comparaison tracking n'est disponible qu'en mode extraction.")
        else:
            with_tr, without_tr = _partition_by_tracking(runs)

            if not with_tr:
                st.warning("Aucun run avec tracking trouvé dans ce mode.")
            elif not without_tr:
                st.warning("Aucun run sans tracking trouvé dans ce mode.")
            else:
                col_no, col_yes = st.columns(2)
                with col_no:
                    labels_no = [_run_label_with_tracking(r) for r in without_tr]
                    sel_no = st.selectbox(
                        "Run A — Sans tracking", labels_no, key="cmp_tr_no"
                    )
                    run_no_tr = without_tr[labels_no.index(sel_no)]
                with col_yes:
                    labels_yes = [_run_label_with_tracking(r) for r in with_tr]
                    sel_yes = st.selectbox(
                        "Run B — Avec tracking", labels_yes, key="cmp_tr_yes"
                    )
                    run_with_tr = with_tr[labels_yes.index(sel_yes)]

                st.divider()
                _render_tracking_delta(run_no_tr, run_with_tr)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    all_runs = list_runs()

    with st.sidebar:
        st.markdown("## Semora")
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
    st.title(" Semora — Benchmark Analysis")

    with st.sidebar:
        mode = st.selectbox("Mode", list(all_runs.keys()))
        runs = all_runs[mode]
        run_labels = [_run_label_with_tracking(r) for r in runs]
        selected_label = st.selectbox("Run", run_labels)
        run_dir = runs[run_labels.index(selected_label)]
        st.caption(f"`{run_dir}`")

    if mode == "extraction":
        summaries = load_scores(run_dir)
        if not summaries:
            st.error("No scores.json found in this run.")
            return

        meta = load_run_meta(run_dir)
        if meta.get("tracking"):
            st.info("Tracking actif sur ce run.")

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
                        "Temporal Consistency", "Latency", "Compare Runs"])
        with tabs[0]: view_scores_overview(filtered)
        with tabs[1]: view_field_detail(filtered)
        with tabs[2]: view_judge_scores(filtered)
        with tabs[3]: view_temporal_consistency(run_dir)
        with tabs[4]: view_latency(filtered)
        with tabs[5]: view_compare_runs_tab(all_runs.get("extraction", []))

    elif mode == "complexity":
        frame_df = _load_complexity_data(run_dir)
        if frame_df.empty:
            st.error("No frame_scores.jsonl found in this run.")
            return

        summaries = load_scores(run_dir)
        models = sorted(frame_df["model"].unique())
        with st.sidebar:
            st.divider()
            st.subheader("Filters")
            sel_models = st.multiselect("Models", models, default=models)

        frame_df = frame_df[frame_df["model"].isin(sel_models)]
        summaries = [s for s in summaries if s.get("model_name") in sel_models]

        # PDR metric cards
        if summaries:
            for s in summaries:
                st.markdown(f"**{s['model_name']}** · N={s['window_size']}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Frames",        s.get("n_frames", "—"))
                c2.metric("PDR ped",       f"{s['mean_pdr_ped']:.3f} ± {s['std_pdr_ped']:.3f}" if s.get("mean_pdr_ped") is not None else "—")
                c3.metric("PDR veh",       f"{s['mean_pdr_veh']:.3f} ± {s['std_pdr_veh']:.3f}" if s.get("mean_pdr_veh") is not None else "—")
                c4.metric("Latency (s)",   f"{s['avg_latency_s']:.2f}" if s.get("avg_latency_s") is not None else "—")
            st.divider()

        tabs = st.tabs(["PDR Table", "Detection Rate"])
        with tabs[0]:
            view_pdr_table(frame_df)
        with tabs[1]:
            view_detection_rate(frame_df)

    else:
        st.info(f"No view implemented for mode '{mode}'.")


if __name__ == "__main__":
    main()
