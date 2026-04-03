"""Analysis module — post-run scoring and temporal consistency for Semora."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box

console = Console(record=True)


def save_report(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analyze_report.html").write_text(console.export_html())
    (output_dir / "analyze_report.txt").write_text(console.export_text())

# Fields flagged as unreliable — shown with a warning in field detail
_UNRELIABLE_FIELDS = {
    "age": "VLMs always predict 'adult' (92% of GT) — cannot distinguish age from dashcam distance",
}


# ─────────────────────────────────────────────────────────────────────────────
# General analysis (scores.json)
# ─────────────────────────────────────────────────────────────────────────────

def print_scores_table(run_dir: Path) -> None:
    scores_path = run_dir / "raw" / "scores.json"
    if not scores_path.exists():
        console.print(f"[red]scores.json not found in {run_dir / 'raw'}[/red]")
        return

    summaries = json.loads(scores_path.read_text())

    table = Table(
        title=f"[bold]Extraction Quality Overview — {run_dir.name}[/bold]",
        box=box.HEAVY_EDGE,
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Model",         justify="left",  style="magenta")
    table.add_column("N",             justify="center")
    table.add_column("Parse %",       justify="right")
    table.add_column("F1 Context",    justify="right")
    table.add_column("F1 Ped",        justify="right")
    table.add_column("F1 Veh",        justify="right")
    table.add_column("Completeness",  justify="right", style="yellow")
    table.add_column("Sem. Rich.",    justify="right", style="yellow")
    table.add_column("Spatial",       justify="right", style="yellow")
    table.add_column("Judge Overall", justify="right", style="bold yellow")
    table.add_column("Latency (s)",   justify="right")

    for s in summaries:
        parse_rate = s.get("parse_success_rate")
        parse_color = "green" if parse_rate and parse_rate >= 0.8 else ("yellow" if parse_rate and parse_rate >= 0.5 else "red")

        f1_ctx   = s.get("f1_context")
        f1_ped   = s.get("f1_pedestrians")
        f1_veh   = s.get("f1_vehicles")
        comp     = s.get("avg_judge_completeness")
        richness = s.get("avg_judge_semantic_richness")
        spatial  = s.get("avg_judge_spatial_relations")
        overall  = s.get("avg_judge_overall")

        table.add_row(
            s["model_name"],
            str(s["window_size"]),
            f"[{parse_color}]{parse_rate:.0%}[/{parse_color}]" if parse_rate is not None else "-",
            f"{f1_ctx:.3f}"   if f1_ctx   is not None else "-",
            f"{f1_ped:.3f}"   if f1_ped   is not None else "-",
            f"{f1_veh:.3f}"   if f1_veh   is not None else "-",
            f"{comp:.3f}"     if comp     is not None else "-",
            f"{richness:.3f}" if richness is not None else "-",
            f"{spatial:.3f}"  if spatial  is not None else "-",
            f"{overall:.3f}"  if overall  is not None else "-",
            f"{s['avg_latency_s']:.2f}",
        )

    console.print(table)

    # Per-field detail
    for s in summaries:
        _print_field_detail(s)


def _print_field_detail(s: dict) -> None:
    label = f"{s['model_name']} · N={s['window_size']}"

    table = Table(title=f"Per-field Scores — {label}", box=box.SIMPLE, header_style="bold")
    table.add_column("Field",     justify="left")
    table.add_column("Category",  justify="center", style="dim")
    table.add_column("Precision", justify="right")
    table.add_column("Recall",    justify="right")
    table.add_column("F1",        justify="right")

    for field, vals in s.get("person_fields", {}).items():
        f1 = vals.get("f1", 0)
        color = "green" if f1 >= 0.7 else ("yellow" if f1 >= 0.4 else "red")
        label = f"{field} ⚠" if field in _UNRELIABLE_FIELDS else field
        table.add_row(
            label, "ped",
            f"{vals['precision']:.3f}",
            f"{vals['recall']:.3f}",
            f"[{color}]{f1:.3f}[/{color}]",
        )

    for field, vals in s.get("vehicle_fields", {}).items():
        f1 = vals.get("f1", 0)
        color = "green" if f1 >= 0.7 else ("yellow" if f1 >= 0.4 else "red")
        table.add_row(
            field, "veh",
            f"{vals['precision']:.3f}",
            f"{vals['recall']:.3f}",
            f"[{color}]{f1:.3f}[/{color}]",
        )

    console.print()
    console.print(table)

    for field, reason in _UNRELIABLE_FIELDS.items():
        if field in s.get("person_fields", {}):
            console.print(f"  [yellow]⚠  {field}:[/yellow] [dim]{reason}[/dim]")
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Temporal consistency — inter-window (parsed_outputs.jsonl)
# ─────────────────────────────────────────────────────────────────────────────

_PERSON_FIELDS   = ["atomic_action", "simple_context", "communicative", "transporting", "age"]
_STATIC_FIELDS   = {"age"}          # expected identical across windows (person doesn't change age)
_MATCH_THRESHOLD = 0.45             # Jaccard similarity to consider two track_hints the same person


def _jaccard(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _center_peds(frames: list[dict], window_size: int) -> list[dict]:
    """Pedestrians from the center frame of a parsed window."""
    if not frames:
        return []
    ci = window_size // 2 if window_size % 2 == 1 else window_size - 1
    ci = min(ci, len(frames) - 1)
    return frames[ci].get("pedestrians", [])


def _match_pedestrians(
    peds_a: list[dict],
    peds_b: list[dict],
    threshold: float = _MATCH_THRESHOLD,
) -> list[tuple[dict, dict]]:
    """
    Greedy best-first matching of pedestrians across two windows by track_hint similarity.
    Returns matched pairs (ped_from_a, ped_from_b).
    """
    matches: list[tuple[dict, dict]] = []
    used_b: set[int] = set()

    for pa in peds_a:
        hint_a = pa.get("track_hint", "").strip()
        if not hint_a:
            continue
        best_sim, best_j = 0.0, -1
        for j, pb in enumerate(peds_b):
            if j in used_b:
                continue
            sim = _jaccard(hint_a, pb.get("track_hint", "").strip())
            if sim > best_sim:
                best_sim, best_j = sim, j
        if best_sim >= threshold and best_j >= 0:
            matches.append((pa, peds_b[best_j]))
            used_b.add(best_j)

    return matches


def run_temporal_analysis(run_dir: Path, output_dir: Path) -> None:
    """
    Inter-window temporal consistency.

    Each window is an independent inference call separated in time (90+ frames apart).
    For each (model, clip, N) we compare ALL pairs of windows: pedestrians are matched
    by track_hint similarity, then each attribute is compared across windows.

    Metrics:
      Re-ID rate  — % of pedestrians from one window matched in another (≥ threshold)
      agree %     — for matched pairs, % of times the same attribute value was predicted
    """
    raw_dir = run_dir / "raw"
    parsed_files = sorted(raw_dir.glob("*_parsed_outputs.jsonl"))
    if not parsed_files:
        console.print(f"[red]No parsed_outputs.jsonl files found in {raw_dir}[/red]")
        return

    console.print(f"\n[bold]Inter-window Temporal Consistency — {run_dir.name}[/bold]")
    console.print("[dim]Compares attribute stability for the same person across independent inferences[/dim]\n")

    # ── Load & group by (model, clip, N) ─────────────────────────────────────
    group_windows: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    n_skipped = 0

    for parsed_path in parsed_files:
        with open(parsed_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if not rec.get("parse_success", False):
                    n_skipped += 1
                    continue
                frames = rec["parsed"].get("frames", [])
                N      = rec["N"]
                peds   = _center_peds(frames, N)
                if not peds:
                    continue
                group_windows[(rec["model"], rec["clip_id"], N)].append({
                    "center_frame": rec["center_frame"],
                    "pedestrians":  peds,
                })

    if n_skipped:
        console.print(f"  [dim]→ {n_skipped} windows skipped (parse failed)[/dim]")

    if not group_windows:
        console.print("[yellow]No valid windows found.[/yellow]")
        return

    # ── Diagnostic: windows per group ─────────────────────────────────────────
    n_total   = sum(len(v) for v in group_windows.values())
    n_singles = sum(1 for v in group_windows.values() if len(v) < 2)
    n_groups  = len(group_windows)
    console.print(f"  [dim]→ {n_total} windows across {n_groups} (model, clip, N) groups — "
                  f"{n_singles}/{n_groups} groups have only 1 window (no pair possible)[/dim]")

    if n_singles == n_groups:
        console.print(
            "[yellow]⚠  No group has ≥ 2 windows — temporal comparison requires multiple "
            "inference windows per clip.\n"
            "   This usually means the clips are short or step is too large.[/yellow]"
        )
        return

    # ── Compute inter-window agreement for each (model, N) ───────────────────
    agg: dict[tuple[str, int], dict] = defaultdict(lambda: {
        "n_pairs":      0,
        "n_matches":    0,
        "n_candidates": 0,
        "field_agrees": defaultdict(list),
    })

    for (model, clip_id, N), windows in group_windows.items():
        windows = sorted(windows, key=lambda w: w["center_frame"])
        key = (model, N)

        for i in range(len(windows)):
            for j in range(i + 1, len(windows)):
                peds_i = windows[i]["pedestrians"]
                peds_j = windows[j]["pedestrians"]
                if not peds_i or not peds_j:
                    continue

                matches = _match_pedestrians(peds_i, peds_j)
                agg[key]["n_pairs"]      += 1
                agg[key]["n_matches"]    += len(matches)
                agg[key]["n_candidates"] += len(peds_i)

                for pa, pb in matches:
                    for field in _PERSON_FIELDS:
                        va, vb = pa.get(field), pb.get(field)
                        if va and vb:
                            agg[key]["field_agrees"][field].append(va == vb)

    # ── Display ───────────────────────────────────────────────────────────────
    by_model: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for (model, N), data in sorted(agg.items()):
        by_model[model].append((N, data))

    all_rows: list[dict] = []

    for model, entries in sorted(by_model.items()):
        col_headers = []
        for f in _PERSON_FIELDS:
            marker = " ⚠" if f in _UNRELIABLE_FIELDS else (" *" if f in _STATIC_FIELDS else "")
            col_headers.append(f"{f.replace('_', ' ')}{marker}")

        table = Table(
            title=f"[bold magenta]{model}[/bold magenta] — Temporal Re-ID & Attribute Consistency",
            box=box.SIMPLE,
            header_style="bold cyan",
            show_lines=True,
        )
        table.add_column("N",          justify="center")
        table.add_column("Re-ID rate", justify="right")
        for h in col_headers:
            table.add_column(h, justify="right")

        for N, data in sorted(entries):
            if data["n_pairs"] == 0:
                continue

            reid = data["n_matches"] / data["n_candidates"] if data["n_candidates"] else 0.0
            reid_color = "green" if reid >= 0.6 else ("yellow" if reid >= 0.35 else "red")

            field_cells = []
            for field in _PERSON_FIELDS:
                agrees = data["field_agrees"].get(field, [])
                if not agrees:
                    field_cells.append("[dim]-[/dim]")
                    continue
                rate = sum(agrees) / len(agrees)
                if field in _STATIC_FIELDS:
                    color = "green" if rate >= 0.85 else ("yellow" if rate >= 0.6 else "red")
                else:
                    color = "green" if rate >= 0.6 else ("yellow" if rate >= 0.35 else "dim")
                field_cells.append(f"[{color}]{rate:.0%}[/{color}]")

            table.add_row(
                str(N),
                f"[{reid_color}]{reid:.0%}[/{reid_color}] ({data['n_matches']}/{data['n_candidates']})",
                *field_cells,
            )

            row: dict = {
                "model_name":   model,
                "window_size":  N,
                "n_pairs":      data["n_pairs"],
                "reid_rate":    round(reid, 4),
                "n_matches":    data["n_matches"],
                "n_candidates": data["n_candidates"],
            }
            for field in _PERSON_FIELDS:
                agrees = data["field_agrees"].get(field, [])
                row[f"agree_{field}"] = round(sum(agrees) / len(agrees), 4) if agrees else None
            all_rows.append(row)

        console.print()
        console.print(table)

    console.print(
        f"  [dim]Re-ID rate: % of pedestrians from one window re-identified in another "
        f"(track_hint Jaccard ≥ {_MATCH_THRESHOLD:.0%}).[/dim]"
    )
    console.print(
        "  [dim]* age (static): should always agree across windows — red = model is inconsistent.[/dim]"
    )
    console.print(
        "  [dim]Dynamic fields: lower agreement is normal — the person may have changed action.[/dim]\n"
    )

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if all_rows:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"temporal_consistency_{run_dir.name}.csv"  # run_dir.name == run_id
        fieldnames = list(all_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        console.print(f"[bold green]CSV saved:[/bold green] {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Field guide
# ─────────────────────────────────────────────────────────────────────────────

_FIELD_GUIDE = """\
╔══════════════════════════════════════════════════════════════════╗
║                    SEMORA — FIELD GUIDE                          ║
╚══════════════════════════════════════════════════════════════════╝

── METRICS ─────────────────────────────────────────────────────────

  Parse %       % of inference windows where the VLM output was
                successfully parsed into structured JSON.

  F1 Context    F1 score on the simple_context field only (scene-level
                situation of a pedestrian, e.g. "crossing a street").

  F1 Ped        Average F1 across all 5 pedestrian fields.

  F1 Veh        Average F1 across all 3 vehicle fields.

  Precision     Of all values predicted by the model, how many are correct.
  Recall        Of all values present in the ground truth, how many were found.
  F1            Harmonic mean of Precision and Recall. Main ranking metric.

── GT SCORER — PEDESTRIAN FIELDS ───────────────────────────────────

  atomic_action   Instantaneous physical action of the pedestrian.
                  Values: walking, standing, running, sitting, ...

  simple_context  Displacement situation within the scene.
                  Values: crossing a street at pedestrian crossing,
                          waiting to cross street,
                          walking along the side of the road, ...

  communicative   Observable communication behaviour.
                  Values: none of the above, talking on phone,
                          looking into phone, talking in group, ...

  transporting    Whether the person is carrying an object.
                  Values: none of the above, carrying with both hands,
                          pushing, pulling, ...

  age ⚠           Visually estimated age group.
                  Values: child, adult, senior over 65.
                  WARNING: VLMs almost always predict 'adult' (matches
                  92% of GT by distribution) — this field is not
                  discriminative and should be interpreted with caution.

── GT SCORER — VEHICLE FIELDS ──────────────────────────────────────

  motion_status   Movement state of the vehicle.
                  Values: moving, stopped, parked.

  trunk_open      Whether the trunk is open.
                  Values: open, closed.

  doors_open      Whether any door is open.
                  Values: open, closed.

── LLM JUDGE CRITERIA (scored 0–1 by gpt-5-mini at temperature 0) ─

  Completeness    Were all entities and attributes in the scene detected?
                  Low score = missed pedestrians or vehicles.

  Sem. Richness   Are descriptions precise, detailed, and relevant?
                  Measures the density of useful information.

  Spatial         Are spatial relations between entities correctly described?
                  e.g. "pedestrian in front of vehicle" vs "on the sidewalk".

  Judge Overall   Global quality score synthesising the 3 criteria above.

── TEMPORAL ANALYSIS ───────────────────────────────────────────────

  Re-ID rate      % of pedestrians from one window re-identified in another
                  window of the same clip, matched by track_hint similarity
                  (Jaccard ≥ 45%). High = model uses stable descriptors.

  agree_%         For matched pedestrian pairs across windows, % of times
                  the same attribute value was predicted.
                  - Static fields (age): should always agree — inconsistency
                    means the model is contradicting itself.
                  - Dynamic fields (atomic_action, etc.): lower agreement
                    is expected — the person may have genuinely changed state.

  N               Window size (number of frames sent per inference call).
"""


def save_field_guide(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "field_guide.txt").write_text(_FIELD_GUIDE)
    _c = Console(record=True)
    (output_dir / "field_guide.html").write_text(_c.export_html())
