"""
analyze_temporal.py — Temporal consistency analysis (PT-02)

For each (model, clip, track_id, field), reconstructs the ordered sequence
of frame-level F1 scores on frames where that pedestrian appears, then
computes an instability rate:

    instability = nb_f1_changes / (nb_frames - 1)

A "change" occurs when the field's F1 score differs between two consecutive
frames where the same track_id is present.

Usage:
    python analyze_temporal.py \\
        --results  results/runs/20240101_120000/raw_outputs.jsonl \\
        --clips-cfg configs/clips.yaml \\
        [--output-dir outputs/]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

from src.sampling.clip_loader import load_all_clips, TITANClip  # noqa: E402

console = Console()

PERSON_FIELDS = ["atomic_action", "simple_context", "communicative", "transporting", "age"]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_results(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def index_results(records: list[dict]) -> dict[tuple[str, int, str, str], dict]:
    """Index by (model_name, window_size, clip_id, center_frame).

    If window_size is absent from the record (older format), defaults to 0.
    """
    idx: dict[tuple[str, int, str, str], dict] = {}
    for r in records:
        key = (
            r["model_name"],
            r.get("window_size", 0),
            r["clip_id"],
            r["center_frame"],
        )
        idx[key] = r
    return idx


# ─────────────────────────────────────────────────────────────────────────────
# GT track sequences
# ─────────────────────────────────────────────────────────────────────────────

def build_track_sequences(clips: list[TITANClip]) -> dict[str, dict[str, list[str]]]:
    """Return {clip_id: {track_id: [frame_name, ...]}} in chronological order."""
    result: dict[str, dict[str, list[str]]] = {}
    for clip in clips:
        track_frames: dict[str, list[str]] = defaultdict(list)
        for frame_name in clip.frame_names:          # already sorted chronologically
            ann = clip.annotations.get(frame_name)
            if ann is None:
                continue
            for person in ann.persons:
                track_id = str(person.get("obj_track_id", "")).strip()
                if track_id:
                    track_frames[track_id].append(frame_name)
        result[clip.clip_id] = dict(track_frames)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# F1 extraction from a FrameScore record
# ─────────────────────────────────────────────────────────────────────────────

def _f1_from_record(record: dict, field: str) -> float | None:
    """Extract F1 for a person field from a FrameScore dict. Returns None if absent."""
    fs = record.get("person_scores", {}).get(field)
    if fs is None:
        return None
    tp = fs.get("tp", 0)
    fp = fs.get("fp", 0)
    fn = fs.get("fn", 0)
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Instability computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_instability(
    records_idx: dict[tuple[str, int, str, str], dict],
    track_sequences: dict[str, dict[str, list[str]]],
    models: list[str],
    window_sizes: list[int],
) -> list[dict]:
    """Return flat list of per-(model, window_size, clip, track_id, field) instability."""
    rows: list[dict] = []

    for model in models:
        for ws in window_sizes:
            for clip_id, track_frames in track_sequences.items():
                for track_id, frame_list in track_frames.items():
                    # Collect (frame, record) pairs with parse_success=True
                    valid: list[tuple[str, dict]] = []
                    for frame_name in frame_list:
                        rec = records_idx.get((model, ws, clip_id, frame_name))
                        if rec is not None and rec.get("parse_success", False):
                            valid.append((frame_name, rec))

                    if len(valid) < 2:
                        continue  # need ≥ 2 frames to compute instability

                    for field in PERSON_FIELDS:
                        f1_seq = [_f1_from_record(rec, field) for _, rec in valid]
                        f1_seq = [v for v in f1_seq if v is not None]
                        if len(f1_seq) < 2:
                            continue

                        n_changes = sum(
                            1 for a, b in zip(f1_seq, f1_seq[1:])
                            if round(a, 4) != round(b, 4)
                        )
                        rows.append({
                            "model_name":   model,
                            "window_size":  ws,
                            "clip_id":      clip_id,
                            "track_id":     track_id,
                            "field":        field,
                            "instability":  n_changes / (len(f1_seq) - 1),
                            "n_frames":     len(f1_seq),
                        })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["model_name"], row["window_size"], row["field"])
        groups[key].append(row["instability"])

    results: list[dict] = []
    for (model, ws, field), values in sorted(groups.items()):
        results.append({
            "model_name":        model,
            "window_size":       ws,
            "field":             field,
            "n_tracks":          len(values),
            "mean_instability":  statistics.mean(values),
            "median_instability": statistics.median(values),
            "pct_unstable":      sum(1 for v in values if v > 0) / len(values) * 100,
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(agg: list[dict], path: Path) -> None:
    if not agg:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        writer.writeheader()
        writer.writerows(agg)


def print_tables(agg: list[dict]) -> None:
    # Group by (model, window_size)
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in agg:
        groups[(row["model_name"], row["window_size"])].append(row)

    for (model, ws), rows in sorted(groups.items()):
        table = Table(title=f"{model}  |  window_size={ws}", show_header=True, header_style="bold")
        table.add_column("Field",               style="cyan")
        table.add_column("Tracks",              justify="right")
        table.add_column("Mean instability",    justify="right")
        table.add_column("Median instability",  justify="right")
        table.add_column("% Unstable",          justify="right")

        for row in rows:
            table.add_row(
                row["field"],
                str(row["n_tracks"]),
                f"{row['mean_instability']:.3f}",
                f"{row['median_instability']:.3f}",
                f"{row['pct_unstable']:.1f}%",
            )
        console.print(table)
        console.print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Temporal consistency analysis (PT-02)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results",    required=True, type=Path,
                        help="FrameScore JSONL file (e.g. results/runs/<run>/raw_outputs.jsonl)")
    parser.add_argument("--clips-cfg",  required=True, type=Path,
                        help="Path to clips.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"),
                        help="Directory where the CSV will be saved")
    args = parser.parse_args()

    # ── Load JSONL ────────────────────────────────────────────────────────────
    console.print(f"[bold]Loading results:[/bold] {args.results}")
    records = load_results(args.results)
    console.print(f"  → {len(records):,} records")

    models      = sorted({r["model_name"]         for r in records})
    window_sizes = sorted({r.get("window_size", 0) for r in records})
    console.print(f"  → Models: {', '.join(models)}")
    console.print(f"  → Window sizes: {window_sizes}")

    records_idx = index_results(records)

    # ── Load GT clips ─────────────────────────────────────────────────────────
    console.print(f"\n[bold]Loading GT clips:[/bold] {args.clips_cfg}")
    with open(args.clips_cfg) as f:
        clips_cfg = yaml.safe_load(f)
    clips = load_all_clips(clips_cfg)
    console.print(f"  → {len(clips)} clip(s) loaded")

    # ── Build track sequences ─────────────────────────────────────────────────
    track_sequences = build_track_sequences(clips)
    n_tracks = sum(len(v) for v in track_sequences.values())
    console.print(f"  → {n_tracks} unique track(s) across all clips")

    # ── Compute instability ───────────────────────────────────────────────────
    console.print("\n[bold]Computing temporal instability…[/bold]")
    rows = compute_instability(records_idx, track_sequences, models, window_sizes)
    console.print(f"  → {len(rows):,} (track × field) sequences analyzed")

    if not rows:
        console.print("[yellow]No sequences found. Check that center_frame names in the "
                      "JSONL match the frame filenames in the clips.[/yellow]")
        return

    # ── Aggregate ─────────────────────────────────────────────────────────────
    agg = aggregate(rows)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"temporal_consistency_{args.results.stem}.csv"
    save_csv(agg, csv_path)
    console.print(f"\n[bold green]CSV saved:[/bold green] {csv_path}\n")

    # ── Print tables ──────────────────────────────────────────────────────────
    print_tables(agg)


if __name__ == "__main__":
    main()
