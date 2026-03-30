"""
analyze_temporal.py — Temporal consistency analysis via track_hint matching

Measures whether a VLM describes the same pedestrian consistently across the
frames of each batch, by checking that the `track_hint` it assigned to each
pedestrian in the center frame reappears (verbatim or near-verbatim) in every
other frame of the same window.

Metric per batch:
    consistency_rate = (# center-frame track_hints found in ALL other frames)
                       / (# track_hints in center frame)

    - Matching: exact string first, then Jaccard on tokens ≥ 0.6
    - Batches with no pedestrians in the center frame are skipped.
    - Batches with parse_success=False are skipped.

Aggregated by (model, N):
    mean_consistency    — average over all valid batches
    median_consistency
    pct_fully_stable    — % of batches where consistency_rate == 1.0

Input: parsed_outputs.jsonl produced by the pipeline (save_parsed_outputs: true).
Each line format:
    {"model": "...", "N": 4, "clip_id": "...", "center_frame": "...",
     "parse_success": true,
     "parsed": {"frames": [{"scene_context":{}, "pedestrians":[...], "vehicles":[...]}, ...]}}

Usage:
    python analyze_temporal.py \\
        --results results/runs/<run>/parsed_outputs.jsonl \\
        [--output-dir outputs/]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Track-hint matching
# ─────────────────────────────────────────────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _hint_found(hint: str, candidates: list[str], threshold: float = 0.6) -> bool:
    """Return True if `hint` matches any candidate (exact or Jaccard ≥ threshold)."""
    if hint in candidates:
        return True
    return any(_jaccard(hint, c) >= threshold for c in candidates)


# ─────────────────────────────────────────────────────────────────────────────
# Center frame index
# ─────────────────────────────────────────────────────────────────────────────

def _center_idx(window_size: int) -> int:
    """0-based index of the center (scored) frame in the frames list."""
    return window_size // 2 if window_size % 2 == 1 else window_size - 1


# ─────────────────────────────────────────────────────────────────────────────
# Consistency computation
# ─────────────────────────────────────────────────────────────────────────────

def _batch_consistency(frames_data: list[dict], window_size: int) -> float | None:
    """Compute consistency_rate for one batch.

    Returns None if the batch should be skipped (no pedestrians in center frame,
    or only one frame in the window).
    """
    if len(frames_data) < 2:
        return None

    ci = _center_idx(window_size)
    if ci >= len(frames_data):
        ci = len(frames_data) - 1

    center_hints: list[str] = [
        p.get("track_hint", "").strip()
        for p in frames_data[ci].get("pedestrians", [])
        if p.get("track_hint", "").strip()
    ]
    if not center_hints:
        return None

    # Other frame indices
    other_indices = [i for i in range(len(frames_data)) if i != ci]

    found_in_all = 0
    for hint in center_hints:
        present_in_every_frame = all(
            _hint_found(
                hint,
                [p.get("track_hint", "").strip() for p in frames_data[i].get("pedestrians", [])],
            )
            for i in other_indices
        )
        if present_in_every_frame:
            found_in_all += 1

    return found_in_all / len(center_hints)


# ─────────────────────────────────────────────────────────────────────────────
# JSONL loading & processing
# ─────────────────────────────────────────────────────────────────────────────

def process_results(path: Path) -> list[dict]:
    """Return per-batch consistency rows from parsed_outputs.jsonl."""
    rows: list[dict] = []
    n_skipped_parse = 0
    n_skipped_empty = 0

    with open(path) as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            rec = json.loads(raw_line)

            if not rec.get("parse_success", False):
                n_skipped_parse += 1
                continue

            parsed = rec.get("parsed", {})
            frames_data: list[dict] = parsed.get("frames", [])
            window_size: int = rec.get("N", len(frames_data))

            rate = _batch_consistency(frames_data, window_size)
            if rate is None:
                n_skipped_empty += 1
                continue

            rows.append({
                "model":            rec["model"],
                "window_size":      window_size,
                "clip_id":          rec.get("clip_id", ""),
                "center_frame":     rec.get("center_frame", ""),
                "consistency_rate": rate,
            })

    if n_skipped_parse:
        console.print(f"  [dim]→ {n_skipped_parse} batches skipped (parse_success=False)[/dim]")
    if n_skipped_empty:
        console.print(f"  [dim]→ {n_skipped_empty} batches skipped (no pedestrians in center frame or single frame)[/dim]")

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["window_size"])].append(row["consistency_rate"])

    results: list[dict] = []
    for (model, ws), values in sorted(groups.items()):
        results.append({
            "model_name":         model,
            "window_size":        ws,
            "n_batches":          len(values),
            "mean_consistency":   statistics.mean(values),
            "median_consistency": statistics.median(values),
            "pct_fully_stable":   sum(1 for v in values if v == 1.0) / len(values) * 100,
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


def print_table(agg: list[dict]) -> None:
    # Group by model for one table per model
    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in agg:
        by_model[row["model_name"]].append(row)

    for model, rows in sorted(by_model.items()):
        table = Table(title=f"Temporal Consistency — {model}", show_header=True, header_style="bold")
        table.add_column("Window size",        justify="center")
        table.add_column("Batches",            justify="right")
        table.add_column("Mean consistency",   justify="right")
        table.add_column("Median consistency", justify="right")
        table.add_column("% Fully stable",     justify="right")

        for row in rows:
            mean = row["mean_consistency"]
            color = "green" if mean >= 0.8 else ("yellow" if mean >= 0.5 else "red")
            table.add_row(
                str(row["window_size"]),
                str(row["n_batches"]),
                f"[{color}]{mean:.3f}[/{color}]",
                f"{row['median_consistency']:.3f}",
                f"{row['pct_fully_stable']:.1f}%",
            )
        console.print(table)
        console.print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Temporal consistency analysis — track_hint stability across frames",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results", required=True, type=Path,
        help="parsed_outputs.jsonl from a Semora run (save_parsed_outputs: true)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs"),
        help="Directory where the CSV will be written",
    )
    args = parser.parse_args()

    if not args.results.exists():
        console.print(f"[bold red]File not found:[/bold red] {args.results}")
        raise SystemExit(1)

    # ── Process ──────────────────────────────────────────────────────────────
    console.print(f"[bold]Loading:[/bold] {args.results}")
    rows = process_results(args.results)
    console.print(f"  → {len(rows):,} valid batches analyzed")

    if not rows:
        console.print("[yellow]No valid batches found. Make sure the run used prompt v2 "
                      "(multi-frame output) and window_size > 1.[/yellow]")
        return

    # ── Aggregate ─────────────────────────────────────────────────────────────
    agg = aggregate(rows)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"temporal_consistency_{args.results.parent.name}.csv"
    save_csv(agg, csv_path)
    console.print(f"\n[bold green]CSV saved:[/bold green] {csv_path}\n")

    # ── Print tables ──────────────────────────────────────────────────────────
    print_table(agg)


if __name__ == "__main__":
    main()
