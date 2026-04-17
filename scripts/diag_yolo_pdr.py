"""Diagnostic : PDR YOLO vs PDR VLM.

Objectif : isoler si la limite de PDR en Plan 3 vient de la détection YOLO
(H1) ou du prompt/format bbox (H2).

Méthode :
    Pour chaque frame d'un run complexity, on recharge l'image et on fait
    tourner YOLO seul (detect_frame) pour compter indépendamment les entités
    détectées. On compare ensuite :
        PDR_yolo = min(n_yolo_ped / n_persons_gt, 1.0)
        PDR_vlm  = min(n_persons_pred / n_persons_gt, 1.0)   ← déjà dans frame_scores

    Trois cas possibles :
        PDR_yolo >> PDR_vlm  →  YOLO détecte mais VLM ne suit pas  →  H2 (format)
        PDR_yolo ≈  PDR_vlm  →  VLM suit YOLO de près              →  YOLO est le plafond (H1)
        PDR_yolo <  PDR_vlm  →  VLM dépasse YOLO (rare)            →  YOLO rate des choses
                                                                        que le VLM trouve seul

Usage :
    python scripts/diag_yolo_pdr.py --run runs/complexity/20260415_144802
    python scripts/diag_yolo_pdr.py --run runs/complexity/20260415_144802 --out diag_out.jsonl
    python scripts/diag_yolo_pdr.py --run runs/complexity/20260415_144802 --model gpt-4o-mini

Sortie :
    Un fichier JSONL (par défaut <run>/raw/diag_yolo_pdr.jsonl) avec un
    enregistrement par frame :
        clip_id, center_frame, model_name,
        n_persons_gt, n_vehicles_gt,
        n_yolo_ped, n_yolo_veh,          ← nouvelles colonnes
        n_persons_pred, n_vehicles_pred,  ← depuis frame_scores existant
        pdr_yolo_ped, pdr_vlm_ped,        ← calculés ici
        pdr_yolo_veh, pdr_vlm_veh,
        delta_ped, delta_veh              ← PDR_vlm - PDR_yolo  (>0 : VLM mieux, <0 : YOLO mieux)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from PIL import Image
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich import box

from src.tracking.detector import LiveTracker

console = Console()

_TITAN_CLIPS = _ROOT / "data" / "titan" / "clips"
_BUCKET_DEFS = [
    ("1",     lambda n: n == 1),
    ("2",     lambda n: n == 2),
    ("3-4",   lambda n: 3 <= n <= 4),
    ("5-6",   lambda n: 5 <= n <= 6),
    ("7-10",  lambda n: 7 <= n <= 10),
    ("11-15", lambda n: 11 <= n <= 15),
    ("16+",   lambda n: n >= 16),
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_frame(clip_id: str, frame_name: str) -> Image.Image | None:
    path = _TITAN_CLIPS / clip_id / "images" / frame_name
    if not path.exists():
        return None
    return Image.open(path).convert("RGB")


def _pdr(pred: int | None, gt: int) -> float | None:
    if gt == 0 or pred is None:
        return None
    return min(pred / gt, 1.0)


def _bucket(n: int) -> str:
    for name, fn in _BUCKET_DEFS:
        if fn(n):
            return name
    return "16+"


def _load_frame_scores(run_dir: Path, model_filter: str | None) -> list[dict]:
    """Read all *_frame_scores.jsonl from <run>/raw/, optionally filtered by model."""
    records: list[dict] = []
    for path in sorted((run_dir / "raw").glob("*_frame_scores.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if model_filter and rec.get("model_name") != model_filter:
                continue
            records.append(rec)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Core logic
# ─────────────────────────────────────────────────────────────────────────────

def run_diagnostic(
    run_dir: Path,
    out_path: Path,
    model_filter: str | None = None,
) -> list[dict]:
    records = _load_frame_scores(run_dir, model_filter)
    if not records:
        console.print(f"[red]No frame_scores.jsonl found in {run_dir}/raw/[/red]")
        sys.exit(1)

    # Déduplique les frames (plusieurs modèles ont la même frame — YOLO n'a besoin
    # de tourner qu'une fois par frame).
    unique_frames: dict[tuple[str, str], None] = {}
    for rec in records:
        unique_frames[(rec["clip_id"], rec["center_frame"])] = None

    console.print(
        f"[cyan]Run:[/cyan] {run_dir.name}  "
        f"[cyan]Records:[/cyan] {len(records)}  "
        f"[cyan]Unique frames:[/cyan] {len(unique_frames)}"
    )

    tracker = LiveTracker()

    # Cache YOLO counts per (clip_id, frame_name) — évite de relancer YOLO deux
    # fois pour le même frame si plusieurs modèles sont dans le run.
    yolo_cache: dict[tuple[str, str], tuple[int, int]] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running YOLO on unique frames...", total=len(unique_frames))
        for clip_id, frame_name in unique_frames:
            img = _load_frame(clip_id, frame_name)
            if img is None:
                console.print(f"[yellow]  Missing frame: {clip_id}/{frame_name}[/yellow]")
                yolo_cache[(clip_id, frame_name)] = (0, 0)
                progress.advance(task)
                continue
            dets = tracker.detect_frame(img)
            n_ped = sum(1 for d in dets if d["class_name"] == "person")
            n_veh = len(dets) - n_ped
            yolo_cache[(clip_id, frame_name)] = (n_ped, n_veh)
            progress.advance(task)

    # Assemble output records
    output: list[dict] = []
    for rec in records:
        key = (rec["clip_id"], rec["center_frame"])
        n_yolo_ped, n_yolo_veh = yolo_cache.get(key, (0, 0))
        n_pg = rec.get("n_persons_gt", 0)
        n_vg = rec.get("n_vehicles_gt", 0)
        n_pp = rec.get("n_persons_pred")
        n_vp = rec.get("n_vehicles_pred")

        pdr_yolo_ped = _pdr(n_yolo_ped, n_pg)
        pdr_vlm_ped  = _pdr(n_pp, n_pg)
        pdr_yolo_veh = _pdr(n_yolo_veh, n_vg)
        pdr_vlm_veh  = _pdr(n_vp, n_vg)

        output.append({
            "clip_id":        rec["clip_id"],
            "center_frame":   rec["center_frame"],
            "model_name":     rec["model_name"],
            "window_size":    rec["window_size"],
            # ground truth
            "n_persons_gt":   n_pg,
            "n_vehicles_gt":  n_vg,
            # detections
            "n_yolo_ped":     n_yolo_ped,
            "n_yolo_veh":     n_yolo_veh,
            "n_persons_pred": n_pp,
            "n_vehicles_pred":n_vp,
            # PDR
            "pdr_yolo_ped":   round(pdr_yolo_ped, 4) if pdr_yolo_ped is not None else None,
            "pdr_vlm_ped":    round(pdr_vlm_ped, 4)  if pdr_vlm_ped  is not None else None,
            "pdr_yolo_veh":   round(pdr_yolo_veh, 4) if pdr_yolo_veh is not None else None,
            "pdr_vlm_veh":    round(pdr_vlm_veh, 4)  if pdr_vlm_veh  is not None else None,
            # delta = PDR_vlm - PDR_yolo  (positif : VLM dépasse YOLO)
            "delta_ped": round(pdr_vlm_ped - pdr_yolo_ped, 4)
                if pdr_vlm_ped is not None and pdr_yolo_ped is not None else None,
            "delta_veh": round(pdr_vlm_veh - pdr_yolo_veh, 4)
                if pdr_vlm_veh is not None and pdr_yolo_veh is not None else None,
            # density bucket
            "bucket":         _bucket(n_pg),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in output:
            f.write(json.dumps(row) + "\n")
    console.print(f"[green]Saved → {out_path}[/green]")
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Summary display
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(records: list[dict]) -> None:
    """Print mean PDR_yolo vs PDR_vlm per model and per density bucket."""
    from collections import defaultdict

    # ── par modèle ───────────────────────────────────────────────────────────
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_model[r["model_name"]].append(r)

    tbl = Table(
        title="PDR YOLO vs PDR VLM — par modèle (piétons)",
        box=box.HEAVY_EDGE,
        header_style="bold cyan",
    )
    tbl.add_column("Model",        style="magenta")
    tbl.add_column("PDR YOLO",     justify="right")
    tbl.add_column("PDR VLM",      justify="right")
    tbl.add_column("Δ (VLM−YOLO)", justify="right")
    tbl.add_column("Frames (n_gt>0)", justify="right")

    for model, rows in sorted(by_model.items()):
        vals = [(r["pdr_yolo_ped"], r["pdr_vlm_ped"], r["delta_ped"])
                for r in rows
                if r["n_persons_gt"] > 0
                and r["pdr_yolo_ped"] is not None
                and r["pdr_vlm_ped"] is not None]
        if not vals:
            continue
        mean_yolo  = sum(v[0] for v in vals) / len(vals)
        mean_vlm   = sum(v[1] for v in vals) / len(vals)
        mean_delta = sum(v[2] for v in vals) / len(vals)
        color = "green" if mean_delta >= 0 else "red"
        tbl.add_row(
            model,
            f"{mean_yolo:.3f}",
            f"{mean_vlm:.3f}",
            f"[{color}]{mean_delta:+.3f}[/{color}]",
            str(len(vals)),
        )
    console.print(tbl)

    # ── par bucket de densité (tous modèles confondus) ───────────────────────
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r["n_persons_gt"] > 0:
            by_bucket[r["bucket"]].append(r)

    tbl2 = Table(
        title="PDR YOLO vs PDR VLM — par densité (piétons, tous modèles)",
        box=box.HEAVY_EDGE,
        header_style="bold cyan",
    )
    tbl2.add_column("Bucket (n_ped_gt)", style="cyan")
    tbl2.add_column("PDR YOLO",          justify="right")
    tbl2.add_column("PDR VLM",           justify="right")
    tbl2.add_column("Δ (VLM−YOLO)",      justify="right")
    tbl2.add_column("Frames",            justify="right")

    bucket_order = [b[0] for b in _BUCKET_DEFS]
    for bucket in bucket_order:
        rows = by_bucket.get(bucket, [])
        if not rows:
            continue
        vals = [(r["pdr_yolo_ped"], r["pdr_vlm_ped"], r["delta_ped"])
                for r in rows
                if r["pdr_yolo_ped"] is not None and r["pdr_vlm_ped"] is not None]
        if not vals:
            continue
        mean_yolo  = sum(v[0] for v in vals) / len(vals)
        mean_vlm   = sum(v[1] for v in vals) / len(vals)
        mean_delta = sum(v[2] for v in vals) / len(vals)
        color = "green" if mean_delta >= 0 else "red"
        tbl2.add_row(
            bucket,
            f"{mean_yolo:.3f}",
            f"{mean_vlm:.3f}",
            f"[{color}]{mean_delta:+.3f}[/{color}]",
            str(len(vals)),
        )
    console.print(tbl2)

    # ── verdict ──────────────────────────────────────────────────────────────
    all_vals = [(r["pdr_yolo_ped"], r["pdr_vlm_ped"], r["delta_ped"])
                for r in records
                if r["n_persons_gt"] > 0
                and r["pdr_yolo_ped"] is not None
                and r["pdr_vlm_ped"] is not None]
    if all_vals:
        global_yolo  = sum(v[0] for v in all_vals) / len(all_vals)
        global_vlm   = sum(v[1] for v in all_vals) / len(all_vals)
        global_delta = sum(v[2] for v in all_vals) / len(all_vals)
        console.print()
        console.print(f"[bold]Global :[/bold]  PDR YOLO = {global_yolo:.3f}  |  PDR VLM = {global_vlm:.3f}  |  Δ = {global_delta:+.3f}")
        if global_delta < -0.05:
            console.print("[bold yellow]→ H1 probable : YOLO est le plafond. VLM suit YOLO mais YOLO rate des entités.[/bold yellow]")
        elif global_delta > 0.05:
            console.print("[bold green]→ H2 probable : VLM dépasse YOLO. Le format de hint n'est pas le goulot d'étranglement.[/bold green]")
        else:
            console.print("[bold]→ YOLO PDR ≈ VLM PDR : les deux se suivent. Regarder par bucket pour plus de précision.[/bold]")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare PDR YOLO vs PDR VLM pour un run complexity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--run",   required=True, help="Chemin vers un run complexity (ex: runs/complexity/20260415_144802)")
    p.add_argument("--out",   default=None,  help="Chemin de sortie du JSONL (défaut: <run>/raw/diag_yolo_pdr.jsonl)")
    p.add_argument("--model", default=None,  help="Filtrer sur un seul modèle (ex: gpt-4o-mini)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run)
    if not run_dir.exists():
        console.print(f"[red]Run directory not found: {run_dir}[/red]")
        sys.exit(1)

    out_path = Path(args.out) if args.out else run_dir / "raw" / "diag_yolo_pdr.jsonl"

    records = run_diagnostic(run_dir, out_path, model_filter=args.model)
    _print_summary(records)


if __name__ == "__main__":
    main()
