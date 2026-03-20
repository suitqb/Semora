from __future__ import annotations
import json
import traceback
from datetime import datetime
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.live import Live
from rich import box

from ..models.base import BaseVLM
from ..models.registry import build_models
from ..sampling.clip_loader import load_all_clips
from ..sampling.frame_sampler import sample_windows
from ..parsing.output_parser import parse
from ..scoring.titan_scorer import score_frame
from ..scoring.aggregator import aggregate

console = Console()

def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run(
    models_cfg_path: Path,
    clips_cfg_path: Path,
    benchmark_cfg_path: Path,
) -> Path:
    models_cfg    = _load_yaml(models_cfg_path)["models"]
    clips_cfg     = _load_yaml(clips_cfg_path)
    benchmark_cfg = _load_yaml(benchmark_cfg_path)["benchmark"]

    sampling_cfg = clips_cfg["sampling"]
    window_sizes = sampling_cfg["window_sizes"]
    strategy     = sampling_cfg.get("frame_selection", "uniform")
    max_res      = tuple(sampling_cfg["max_resolution"]) if sampling_cfg.get("max_resolution") else None

    prompt = Path(benchmark_cfg["prompt"]["file"]).read_text().strip()

    run_id = benchmark_cfg.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(benchmark_cfg["output"]["results_dir"]) / run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    raw_log_path    = results_dir / "raw_outputs.jsonl"
    parsed_log_path = results_dir / "parsed_outputs.jsonl"

    clips = load_all_clips(clips_cfg)
    
    header = Panel(
        f"[bold blue]Run ID:[/bold blue] {run_id}\n"
        f"[bold blue]Clips:[/bold blue] {len(clips)} | "
        f"[bold blue]Modèles:[/bold blue] {len(models_cfg)} | "
        f"[bold blue]Windows:[/bold blue] {window_sizes}",
        title="[bold white]Semora Benchmark[/bold white]",
        border_style="bright_magenta",
        box=box.ROUNDED
    )
    console.print(header)

    models: dict[str, BaseVLM] = build_models(models_cfg)

    all_scores   = []
    latencies    = {}
    token_counts = {}

    for model_name, model in models.items():
        with console.status(f"[bold yellow]Chargement du modèle [magenta]{model_name}[/magenta]...", spinner="dots"):
            try:
                model.load()
            except Exception:
                console.print(f"[bold red]✗ Échec du chargement de {model_name}[/bold red]")
                console.print(traceback.format_exc(limit=3), style="dim red")
                continue

        for N in window_sizes:
            key = (model_name, N)
            latencies[key]    = []
            token_counts[key] = {"prompt": 0, "completion": 0}

            total_windows = sum(len(sample_windows(c, window_size=N, strategy=strategy, max_resolution=max_res)) for c in clips)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task_id = progress.add_task(f"[cyan]Traitement {model_name} (N={N})", total=total_windows)

                for clip in clips:
                    windows = sample_windows(
                        clip, window_size=N, strategy=strategy, max_resolution=max_res
                    )

                    for window in windows:
                        try:
                            vlm_out = model.infer(window.frames, prompt)
                            vlm_out.clip_id      = clip.clip_id
                            vlm_out.center_frame = window.center_frame
                            vlm_out.frame_names  = window.frame_names

                            if benchmark_cfg["output"].get("save_raw_outputs"):
                                with open(raw_log_path, "a") as f:
                                    f.write(json.dumps({
                                        "model": model_name, "N": N,
                                        "clip_id": clip.clip_id,
                                        "center_frame": window.center_frame,
                                        "raw_text": vlm_out.raw_text,
                                        "latency_s": vlm_out.latency_s,
                                    }) + "\n")

                            parsed = parse(vlm_out.raw_text)

                            if benchmark_cfg["output"].get("save_parsed_outputs"):
                                with open(parsed_log_path, "a") as f:
                                    f.write(json.dumps({
                                        "model": model_name, "N": N,
                                        "clip_id": clip.clip_id,
                                        "center_frame": window.center_frame,
                                        "parse_success": parsed.parse_success,
                                        "parsed": {
                                            "scene_context": parsed.scene_context,
                                            "pedestrians": parsed.pedestrians,
                                            "vehicles": parsed.vehicles,
                                        },
                                    }) + "\n")

                            frame_score = score_frame(
                                parsed=parsed,
                                annotation=window.annotation,
                                model_name=model_name,
                                clip_id=clip.clip_id,
                                center_frame=window.center_frame,
                                window_size=N,
                            )
                            all_scores.append(frame_score)
                            latencies[key].append(vlm_out.latency_s)
                            token_counts[key]["prompt"]     += vlm_out.prompt_tokens or 0
                            token_counts[key]["completion"] += vlm_out.completion_tokens or 0
                            
                        except Exception:
                            console.print(f"[red]✗ Erreur sur {clip.clip_id}/{window.center_frame}[/red]")
                            console.print(traceback.format_exc(limit=2), style="dim red")
                        
                        progress.update(task_id, advance=1)

        try:
            model.unload()
        except Exception:
            pass

    console.print("\n[bold green]✓ Benchmark terminé. Agrégation des scores...[/bold green]")

    summaries = aggregate(all_scores, latencies, token_counts)
    
    # Affichage d'un beau tableau de résultats
    table = Table(title="Résumé des Scores", box=box.HEAVY_EDGE, header_style="bold cyan")
    table.add_column("Modèle", justify="left", style="magenta")
    table.add_column("N", justify="center")
    table.add_column("F1 Context", justify="right")
    table.add_column("F1 Ped", justify="right")
    table.add_column("F1 Veh", justify="right")
    table.add_column("Latence (s)", justify="right")

    for s in summaries:
        table.add_row(
            s.model_name,
            str(s.window_size),
            f"{s.f1_context:.3f}" if s.f1_context is not None else "-",
            f"{s.f1_pedestrians:.3f}" if s.f1_pedestrians is not None else "-",
            f"{s.f1_vehicles:.3f}" if s.f1_vehicles is not None else "-",
            f"{s.avg_latency_s:.2f}"
        )
    
    console.print(table)

    scores_path = results_dir / "scores.json"
    with open(scores_path, "w") as f:
        json.dump([s.__dict__ for s in summaries], f, indent=2)

    console.print(Panel(f"Résultats sauvegardés dans : [bold white]{results_dir}[/bold white]", border_style="green"))

    return results_dir

