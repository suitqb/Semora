"""Complexity pipeline — Plan 3 (scene density scaling).

Reuses load_pipeline / run_scoring / save_scores from pipeline.py.
Only the inference loop differs: frames are selected by GT entity density
via complexity_sampler instead of uniform temporal sampling.

Each JSONL record is enriched with `bucket`, `n_persons_gt`, `n_vehicles_gt`
so that post-run analysis can stratify scores by complexity level without
re-reading GT CSVs.
"""

from __future__ import annotations
import dataclasses
import json
import traceback
from pathlib import Path

from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TimeElapsedColumn, MofNCompleteColumn,
)

from ..core.console import console
from ..core.utils import dbg_frame, dbg_parse
from ..parsing.output_parser import parse
from ..scoring.titan_scorer import score_window, FrameScore
from ..sampling.complexity_sampler import sample_complexity_windows
from rich.panel import Panel

from .pipeline import (
    load_pipeline, run_scoring, save_scores,
    PipelineContext, InferenceResults,
)


def _serialize_frame_score(fs: FrameScore) -> dict:
    return {
        "clip_id":      fs.clip_id,
        "center_frame": fs.center_frame,
        "model_name":   fs.model_name,
        "window_size":  fs.window_size,
        "parse_success": fs.parse_success,
        "person_scores": {
            field: {"precision": s.precision, "recall": s.recall, "f1": s.f1}
            for field, s in fs.person_scores.items()
        },
        "vehicle_scores": {
            field: {"precision": s.precision, "recall": s.recall, "f1": s.f1}
            for field, s in fs.vehicle_scores.items()
        },
    }


def _run_inference(context: PipelineContext) -> InferenceResults:
    all_scores  = []
    latencies: dict[tuple[str, int], list[float]] = {}
    token_counts: dict[tuple[str, int], dict[str, int]] = {}

    window_sizes      = context.sampling_cfg["window_sizes"]
    strategy          = context.sampling_cfg.get("frame_selection", "uniform")
    max_res           = tuple(context.sampling_cfg["max_resolution"]) if context.sampling_cfg.get("max_resolution") else None
    frames_per_count = context.sampling_cfg.get("frames_per_count", 3)
    max_entities     = context.sampling_cfg.get("max_entities")
    total_frames     = context.sampling_cfg.get("total_frames")

    for model_name, model in context.models.items():
        with console.status(f"[bold yellow]Loading model [magenta]{model_name}[/magenta]...", spinner="dots"):
            try:
                model.load()
            except Exception:
                console.print(f"[bold red]✗ Failed to load {model_name}[/bold red]")
                console.print(traceback.format_exc(limit=3), style="dim red")
                continue

        for N in window_sizes:
            key = (model_name, N)
            latencies[key]    = []
            token_counts[key] = {"prompt": 0, "completion": 0}

            _raw_dir        = context.results_dir / "raw"
            _prefix         = f"{model_name}_N{N}_"
            raw_log_path          = _raw_dir / f"{_prefix}raw_outputs.jsonl"
            parsed_log_path       = _raw_dir / f"{_prefix}parsed_outputs.jsonl"
            frame_scores_log_path = _raw_dir / f"{_prefix}frame_scores.jsonl"

            # Pre-build all windows so we have an accurate total for the progress bar
            all_cw = []
            for clip in context.clips:
                all_cw.extend(sample_complexity_windows(
                    clip, N, strategy, max_res, frames_per_count, max_entities,
                ))

            # Select total_frames evenly spread across the full difficulty spectrum
            all_cw.sort(key=lambda cw: cw.n_entities_gt)
            if total_frames is not None and len(all_cw) > total_frames:
                step = len(all_cw) / total_frames
                all_cw = [all_cw[round(i * step)] for i in range(total_frames)]

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                console=console,
            ) as progress:
                task_id = progress.add_task(
                    f"[cyan]Processing {model_name} (N={N})", total=len(all_cw),
                )

                for cw in all_cw:
                    window  = cw.window
                    clip_id = window.clip_id
                    try:
                        dbg_frame(model_name, clip_id, window.center_frame, N)
                        vlm_out = model.infer(window.frames, context.prompt)
                        vlm_out.clip_id       = clip_id
                        vlm_out.center_frame  = window.center_frame
                        vlm_out.frame_names   = window.frame_names

                        _meta = {
                            "n_persons_gt":  cw.n_persons_gt,
                            "n_vehicles_gt": cw.n_vehicles_gt,
                            "n_entities_gt": cw.n_entities_gt,
                        }

                        if context.benchmark_cfg["output"].get("save_raw_outputs"):
                            with open(raw_log_path, "a") as f:
                                f.write(json.dumps({
                                    "model": model_name, "N": N,
                                    "clip_id": clip_id,
                                    "center_frame": window.center_frame,
                                    "raw_text": vlm_out.raw_text,
                                    "latency_s": vlm_out.latency_s,
                                    "prompt_tokens":     vlm_out.prompt_tokens,
                                    "completion_tokens": vlm_out.completion_tokens,
                                    **_meta,
                                }) + "\n")

                        parsed = parse(vlm_out.raw_text, window_size=N)
                        dbg_parse(parsed.parse_success, len(parsed.frames), N, parsed.parse_error)

                        if context.benchmark_cfg["output"].get("save_parsed_outputs"):
                            with open(parsed_log_path, "a") as f:
                                f.write(json.dumps({
                                    "model": model_name, "N": N,
                                    "clip_id": clip_id,
                                    "center_frame": window.center_frame,
                                    "frame_names": window.frame_names,
                                    "parse_success": parsed.parse_success,
                                    "parsed": dataclasses.asdict(parsed),
                                    **_meta,
                                }) + "\n")

                        frame_scores = score_window(
                            parsed, window.annotations, model_name,
                            clip_id, window.frame_names, N,
                        )
                        all_scores.extend(frame_scores)

                        # Predicted entity counts for the center frame
                        center_win_idx = window.frame_names.index(window.center_frame)
                        if parsed.parse_success and center_win_idx < len(parsed.frames):
                            fo = parsed.frames[center_win_idx]
                            n_persons_pred  = len(fo.pedestrians) if fo.pedestrians  else 0
                            n_vehicles_pred = len(fo.vehicles)    if fo.vehicles     else 0
                        else:
                            n_persons_pred  = 0
                            n_vehicles_pred = 0

                        # Save per-frame scores with complexity metadata
                        with open(frame_scores_log_path, "a") as f:
                            for fs in frame_scores:
                                f.write(json.dumps({
                                    **_serialize_frame_score(fs),
                                    **_meta,
                                    "n_persons_pred":  n_persons_pred,
                                    "n_vehicles_pred": n_vehicles_pred,
                                }) + "\n")

                        latencies[key].append(vlm_out.latency_s)
                        token_counts[key]["prompt"]     += vlm_out.prompt_tokens or 0
                        token_counts[key]["completion"] += vlm_out.completion_tokens or 0

                    except (KeyError, AttributeError, ImportError):
                        raise
                    except Exception:
                        console.print(f"[red]✗ Error on {clip_id}/{window.center_frame}[/red]")
                        console.print(traceback.format_exc(limit=2), style="dim red")
                    progress.update(task_id, advance=1)

        try:
            model.unload()
        except Exception:
            pass

    return InferenceResults(all_scores, [], latencies, token_counts)


def run_complexity(
    models_cfg_path: Path,
    clips_cfg_path: Path,
    benchmark_cfg_path: Path,
    selected_models: list[str] | None = None,
) -> Path:
    """Complexity pipeline entry point (Plan 3)."""
    try:
        context  = load_pipeline(models_cfg_path, clips_cfg_path, benchmark_cfg_path, selected_models, mode="complexity")

        results  = _run_inference(context)
        summaries = run_scoring(results)
        save_scores(summaries, context)
        return context.results_dir
    except Exception:
        console.print("[bold red]Critical pipeline failure[/bold red]")
        console.print(traceback.format_exc())
        raise
