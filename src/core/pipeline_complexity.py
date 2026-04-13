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
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _infer_complexity_window(
    model, model_name: str, cw, N: int,
    prompt: str, benchmark_cfg: dict,
    raw_log_path: Path, parsed_log_path: Path, frame_scores_log_path: Path,
    raw_lock: threading.Lock, parsed_lock: threading.Lock, frame_lock: threading.Lock,
):
    """Process a single complexity window: infer → parse → score. Thread-safe."""
    window  = cw.window
    clip_id = window.clip_id

    vlm_out = model.infer(window.frames, prompt)
    vlm_out.clip_id      = clip_id
    vlm_out.center_frame = window.center_frame
    vlm_out.frame_names  = window.frame_names

    _meta = {
        "n_persons_gt":  cw.n_persons_gt,
        "n_vehicles_gt": cw.n_vehicles_gt,
        "n_entities_gt": cw.n_entities_gt,
    }

    if benchmark_cfg["output"].get("save_raw_outputs"):
        record = json.dumps({
            "model": model_name, "N": N,
            "clip_id": clip_id,
            "center_frame": window.center_frame,
            "raw_text": vlm_out.raw_text,
            "latency_s": vlm_out.latency_s,
            "prompt_tokens":     vlm_out.prompt_tokens,
            "completion_tokens": vlm_out.completion_tokens,
            **_meta,
        }) + "\n"
        with raw_lock:
            with open(raw_log_path, "a") as f:
                f.write(record)

    parsed = parse(vlm_out.raw_text, window_size=N)

    if benchmark_cfg["output"].get("save_parsed_outputs"):
        record = json.dumps({
            "model": model_name, "N": N,
            "clip_id": clip_id,
            "center_frame": window.center_frame,
            "frame_names": window.frame_names,
            "parse_success": parsed.parse_success,
            "parsed": dataclasses.asdict(parsed),
            **_meta,
        }) + "\n"
        with parsed_lock:
            with open(parsed_log_path, "a") as f:
                f.write(record)

    frame_scores = score_window(
        parsed, window.annotations, model_name,
        clip_id, window.frame_names, N,
    )

    # Predicted entity counts for the center frame
    center_win_idx = window.frame_names.index(window.center_frame)
    if parsed.parse_success and center_win_idx < len(parsed.frames):
        fo = parsed.frames[center_win_idx]
        n_persons_pred  = len(fo.pedestrians) if fo.pedestrians  else 0
        n_vehicles_pred = len(fo.vehicles)    if fo.vehicles     else 0
    else:
        n_persons_pred  = 0
        n_vehicles_pred = 0

    frame_score_lines = "".join(
        json.dumps({
            **_serialize_frame_score(fs),
            **_meta,
            "n_persons_pred":  n_persons_pred,
            "n_vehicles_pred": n_vehicles_pred,
        }) + "\n"
        for fs in frame_scores
    )
    with frame_lock:
        with open(frame_scores_log_path, "a") as f:
            f.write(frame_score_lines)

    return frame_scores, vlm_out.latency_s, vlm_out.prompt_tokens or 0, vlm_out.completion_tokens or 0


def _run_single_complexity_model(
    model_name: str, model, context: PipelineContext,
    window_sizes: list[int], strategy: str, max_res,
    frames_per_count: int, max_entities, total_frames,
    progress: Progress,
) -> tuple[list, dict, dict]:
    """Run all (N, window) iterations for one model against a shared progress bar."""
    model_scores: list = []
    model_lat:    dict = {}
    model_tok:    dict = {}

    for N in window_sizes:
        key = (model_name, N)
        model_lat[key] = []
        model_tok[key] = {"prompt": 0, "completion": 0}

        _raw_dir              = context.results_dir / "raw"
        _prefix               = f"{model_name}_N{N}_"
        raw_log_path          = _raw_dir / f"{_prefix}raw_outputs.jsonl"
        parsed_log_path       = _raw_dir / f"{_prefix}parsed_outputs.jsonl"
        frame_scores_log_path = _raw_dir / f"{_prefix}frame_scores.jsonl"

        # Pre-build and sort windows by entity count for even difficulty spread
        all_cw = []
        for clip in context.clips:
            all_cw.extend(sample_complexity_windows(
                clip, N, strategy, max_res, frames_per_count, max_entities,
            ))
        all_cw.sort(key=lambda cw: cw.n_entities_gt)
        if total_frames is not None and len(all_cw) > total_frames:
            _step = len(all_cw) / total_frames
            all_cw = [all_cw[round(i * _step)] for i in range(total_frames)]

        workers     = model.parallel_workers
        raw_lock    = threading.Lock()
        parsed_lock = threading.Lock()
        frame_lock  = threading.Lock()

        _win_args = dict(
            model=model, model_name=model_name, N=N,
            prompt=context.prompt, benchmark_cfg=context.benchmark_cfg,
            raw_log_path=raw_log_path, parsed_log_path=parsed_log_path,
            frame_scores_log_path=frame_scores_log_path,
            raw_lock=raw_lock, parsed_lock=parsed_lock, frame_lock=frame_lock,
        )

        task_id = progress.add_task(
            f"[cyan]{model_name} (N={N})" + (f" [dim]· {workers}t[/dim]" if workers > 1 else ""),
            total=len(all_cw),
        )

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_infer_complexity_window, cw=cw, **_win_args): cw
                    for cw in all_cw
                }
                for future in as_completed(futures):
                    cw = futures[future]
                    try:
                        frame_scores, lat, pt, ct = future.result()
                        model_scores.extend(frame_scores)
                        model_lat[key].append(lat)
                        model_tok[key]["prompt"]     += pt
                        model_tok[key]["completion"] += ct
                    except (KeyError, AttributeError, ImportError):
                        raise
                    except Exception:
                        console.print(f"[red]✗ Error on {cw.window.clip_id}/{cw.window.center_frame}[/red]")
                        console.print(traceback.format_exc(limit=2), style="dim red")
                    progress.update(task_id, advance=1)
        else:
            for cw in all_cw:
                try:
                    dbg_frame(model_name, cw.window.clip_id, cw.window.center_frame, N)
                    frame_scores, lat, pt, ct = _infer_complexity_window(cw=cw, **_win_args)
                    model_scores.extend(frame_scores)
                    model_lat[key].append(lat)
                    model_tok[key]["prompt"]     += pt
                    model_tok[key]["completion"] += ct
                except (KeyError, AttributeError, ImportError):
                    raise
                except Exception:
                    console.print(f"[red]✗ Error on {cw.window.clip_id}/{cw.window.center_frame}[/red]")
                    console.print(traceback.format_exc(limit=2), style="dim red")
                progress.update(task_id, advance=1)

    return model_scores, model_lat, model_tok


def _run_inference(context: PipelineContext) -> InferenceResults:
    all_scores:   list = []
    latencies:    dict[tuple[str, int], list[float]]    = {}
    token_counts: dict[tuple[str, int], dict[str, int]] = {}

    window_sizes     = context.sampling_cfg["window_sizes"]
    strategy         = context.sampling_cfg.get("frame_selection", "uniform")
    max_res          = tuple(context.sampling_cfg["max_resolution"]) if context.sampling_cfg.get("max_resolution") else None
    frames_per_count = context.sampling_cfg.get("frames_per_count", 3)
    max_entities     = context.sampling_cfg.get("max_entities")
    total_frames     = context.sampling_cfg.get("total_frames")

    api_models   = {n: m for n, m in context.models.items() if m.parallel_workers > 1}
    local_models = {n: m for n, m in context.models.items() if m.parallel_workers <= 1}

    def _collect(scores, lats, toks):
        all_scores.extend(scores)
        for k, v in lats.items():
            latencies.setdefault(k, []).extend(v)
        for k, v in toks.items():
            tc = token_counts.setdefault(k, {"prompt": 0, "completion": 0})
            tc["prompt"]     += v["prompt"]
            tc["completion"] += v["completion"]

    _model_kwargs = dict(
        context=context, window_sizes=window_sizes, strategy=strategy, max_res=max_res,
        frames_per_count=frames_per_count, max_entities=max_entities, total_frames=total_frames,
    )

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
        console=console,
    ) as progress:

        # ── API models: load all then run in parallel ─────────────────────────
        if api_models:
            loaded_api: dict = {}
            for name, model in api_models.items():
                try:
                    model.load()
                    loaded_api[name] = model
                except Exception:
                    console.print(f"[bold red]✗ Failed to load {name}[/bold red]")
                    console.print(traceback.format_exc(limit=3), style="dim red")

            if loaded_api:
                console.print(
                    f"[bold cyan]Running {len(loaded_api)} API model(s) in parallel "
                    f"({list(loaded_api)} · {next(iter(loaded_api.values())).parallel_workers} window threads each)[/bold cyan]"
                )
                with ThreadPoolExecutor(max_workers=len(loaded_api)) as executor:
                    futures = {
                        executor.submit(
                            _run_single_complexity_model,
                            name, model, progress=progress, **_model_kwargs,
                        ): name
                        for name, model in loaded_api.items()
                    }
                    for future in as_completed(futures):
                        name = futures[future]
                        try:
                            scores, lats, toks = future.result()
                            _collect(scores, lats, toks)
                        except (KeyError, AttributeError, ImportError):
                            raise
                        except Exception:
                            console.print(f"[bold red]✗ Model {name} failed[/bold red]")
                            console.print(traceback.format_exc(limit=3), style="dim red")

            for model in loaded_api.values():
                try:
                    model.unload()
                except Exception:
                    pass

        # ── Local models: sequential ──────────────────────────────────────────
        for model_name, model in local_models.items():
            with console.status(f"[bold yellow]Loading model [magenta]{model_name}[/magenta]...", spinner="dots"):
                try:
                    model.load()
                except Exception:
                    console.print(f"[bold red]✗ Failed to load {model_name}[/bold red]")
                    console.print(traceback.format_exc(limit=3), style="dim red")
                    continue
            try:
                scores, lats, toks = _run_single_complexity_model(
                    model_name, model, progress=progress, **_model_kwargs,
                )
                _collect(scores, lats, toks)
            finally:
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
        context   = load_pipeline(models_cfg_path, clips_cfg_path, benchmark_cfg_path, selected_models, mode="complexity")
        results   = _run_inference(context)
        summaries = run_scoring(results)
        save_scores(summaries, context)
        return context.results_dir
    except Exception:
        console.print("[bold red]Critical pipeline failure[/bold red]")
        console.print(traceback.format_exc())
        raise
