from __future__ import annotations
import dataclasses
import json
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..tracking.context_builder import (
    build_tracking_context_from_detections,
    build_detection_context,
)

_TRACKING_PROMPT_FILE   = Path("prompts/extraction_v4_tracking.txt")
_DETECTION_PROMPT_FILE  = Path("prompts/complexity_v2_detection.txt")

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, MofNCompleteColumn
from rich import box

from ..core.console import console
from ..core.utils import dbg, dbg_frame, dbg_parse, dbg_judge, dbg_judge_skip, dbg_judge_error
from ..models.registry import build_models
from ..sampling.clip_loader import load_all_clips
from ..sampling.frame_sampler import sample_windows
from ..parsing.output_parser import parse
from ..scoring.titan_scorer import score_window
from ..scoring.llm_judge import judge
from ..scoring.aggregator import aggregate, build_scores_payload

if TYPE_CHECKING:
    from ..models.base import BaseVLM
    from ..sampling.clip_loader import TITANClip
    from ..scoring.titan_scorer import FrameScore
    from ..scoring.llm_judge import JudgeScore
    from ..scoring.aggregator import ModelSummary

@dataclass
class PipelineContext:
    """Groups all resources and configuration needed for the benchmark run."""
    models: dict[str, BaseVLM]
    clips: list[TITANClip]
    benchmark_cfg: dict
    sampling_cfg: dict
    results_dir: Path
    run_id: str
    prompt: str
    mode: str = "extraction"
    tracking_enabled: bool = False
    fallback_prompt: str = ""   # non-tracking prompt, used when tracking context is empty
    live_tracker: object = None  # LiveTracker instance (set when tracking_enabled=True)

@dataclass
class InferenceResults:
    """Aggregated outputs and scores from the inference loop."""
    all_scores: list[FrameScore]
    judge_scores: list[JudgeScore]
    latencies: dict[tuple[str, int], list[float]]
    token_counts: dict[tuple[str, int], dict[str, int]]

def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def load_pipeline(
    models_cfg_path: Path,
    clips_cfg_path: Path,
    benchmark_cfg_path: Path,
    selected_models: list[str] | None = None,
    mode: str = "extraction",
) -> PipelineContext:
    """Load configurations, clips, and models required for the pipeline."""
    models_cfg    = _load_yaml(models_cfg_path)["models"]
    clips_cfg     = _load_yaml(clips_cfg_path)
    benchmark_cfg = _load_yaml(benchmark_cfg_path)["benchmark"]

    if selected_models:
        console.print(f"[bold cyan]Filtering models:[/bold cyan] {', '.join(selected_models)}")
        for m_key in models_cfg:
            models_cfg[m_key]["enabled"] = m_key in selected_models

    sampling_cfg = clips_cfg["sampling"]
    tracking_enabled = benchmark_cfg.get("features", {}).get("tracking", False)

    prompt_cfg = clips_cfg.get("prompt") or benchmark_cfg["prompt"]
    prompt_file = Path(prompt_cfg["file"])
    base_prompt = prompt_file.read_text().strip()

    if tracking_enabled:
        from ..tracking.detector import LiveTracker
        live_tracker = LiveTracker()
        if mode == "complexity":
            prompt = _DETECTION_PROMPT_FILE.read_text().strip()
            fallback_prompt = base_prompt
            console.print("[bold cyan]Detection enabled — YOLO will annotate each frame before inference.[/bold cyan]")
        else:
            prompt = _TRACKING_PROMPT_FILE.read_text().strip()
            fallback_prompt = base_prompt
            console.print("[bold cyan]Live tracking enabled — YOLO will run per window.[/bold cyan]")
    else:
        prompt = base_prompt
        fallback_prompt = ""
        live_tracker = None


    run_id = benchmark_cfg.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(benchmark_cfg["output"]["runs_dir"]) / mode / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "raw").mkdir(exist_ok=True)

    clips = load_all_clips(clips_cfg)
    models = build_models(models_cfg)
    model_names = ", ".join(models.keys())

    mode_label = {
        "extraction": "Extraction  — Plans 1 & 2",
        "complexity": "Complexity  — Plan 3",
    }.get(mode, mode)

    header = Panel(
        f"[bold blue]Run ID:[/bold blue] {run_id}\n"
        f"[bold blue]Clips:[/bold blue] {len(clips)} | "
        f"[bold blue]Models ({len(models)}):[/bold blue] {model_names}",
        title=f"[bold white]Semora Benchmark[/bold white]  [dim]·  {mode_label}[/dim]",
        border_style="bright_magenta",
        box=box.ROUNDED
    )
    console.print(header)

    return PipelineContext(
        models=models, clips=clips, benchmark_cfg=benchmark_cfg,
        sampling_cfg=sampling_cfg, results_dir=results_dir, run_id=run_id, prompt=prompt,
        mode=mode, tracking_enabled=tracking_enabled,
        fallback_prompt=fallback_prompt, live_tracker=live_tracker,
    )

def _build_clip_detection_cache(
    tracker,
    clip: "TITANClip",
    max_res: tuple | None,
) -> dict[str, list[dict]]:
    """Scan every frame of a clip in chronological order with ByteTrack.

    ByteTrack sees every frame — same temporal context as offline preprocessing —
    while adding negligible wall-clock time (YOLO ~50 ms/frame vs VLM ~2-5 s/frame).
    Returns a dict {frame_name: [detections]} for all frames in the clip.
    """
    cache: dict[str, list[dict]] = {}
    for frame_name in clip.frame_names:
        img = clip.get_frame(frame_name, max_resolution=max_res)
        cache[frame_name] = tracker.process_frames([img])[0]
    return cache


def _resolve_window_prompt(
    context: PipelineContext,
    clip_id: str,
    frame_names: list[str],
    frames: list | None = None,
    det_cache: dict[str, list[dict]] | None = None,
) -> str:
    """Return the prompt for this window, injecting YOLO context when enabled.

    - complexity mode (N=1): plain detection → {detection_context}
    - extraction mode (N>1): ByteTrack → {tracking_context}

    When det_cache is provided (built by _build_clip_detection_cache), detections
    come from the full-clip scan and ByteTrack quality matches offline preprocessing.
    Without a cache, only the window frames are seen — tracks may break at gaps.
    """
    if not context.tracking_enabled or context.live_tracker is None:
        return context.prompt

    actual_frames = frames or []

    if context.mode == "complexity":
        # Single-frame detection — no tracking state needed
        frame = actual_frames[0] if actual_frames else None
        if frame is None:
            return context.fallback_prompt
        detections = context.live_tracker.detect_frame(frame)
        detection_ctx = build_detection_context(detections)
        if not detection_ctx:
            return context.fallback_prompt
        return context.prompt.replace("{detection_context}", detection_ctx)
    else:
        # Multi-frame tracking
        if det_cache is not None:
            # Full-clip scan: every frame was fed to ByteTrack in order → stable IDs
            frame_detections = [det_cache.get(fn, []) for fn in frame_names]
        else:
            # Fallback: only window frames are visible to ByteTrack (degraded quality)
            frame_detections = context.live_tracker.process_frames(actual_frames)
        tracking_ctx = build_tracking_context_from_detections(frame_detections)
        if not tracking_ctx:
            return context.fallback_prompt
        return context.prompt.replace("{tracking_context}", tracking_ctx)


def _infer_window(
    model, model_name: str, clip, window, N: int,
    prompt: str, benchmark_cfg: dict,
    raw_log_path: Path, parsed_log_path: Path, judge_log_path: Path,
    raw_lock: threading.Lock, parsed_lock: threading.Lock, judge_lock: threading.Lock,
):
    """Process a single window: infer → parse → score → judge. Thread-safe."""
    vlm_out = model.infer(window.frames, prompt)
    vlm_out.clip_id, vlm_out.center_frame, vlm_out.frame_names = clip.clip_id, window.center_frame, window.frame_names

    if benchmark_cfg["output"].get("save_raw_outputs"):
        record = json.dumps({"model": model_name, "N": N, "clip_id": clip.clip_id, "center_frame": window.center_frame, "raw_text": vlm_out.raw_text, "latency_s": vlm_out.latency_s}) + "\n"
        with raw_lock:
            with open(raw_log_path, "a") as f:
                f.write(record)

    parsed = parse(vlm_out.raw_text, window_size=N)

    if benchmark_cfg["output"].get("save_parsed_outputs"):
        record = json.dumps({"model": model_name, "N": N, "clip_id": clip.clip_id, "center_frame": window.center_frame, "frame_names": window.frame_names, "parse_success": parsed.parse_success, "parsed": dataclasses.asdict(parsed)}) + "\n"
        with parsed_lock:
            with open(parsed_log_path, "a") as f:
                f.write(record)

    frame_scores = score_window(parsed, window.annotations, model_name, clip.clip_id, window.frame_names, N)

    judge_score = None
    judge_cfg = benchmark_cfg["scorers"].get("llm_judge")
    if judge_cfg and judge_cfg.get("enabled"):
        if parsed.parse_success:
            judge_score = judge(parsed, window.annotation, model_name, clip.clip_id, window.center_frame, N, judge_cfg)
            if benchmark_cfg["output"].get("save_parsed_outputs"):
                record = json.dumps({"model": model_name, "N": N, "clip_id": clip.clip_id, "center_frame": window.center_frame, "judge_model": judge_cfg.get("model_id"), "scores": judge_score.__dict__}) + "\n"
                with judge_lock:
                    with open(judge_log_path, "a") as f:
                        f.write(record)

    return frame_scores, judge_score, vlm_out.latency_s, vlm_out.prompt_tokens or 0, vlm_out.completion_tokens or 0


def _run_single_model(
    model_name: str, model, context: PipelineContext,
    max_res, step: int,
    progress: Progress,
) -> tuple[list, list, dict, dict]:
    """Run all (clip, frame) iterations for one model against a shared progress bar."""
    model_scores: list = []
    model_judge: list  = []
    model_lat:   dict  = {}
    model_tok:   dict  = {}

    N = 1
    key = (model_name, N)
    model_lat[key] = []
    model_tok[key] = {"prompt": 0, "completion": 0}

    _raw_dir        = context.results_dir / "raw"
    _prefix         = f"{model_name}_"
    raw_log_path    = _raw_dir / f"{_prefix}raw_outputs.jsonl"
    parsed_log_path = _raw_dir / f"{_prefix}parsed_outputs.jsonl"
    judge_log_path  = _raw_dir / f"{_prefix}judge_outputs.jsonl"
    total_windows   = sum(len(sample_windows(c, max_res, step)) for c in context.clips)

    workers = model.parallel_workers
    raw_lock, parsed_lock, judge_lock = threading.Lock(), threading.Lock(), threading.Lock()
    _win_args = dict(
        model=model, model_name=model_name, N=N,
        prompt=context.prompt, benchmark_cfg=context.benchmark_cfg,
        raw_log_path=raw_log_path, parsed_log_path=parsed_log_path, judge_log_path=judge_log_path,
        raw_lock=raw_lock, parsed_lock=parsed_lock, judge_lock=judge_lock,
    )

    task_id = progress.add_task(
        f"[cyan]{model_name}" + (f" [dim]· {workers}t[/dim]" if workers > 1 else ""),
        total=total_windows,
    )

    if workers > 1:
        clip_caches: dict[str, dict[str, list[dict]]] = {}
        if context.live_tracker is not None and context.mode != "complexity":
            for clip in context.clips:
                context.live_tracker.reset()
                console.print(
                    f"[cyan]  Scanning {clip.clip_id} ({len(clip.frame_names)} frames) "
                    f"for tracking context...[/cyan]"
                )
                clip_caches[clip.clip_id] = _build_clip_detection_cache(
                    context.live_tracker, clip, max_res
                )
        all_windows = [(clip, w) for clip in context.clips for w in sample_windows(clip, max_res, step)]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _infer_window, clip=clip, window=window,
                    **{**_win_args, "prompt": _resolve_window_prompt(
                        context, clip.clip_id, window.frame_names, window.frames,
                        det_cache=clip_caches.get(clip.clip_id),
                    )},
                ): (clip, window)
                for clip, window in all_windows
            }
            for future in as_completed(futures):
                clip, window = futures[future]
                try:
                    frame_scores, js, lat, pt, ct = future.result()
                    model_scores.extend(frame_scores)
                    if js:
                        model_judge.append(js)
                    model_lat[key].append(lat)
                    model_tok[key]["prompt"] += pt
                    model_tok[key]["completion"] += ct
                except (KeyError, AttributeError, ImportError):
                    raise
                except Exception:
                    console.print(f"[red]✗ Error on {clip.clip_id}/{window.center_frame}[/red]")
                    console.print(traceback.format_exc(), style="dim red")
                progress.update(task_id, advance=1)
    else:
        for clip in context.clips:
            det_cache: dict[str, list[dict]] | None = None
            if context.live_tracker is not None:
                context.live_tracker.reset()
                if context.mode != "complexity":
                    console.print(
                        f"[cyan]  Scanning {clip.clip_id} ({len(clip.frame_names)} frames) "
                        f"for tracking context...[/cyan]"
                    )
                    det_cache = _build_clip_detection_cache(
                        context.live_tracker, clip, max_res
                    )
            for window in sample_windows(clip, max_res, step):
                try:
                    dbg_frame(model_name, clip.clip_id, window.center_frame, N)
                    window_prompt = _resolve_window_prompt(
                        context, clip.clip_id, window.frame_names, window.frames,
                        det_cache=det_cache,
                    )
                    frame_scores, js, lat, pt, ct = _infer_window(clip=clip, window=window, **{**_win_args, "prompt": window_prompt})
                    model_scores.extend(frame_scores)
                    if js:
                        model_judge.append(js)
                        if js.judge_error and js.judge_error != "parse_failed":
                            dbg_judge_error(js.judge_error)
                        else:
                            dbg_judge(js.completeness, js.semantic_richness, js.spatial_relations, js.overall)
                    model_lat[key].append(lat)
                    model_tok[key]["prompt"] += pt
                    model_tok[key]["completion"] += ct
                except (KeyError, AttributeError, ImportError):
                    raise
                except Exception:
                    console.print(f"[red]✗ Error on {clip.clip_id}/{window.center_frame}[/red]")
                    console.print(traceback.format_exc(), style="dim red")
                progress.update(task_id, advance=1)

    return model_scores, model_judge, model_lat, model_tok


def run_inference(context: PipelineContext) -> InferenceResults:
    """Execute the inference loop. API models run in parallel; local models run sequentially."""
    all_scores:   list[FrameScore] = []
    judge_scores: list[JudgeScore] = []
    latencies:    dict[tuple[str, int], list[float]]       = {}
    token_counts: dict[tuple[str, int], dict[str, int]]    = {}

    max_res = tuple(context.sampling_cfg["max_resolution"]) if context.sampling_cfg.get("max_resolution") else None
    step    = context.sampling_cfg.get("step", 1)

    api_models   = {n: m for n, m in context.models.items() if m.parallel_workers > 1}
    local_models = {n: m for n, m in context.models.items() if m.parallel_workers <= 1}

    def _collect(result):
        scores, jscores, lats, toks = result
        all_scores.extend(scores)
        judge_scores.extend(jscores)
        for k, v in lats.items():
            latencies.setdefault(k, []).extend(v)
        for k, v in toks.items():
            tc = token_counts.setdefault(k, {"prompt": 0, "completion": 0})
            tc["prompt"] += v["prompt"]
            tc["completion"] += v["completion"]

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
                            _run_single_model,
                            name, model, context, max_res, step, progress,
                        ): name
                        for name, model in loaded_api.items()
                    }
                    for future in as_completed(futures):
                        name = futures[future]
                        try:
                            _collect(future.result())
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
            with console.status(f"[bold yellow]Loading [magenta]{model_name}[/magenta]...", spinner="dots"):
                try:
                    model.load()
                except Exception:
                    console.print(f"[bold red]✗ Failed to load {model_name}[/bold red]")
                    console.print(traceback.format_exc(limit=3), style="dim red")
                    continue
            try:
                _collect(_run_single_model(model_name, model, context, max_res, step, progress))
            finally:
                try:
                    model.unload()
                except Exception:
                    pass

    return InferenceResults(all_scores, judge_scores, latencies, token_counts)

def run_scoring(results: InferenceResults) -> list[ModelSummary]:
    """Aggregate per-frame results into model-level performance summaries."""
    console.print("\n[bold green]✓ Benchmark finished. Aggregating scores...[/bold green]")
    return aggregate(results.all_scores, results.latencies, results.token_counts, judge_scores=results.judge_scores)

def save_scores(summaries: list[ModelSummary], context: PipelineContext) -> None:
    """Save consolidated scores to disk."""
    payload = build_scores_payload(summaries, tracking=context.tracking_enabled, mode=context.mode)
    with open(context.results_dir / "raw" / "scores.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    console.print(Panel(f"Results saved in: [bold white]{context.results_dir}[/bold white]", border_style="green"))


def report_results(summaries: list[ModelSummary], context: PipelineContext) -> None:
    """Display the summary table and save consolidated scores to disk."""
    table = Table(title="Extraction Results — Quick Summary", box=box.HEAVY_EDGE, header_style="bold cyan")
    table.add_column("Model",        justify="left",  style="magenta")
    table.add_column("N",            justify="center")
    table.add_column("F1 Context",   justify="right")
    table.add_column("F1 Ped",       justify="right")
    table.add_column("F1 Veh",       justify="right")
    table.add_column("Completeness", justify="right", style="yellow")
    table.add_column("Sem. Rich.",   justify="right", style="yellow")
    table.add_column("Spatial",      justify="right", style="yellow")
    table.add_column("Judge Overall",justify="right", style="bold yellow")
    table.add_column("Latency (s)",  justify="right")

    for s in summaries:
        table.add_row(
            s.model_name, str(s.window_size),
            f"{s.f1_context:.3f}"      if s.f1_context      is not None else "-",
            f"{s.f1_pedestrians:.3f}"  if s.f1_pedestrians  is not None else "-",
            f"{s.f1_vehicles:.3f}"     if s.f1_vehicles      is not None else "-",
            f"{s.avg_judge_completeness:.3f}"      if s.avg_judge_completeness      is not None else "-",
            f"{s.avg_judge_semantic_richness:.3f}" if s.avg_judge_semantic_richness is not None else "-",
            f"{s.avg_judge_spatial_relations:.3f}" if s.avg_judge_spatial_relations is not None else "-",
            f"{s.avg_judge_overall:.3f}"           if s.avg_judge_overall           is not None else "-",
            f"{s.avg_latency_s:.2f}",
        )
    console.print(table)
    save_scores(summaries, context)

def run(
    models_cfg_path: Path,
    clips_cfg_path: Path,
    benchmark_cfg_path: Path,
    selected_models: list[str] | None = None,
    tracking: bool | None = None,
) -> Path:
    """Main pipeline entry point. Orchestrates loading, inference, scoring, and reporting."""
    try:
        context = load_pipeline(models_cfg_path, clips_cfg_path, benchmark_cfg_path, selected_models, mode="extraction")
        if tracking is not None:
            context.tracking_enabled = tracking
            if tracking and not context.fallback_prompt:
                context.fallback_prompt = context.prompt
                context.prompt = _TRACKING_PROMPT_FILE.read_text().strip()
        results   = run_inference(context)
        summaries = run_scoring(results)
        report_results(summaries, context)
        return context.results_dir
    except Exception:
        console.print("[bold red]Critical pipeline failure[/bold red]")
        console.print(traceback.format_exc())
        raise
