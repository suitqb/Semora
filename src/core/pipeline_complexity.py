"""Complexity pipeline — Plan 3 (scene density scaling).

Only measures PDR (Pedestrian Detection Rate) stratified by scene density.
F1 scoring and LLM judge are Plan 1 concerns — not computed here.

Each JSONL record contains n_persons_gt, n_vehicles_gt, n_persons_pred,
n_vehicles_pred so the app can compute PDR = min(pred/gt, 1) per frame.
"""

from __future__ import annotations
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
from ..core.utils import dbg_frame
from ..parsing.output_parser import parse
from ..sampling.complexity_sampler import sample_complexity_windows
from rich.panel import Panel

from .pipeline import (
    load_pipeline,
    PipelineContext, InferenceResults,
)


def _precompute_prompts(context: PipelineContext, all_cw: list) -> dict[str, str]:
    """Run YOLO+ByteTrack detection in chronological order per clip.

    For each clip, all frames are processed sequentially with a persistent
    ByteTrack state (reset between clips). Only detections for target frames
    are retained. This mimics a live video stream: ByteTrack naturally
    accumulates temporal context from every preceding frame without
    pre-scanning future frames.

    Returns {center_frame: resolved_prompt_str}.
    """
    from ..tracking.context_builder import build_detection_context

    if context.live_tracker is None:
        fallback = context.fallback_prompt or context.prompt
        return {cw.window.center_frame: fallback for cw in all_cw}

    # Build clip lookup and per-clip target set
    clip_by_id  = {clip.clip_id: clip for clip in context.clips}
    max_res     = tuple(context.sampling_cfg.get("max_resolution") or [1280, 720])

    # Group target frames by clip
    targets_by_clip: dict[str, set[str]] = {}
    for cw in all_cw:
        targets_by_clip.setdefault(cw.window.clip_id, set()).add(cw.window.center_frame)

    # Per-frame detection cache: {center_frame: [detections]}
    detection_cache: dict[str, list[dict]] = {}

    for clip_id, target_frames in targets_by_clip.items():
        clip = clip_by_id.get(clip_id)
        if clip is None:
            continue

        context.live_tracker.reset()
        console.print(
            f"[cyan]  Clip {clip_id}: scanning {len(clip.frame_names)} frames "
            f"(targets: {len(target_frames)})[/cyan]"
        )

        for frame_name in clip.frame_names:
            img = clip.get_frame(frame_name, max_resolution=max_res)
            # process_frames maintains ByteTrack state via persist=True
            dets = context.live_tracker.process_frames([img])[0]
            if frame_name in target_frames:
                detection_cache[frame_name] = dets

    # Build resolved prompts from cache
    resolved: dict[str, str] = {}
    seen: set[str] = set()
    for cw in all_cw:
        cf = cw.window.center_frame
        if cf in seen:
            continue
        seen.add(cf)

        dets = detection_cache.get(cf, [])
        detection_ctx = build_detection_context(dets)
        if detection_ctx:
            resolved[cf] = context.prompt.replace("{detection_context}", detection_ctx)
        else:
            resolved[cf] = context.fallback_prompt or context.prompt

    return resolved


def _infer_complexity_window(
    model, model_name: str, cw, N: int,
    prompt: str, benchmark_cfg: dict,
    raw_log_path: Path, parsed_log_path: Path, pdr_log_path: Path,
    raw_lock: threading.Lock, parsed_lock: threading.Lock, pdr_lock: threading.Lock,
):
    """Process a single complexity window: infer → parse → write PDR record. Thread-safe."""
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
            **_meta,
        }) + "\n"
        with parsed_lock:
            with open(parsed_log_path, "a") as f:
                f.write(record)

    # PDR: predicted entity counts for the center frame only
    center_win_idx = window.frame_names.index(window.center_frame)
    if parsed.parse_success and center_win_idx < len(parsed.frames):
        fo = parsed.frames[center_win_idx]
        n_persons_pred  = len(fo.pedestrians) if fo.pedestrians  else 0
        n_vehicles_pred = len(fo.vehicles)    if fo.vehicles     else 0
    else:
        n_persons_pred  = 0
        n_vehicles_pred = 0

    pdr_record = json.dumps({
        "model_name":      model_name,
        "window_size":     N,
        "clip_id":         clip_id,
        "center_frame":    window.center_frame,
        "parse_success":   parsed.parse_success,
        "n_persons_pred":  n_persons_pred,
        "n_vehicles_pred": n_vehicles_pred,
        **_meta,
    }) + "\n"
    with pdr_lock:
        with open(pdr_log_path, "a") as f:
            f.write(pdr_record)

    return vlm_out.latency_s, vlm_out.prompt_tokens or 0, vlm_out.completion_tokens or 0


def _run_single_complexity_model(
    model_name: str, model, context: PipelineContext,
    max_res, frames_per_count: int, max_entities, total_frames,
    progress: Progress,
    prompt_map: dict[str, str] | None = None,
) -> tuple[dict, dict]:
    """Run all frame iterations for one model against a shared progress bar."""
    model_lat: dict = {}
    model_tok: dict = {}

    N = 1
    key = (model_name, N)
    model_lat[key] = []
    model_tok[key] = {"prompt": 0, "completion": 0}

    _raw_dir        = context.results_dir / "raw"
    _prefix         = f"{model_name}_"
    raw_log_path    = _raw_dir / f"{_prefix}raw_outputs.jsonl"
    parsed_log_path = _raw_dir / f"{_prefix}parsed_outputs.jsonl"
    pdr_log_path    = _raw_dir / f"{_prefix}frame_scores.jsonl"

    # Pre-build and sort windows by entity count for even difficulty spread
    all_cw = []
    for clip in context.clips:
        all_cw.extend(sample_complexity_windows(
            clip, max_res, frames_per_count, max_entities,
        ))
    all_cw.sort(key=lambda cw: cw.n_entities_gt)
    if total_frames is not None and len(all_cw) > total_frames:
        _step = len(all_cw) / total_frames
        all_cw = [all_cw[round(i * _step)] for i in range(total_frames)]

    workers     = model.parallel_workers
    raw_lock    = threading.Lock()
    parsed_lock = threading.Lock()
    pdr_lock    = threading.Lock()

    _win_args = dict(
        model=model, model_name=model_name, N=N,
        benchmark_cfg=context.benchmark_cfg,
        raw_log_path=raw_log_path, parsed_log_path=parsed_log_path,
        pdr_log_path=pdr_log_path,
        raw_lock=raw_lock, parsed_lock=parsed_lock, pdr_lock=pdr_lock,
    )

    task_id = progress.add_task(
        f"[cyan]{model_name}" + (f" [dim]· {workers}t[/dim]" if workers > 1 else ""),
        total=len(all_cw),
    )

    _pm = prompt_map or {}

    def _resolved_prompt(cw) -> str:
        return _pm.get(cw.window.center_frame, context.prompt)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _infer_complexity_window, cw=cw,
                    prompt=_resolved_prompt(cw), **_win_args,
                ): cw
                for cw in all_cw
            }
            for future in as_completed(futures):
                cw = futures[future]
                try:
                    lat, pt, ct = future.result()
                    model_lat[key].append(lat)
                    model_tok[key]["prompt"]     += pt
                    model_tok[key]["completion"] += ct
                except (KeyError, AttributeError, ImportError):
                    raise
                except Exception:
                    console.print(f"[red]✗ Error on {cw.window.clip_id}/{cw.window.center_frame}[/red]")
                    console.print(traceback.format_exc(), style="dim red")
                progress.update(task_id, advance=1)
    else:
        for cw in all_cw:
            try:
                dbg_frame(model_name, cw.window.clip_id, cw.window.center_frame, N)
                lat, pt, ct = _infer_complexity_window(
                    cw=cw, prompt=_resolved_prompt(cw), **_win_args,
                )
                model_lat[key].append(lat)
                model_tok[key]["prompt"]     += pt
                model_tok[key]["completion"] += ct
            except (KeyError, AttributeError, ImportError):
                raise
            except Exception:
                console.print(f"[red]✗ Error on {cw.window.clip_id}/{cw.window.center_frame}[/red]")
                console.print(traceback.format_exc(), style="dim red")
            progress.update(task_id, advance=1)

    return model_lat, model_tok


def _run_inference(context: PipelineContext) -> InferenceResults:
    latencies:    dict[tuple[str, int], list[float]]    = {}
    token_counts: dict[tuple[str, int], dict[str, int]] = {}

    max_res          = tuple(context.sampling_cfg["max_resolution"]) if context.sampling_cfg.get("max_resolution") else None
    frames_per_count = context.sampling_cfg.get("frames_per_count", 3)
    max_entities     = context.sampling_cfg.get("max_entities")
    total_frames     = context.sampling_cfg.get("total_frames")

    api_models   = {n: m for n, m in context.models.items() if m.parallel_workers > 1}
    local_models = {n: m for n, m in context.models.items() if m.parallel_workers <= 1}

    # Pre-compute YOLO detections once (before any parallel inference)
    prompt_map: dict[str, str] = {}
    if context.tracking_enabled and context.live_tracker is not None:
        # Build full window list to detect on all unique frames
        _all_cw_for_detection = []
        for clip in context.clips:
            _all_cw_for_detection.extend(sample_complexity_windows(
                clip, max_res, frames_per_count, max_entities,
            ))
        if total_frames is not None and len(_all_cw_for_detection) > total_frames:
            _step = len(_all_cw_for_detection) / total_frames
            _all_cw_for_detection = [_all_cw_for_detection[round(i * _step)] for i in range(total_frames)]
        n_unique = len({cw.window.center_frame for cw in _all_cw_for_detection})
        console.print(f"[cyan]Running YOLO detection on {n_unique} unique frames...[/cyan]")
        prompt_map = _precompute_prompts(context, _all_cw_for_detection)
        console.print(f"[green]✓ Detection done — {sum(1 for v in prompt_map.values() if '{detection_context}' not in v)}/{n_unique} frames with detections[/green]")

    def _collect(lats, toks):
        for k, v in lats.items():
            latencies.setdefault(k, []).extend(v)
        for k, v in toks.items():
            tc = token_counts.setdefault(k, {"prompt": 0, "completion": 0})
            tc["prompt"]     += v["prompt"]
            tc["completion"] += v["completion"]

    _model_kwargs = dict(
        context=context, max_res=max_res,
        frames_per_count=frames_per_count, max_entities=max_entities, total_frames=total_frames,
        prompt_map=prompt_map,
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
                            lats, toks = future.result()
                            _collect(lats, toks)
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
                lats, toks = _run_single_complexity_model(
                    model_name, model, progress=progress, **_model_kwargs,
                )
                _collect(lats, toks)
            finally:
                try:
                    model.unload()
                except Exception:
                    pass

    return InferenceResults([], [], latencies, token_counts)


def _save_pdr_scores(context: PipelineContext, results: InferenceResults) -> None:
    """Aggregate PDR per (model, N) from frame_scores.jsonl and write scores.json."""
    import math
    from collections import defaultdict

    raw_dir = context.results_dir / "raw"

    # Read all PDR records written during inference
    records: list[dict] = []
    for path in sorted(raw_dir.glob("*_frame_scores.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    # Aggregate per (model_name, window_size)
    buckets: dict[tuple, dict] = defaultdict(lambda: {
        "pdr_ped": [], "pdr_veh": [], "parse_ok": [], "n": 0,
    })
    for rec in records:
        key = (rec["model_name"], rec["window_size"])
        b = buckets[key]
        n_pg = rec.get("n_persons_gt", 0)
        n_vg = rec.get("n_vehicles_gt", 0)
        n_pp = rec.get("n_persons_pred")
        n_vp = rec.get("n_vehicles_pred")
        if n_pg and n_pp is not None:
            b["pdr_ped"].append(min(n_pp / n_pg, 1.0))
        if n_vg and n_vp is not None:
            b["pdr_veh"].append(min(n_vp / n_vg, 1.0))
        b["parse_ok"].append(1 if rec.get("parse_success") else 0)
        b["n"] += 1

    def _mean(xs): return sum(xs) / len(xs) if xs else None
    def _std(xs):
        if len(xs) < 2: return None
        m = _mean(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

    result_rows = []
    for (model_name, window_size), b in sorted(buckets.items()):
        lat_key = (model_name, window_size)
        lats = results.latencies.get(lat_key, [])
        toks = results.token_counts.get(lat_key, {})
        result_rows.append({
            "model_name":        model_name,
            "window_size":       window_size,
            "n_frames":          b["n"],
            "parse_success_rate": _mean(b["parse_ok"]),
            "mean_pdr_ped":      _mean(b["pdr_ped"]),
            "std_pdr_ped":       _std(b["pdr_ped"]),
            "mean_pdr_veh":      _mean(b["pdr_veh"]),
            "std_pdr_veh":       _std(b["pdr_veh"]),
            "avg_latency_s":     _mean(lats) or 0.0,
            "total_prompt_tokens":     toks.get("prompt", 0),
            "total_completion_tokens": toks.get("completion", 0),
        })

    payload = {
        "meta": {
            "tracking": context.tracking_enabled,
            "mode":     context.mode,
            "run_id":   context.run_id,
        },
        "results": result_rows,
    }
    scores_path = context.results_dir / "raw" / "scores.json"
    with open(scores_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    console.print(Panel(
        f"Results saved in: [bold white]{context.results_dir}[/bold white]",
        border_style="green",
    ))


def run_complexity(
    models_cfg_path: Path,
    clips_cfg_path: Path,
    benchmark_cfg_path: Path,
    selected_models: list[str] | None = None,
    tracking: bool | None = None,
) -> Path:
    """Complexity pipeline entry point (Plan 3)."""
    from .pipeline import _DETECTION_PROMPT_FILE
    try:
        context = load_pipeline(models_cfg_path, clips_cfg_path, benchmark_cfg_path, selected_models, mode="complexity")
        if tracking is not None:
            context.tracking_enabled = tracking
            if tracking and context.live_tracker is None:
                from ..tracking.detector import LiveTracker
                context.live_tracker = LiveTracker()
            if tracking and not context.fallback_prompt:
                context.fallback_prompt = context.prompt
                context.prompt = _DETECTION_PROMPT_FILE.read_text().strip()
        results = _run_inference(context)
        _save_pdr_scores(context, results)
        return context.results_dir
    except Exception:
        console.print("[bold red]Critical pipeline failure[/bold red]")
        console.print(traceback.format_exc())
        raise
