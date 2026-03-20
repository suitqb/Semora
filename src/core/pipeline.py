from __future__ import annotations
import json
import traceback
from datetime import datetime
from pathlib import Path

import yaml

from ..models.base import BaseVLM
from ..models.registry import build_models
from ..sampling.clip_loader import load_all_clips
from ..sampling.frame_sampler import sample_windows
from ..parsing.output_parser import parse
from ..scoring.titan_scorer import score_frame
from ..scoring.aggregator import aggregate


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _log(msg: str) -> None:
    print(msg, flush=True)


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
    _log(f"\n{'='*60}")
    _log(f"  Run : {run_id}")
    _log(f"  Clips : {len(clips)}  |  Modèles : {len(models_cfg)}  |  Windows : {window_sizes}")
    _log(f"{'='*60}\n")

    models: dict[str, BaseVLM] = build_models(models_cfg)

    all_scores   = []
    latencies    = {}
    token_counts = {}

    for model_name, model in models.items():
        _log(f"┌─ [{model_name}] chargement du modèle...")

        try:
            model.load()
            _log("│  ✓ modèle chargé")
        except Exception:
            _log("│  ✗ échec du chargement — modèle ignoré")
            _log(f"│  {traceback.format_exc(limit=3)}")
            _log(f"└─ skip {model_name}\n")
            continue

        for N in window_sizes:
            _log("│")
            _log(f"├─ window N={N}")
            key = (model_name, N)
            latencies[key]    = []
            token_counts[key] = {"prompt": 0, "completion": 0}

            n_ok = 0
            n_fail = 0

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
                        n_ok += 1

                    except Exception:
                        n_fail += 1
                        _log(f"│    ✗ erreur sur {clip.clip_id}/{window.center_frame}")
                        _log(f"│    {traceback.format_exc(limit=2)}")

            _log(f"│  ✓ {n_ok} frames traitées  |  ✗ {n_fail} échecs")

        try:
            model.unload()
        except Exception:
            pass

        _log(f"└─ {model_name} terminé\n")

    _log(f"{'='*60}")
    _log("  Agrégation des scores...")

    summaries = aggregate(all_scores, latencies, token_counts)
    scores_path = results_dir / "scores.json"
    with open(scores_path, "w") as f:
        json.dump([s.__dict__ for s in summaries], f, indent=2)

    _log(f"  Résultats → {results_dir}")
    _log(f"{'='*60}\n")

    return results_dir
