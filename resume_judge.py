#!/usr/bin/env python3
"""Resume incomplete LLM judge scoring for a run directory.

Usage:
    python resume_judge.py <run_dir> [--dry-run]

Example:
    python resume_judge.py runs/extraction/extraction_baseline_1280x720_molmo2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent))

from src.parsing.output_parser import ParsedOutput, FrameOutput
from src.sampling.clip_loader import FrameAnnotation, _parse_csv
from src.scoring.llm_judge import judge, JudgeScore

DATA_ROOT = Path(__file__).parent / "data" / "titan"
BENCH_CFG_PATH = Path(__file__).parent / "configs" / "benchmark.yaml"


def load_benchmark_cfg() -> dict:
    import yaml
    with open(BENCH_CFG_PATH) as f:
        return yaml.safe_load(f)


def reconstruct_parsed(d: dict) -> ParsedOutput:
    frames = [
        FrameOutput(
            scene_context=f.get("scene_context", {}),
            pedestrians=f.get("pedestrians", []),
            vehicles=f.get("vehicles", []),
        )
        for f in d.get("frames", [])
    ]
    return ParsedOutput(
        frames=frames,
        parse_success=d["parse_success"],
        parse_error=d.get("parse_error"),
    )


def load_annotation_cache(clip_ids: set[str]) -> dict[str, dict[str, FrameAnnotation]]:
    """Load CSV annotations for each clip, keyed by clip_id → frame_name."""
    ann_root = DATA_ROOT / "annotations"
    cache: dict[str, dict[str, FrameAnnotation]] = {}
    for clip_id in clip_ids:
        csv_path = ann_root / f"{clip_id}.csv"
        if not csv_path.exists():
            print(f"[WARN] Missing annotation CSV: {csv_path}")
            cache[clip_id] = {}
        else:
            cache[clip_id] = _parse_csv(csv_path)
    return cache


def resume_run(run_dir: Path, dry_run: bool = False) -> None:
    raw_dir = run_dir / "raw"

    # Find the model prefix
    parsed_files = list(raw_dir.glob("*_parsed_outputs.jsonl"))
    if not parsed_files:
        print(f"[ERROR] No parsed_outputs.jsonl found in {raw_dir}")
        sys.exit(1)

    bench_cfg = load_benchmark_cfg()
    judge_cfg = bench_cfg.get("benchmark", {}).get("scorers", {}).get("llm_judge")
    if not judge_cfg or not judge_cfg.get("enabled"):
        print("[ERROR] LLM judge not enabled in benchmark.yaml")
        sys.exit(1)

    for parsed_path in parsed_files:
        prefix = parsed_path.name.replace("_parsed_outputs.jsonl", "")
        judge_path = raw_dir / f"{prefix}_judge_outputs.jsonl"

        # Load already-done frames
        done: set[tuple[str, str]] = set()
        if judge_path.exists():
            with open(judge_path) as f:
                for line in f:
                    d = json.loads(line)
                    done.add((d["clip_id"], d["center_frame"]))

        # Load all parsed entries
        parsed_entries: list[dict] = []
        with open(parsed_path) as f:
            for line in f:
                parsed_entries.append(json.loads(line))

        missing = [e for e in parsed_entries if (e["clip_id"], e["center_frame"]) not in done]
        total = len(parsed_entries)
        print(f"\n[{prefix}] {len(done)}/{total} already judged, {len(missing)} remaining")

        if not missing or dry_run:
            if dry_run:
                print("  [dry-run] skipping judge calls")
            continue

        # Load annotations for required clips
        clip_ids = {e["clip_id"] for e in missing}
        ann_cache = load_annotation_cache(clip_ids)

        # Run judge on missing frames and append
        with open(judge_path, "a") as f_judge:
            for i, entry in enumerate(missing, 1):
                clip_id = entry["clip_id"]
                center_frame = entry["center_frame"]
                N = entry["N"]
                model_name = entry["model"]

                parsed_obj = reconstruct_parsed(entry["parsed"])

                annotation = ann_cache.get(clip_id, {}).get(center_frame)
                if annotation is None:
                    # Create empty annotation so judge still runs
                    annotation = FrameAnnotation(
                        frame_name=center_frame, persons=[], vehicles=[]
                    )

                print(f"  [{i}/{len(missing)}] Judging {clip_id}/{center_frame} ... ", end="", flush=True)
                try:
                    score = judge(
                        parsed=parsed_obj,
                        annotation=annotation,
                        model_name=model_name,
                        clip_id=clip_id,
                        center_frame=center_frame,
                        window_size=N,
                        judge_cfg=judge_cfg,
                    )
                    record = json.dumps({
                        "model": model_name,
                        "N": N,
                        "clip_id": clip_id,
                        "center_frame": center_frame,
                        "judge_model": judge_cfg.get("model_id"),
                        "scores": score.__dict__,
                    }) + "\n"
                    f_judge.write(record)
                    f_judge.flush()

                    status = score.judge_error or f"ok ({score.overall:.2f})"
                    print(status)
                except Exception as e:
                    print(f"ERROR: {e}")

        # Recompute judge averages and patch scores.json
        _update_scores_json(raw_dir, prefix, judge_path)


def _update_scores_json(raw_dir: Path, prefix: str, judge_path: Path) -> None:
    scores_path = raw_dir / "scores.json"
    if not scores_path.exists():
        print(f"  [WARN] scores.json not found, skipping update")
        return

    with open(scores_path) as f:
        payload = json.load(f)

    # Load all judge entries for this model prefix
    judge_entries: list[dict] = []
    with open(judge_path) as f:
        for line in f:
            judge_entries.append(json.loads(line))

    # Group by (model, N)
    from collections import defaultdict
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for e in judge_entries:
        groups[(e["model"], e["N"])].append(e)

    # Patch each result entry in scores.json
    for result in payload.get("results", []):
        model_name = result["model_name"]
        window_size = result["window_size"]
        key = (model_name, window_size)

        j_group = groups.get(key, [])
        valid = [
            e for e in j_group
            if e.get("scores", {}).get("judge_error") is None
        ]
        if valid:
            result["avg_judge_completeness"]    = sum(e["scores"]["completeness"]    for e in valid) / len(valid)
            result["avg_judge_semantic_richness"] = sum(e["scores"]["semantic_richness"] for e in valid) / len(valid)
            result["avg_judge_spatial_relations"] = sum(e["scores"]["spatial_relations"] for e in valid) / len(valid)
            result["avg_judge_overall"]          = sum(e["scores"]["overall"]          for e in valid) / len(valid)
            print(f"\n  scores.json updated: judge_overall={result['avg_judge_overall']:.4f} (over {len(valid)} valid frames)")
        else:
            print(f"\n  [WARN] No valid judge entries for ({model_name}, N={window_size})")

    with open(scores_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  scores.json saved → {scores_path}")


def main():
    parser = argparse.ArgumentParser(description="Resume incomplete LLM judge scoring")
    parser.add_argument("run_dir", type=Path, help="Run directory (e.g. runs/extraction/extraction_baseline_1280x720_molmo2)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without calling the judge")
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir.is_absolute():
        run_dir = Path(__file__).parent / run_dir

    if not run_dir.exists():
        print(f"[ERROR] Directory not found: {run_dir}")
        sys.exit(1)

    resume_run(run_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
