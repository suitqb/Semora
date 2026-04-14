"""Offline YOLO+ByteTrack pre-computation for all clips.

Usage:
    python scripts/precompute_tracking.py

Reads clip list from configs/clips_extraction.yaml.
Writes one JSON file per clip to data/titan/tracking/{clip_id}.json.
Clips whose output file already exists are skipped (cache behaviour).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

# Allow imports from the project root regardless of where the script is run.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.tracking.detector import TrackingPrecomputer  # noqa: E402


_CLIPS_CFG = ROOT / "configs" / "clips_extraction.yaml"


def main() -> None:
    with open(_CLIPS_CFG) as f:
        cfg = yaml.safe_load(f)

    data_root = ROOT / cfg["data_root"]
    tracking_dir = data_root / "tracking"
    tracking_dir.mkdir(parents=True, exist_ok=True)

    clips = cfg["clips"]
    precomputer = TrackingPrecomputer()

    for i, clip_cfg in enumerate(clips, start=1):
        clip_id = clip_cfg["clip_id"]
        out_path = tracking_dir / f"{clip_id}.json"

        if out_path.exists():
            print(f"[{i}/{len(clips)}] {clip_id} — already cached, skipping.")
            continue

        video_dir = data_root / clip_cfg["video_path"]
        frame_paths = sorted(video_dir.glob("*.png"), key=lambda p: int(p.stem))

        if not frame_paths:
            print(f"[{i}/{len(clips)}] {clip_id} — no frames found, skipping.")
            continue

        print(f"[{i}/{len(clips)}] {clip_id} — {len(frame_paths)} frames ...", end=" ", flush=True)

        # Reset tracker state between clips.
        precomputer.reset()
        tracking_data = precomputer.process_clip(frame_paths)

        total_entities = sum(len(v) for v in tracking_data.values())
        unique_tracks = len({d["track_id"] for dets in tracking_data.values() for d in dets})

        with open(out_path, "w") as f:
            json.dump(tracking_data, f)

        print(f"done. {total_entities} detections, {unique_tracks} unique tracks → {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
