from __future__ import annotations

import json
from pathlib import Path


_HEADER = (
    "PRE-COMPUTED TRACKING CONTEXT:\n"
    "The following entities were detected and tracked by an external vision system.\n"
    "Use the provided track_id values exactly — do not invent or modify them."
)


def build_tracking_context(
    clip_id: str,
    frame_ids: list[str],
    tracking_dir: str,
) -> str:
    """Build a tracking context block for injection into a VLM prompt.

    Parameters
    ----------
    clip_id:
        Clip identifier, e.g. "clip_1".
    frame_ids:
        Ordered list of frame IDs in the current window, matching the stems
        stored in the tracking JSON (e.g. ["00001", "00016"]).
    tracking_dir:
        Path to the directory containing per-clip tracking JSON files
        (e.g. "data/titan/tracking").

    Returns
    -------
    Formatted string block ready for prompt injection, or "" if the tracking
    file does not exist for this clip.
    """
    tracking_path = Path(tracking_dir) / f"{clip_id}.json"
    if not tracking_path.exists():
        return ""

    with open(tracking_path) as f:
        tracking_data: dict[str, list[dict]] = json.load(f)

    lines: list[str] = [_HEADER, ""]

    for k, frame_id in enumerate(frame_ids, start=1):
        detections = tracking_data.get(frame_id, [])

        if not detections:
            lines.append(f"Frame {k}: (no detections)")
            continue

        parts: list[str] = []
        for det in detections:
            label = "Pedestrian" if det["class_name"] == "person" else "Vehicle"
            x1, y1, x2, y2 = (round(v) for v in det["bbox"])
            parts.append(f"{label} #{det['track_id']} [bbox: {x1},{y1},{x2},{y2}]")

        lines.append(f"Frame {k}: {', '.join(parts)}")

    return "\n".join(lines)
