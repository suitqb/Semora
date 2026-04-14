from __future__ import annotations


_HEADER = (
    "TRACKING CONTEXT:\n"
    "The following entities were detected by an external vision system. "
    "Use their track_id as a stable reference if you observe them in the frames. "
    "Only describe entities that are clearly visible — do not fabricate attributes "
    "for entities you cannot see."
)

_DETECTION_HEADER = (
    "PRE-DETECTION HINT (YOLO — may miss occluded or distant entities):\n"
    "The following entities were detected in this frame. "
    "Use this as a starting point for your scan, but look carefully for additional entities "
    "that may be partially occluded, far away, or missed by the detector."
)


def _format_detections(detections: list[dict]) -> str:
    parts: list[str] = []
    for det in detections:
        label = "Pedestrian" if det["class_name"] == "person" else "Vehicle"
        x1, y1, x2, y2 = (round(v) for v in det["bbox"])
        parts.append(f"{label} #{det['track_id']} [bbox: {x1},{y1},{x2},{y2}]")
    return ", ".join(parts) if parts else "(no detections)"


def build_detection_context(detections: list[dict]) -> str:
    """Build a detection context string for a single frame (no tracking state).

    Parameters
    ----------
    detections:
        Flat list of detections as returned by LiveTracker.detect_frame().

    Returns
    -------
    Formatted string ready for {detection_context} injection in the prompt,
    or "" if no detections.
    """
    if not detections:
        return ""

    n_peds = sum(1 for d in detections if d["class_name"] == "person")
    n_vehs = len(detections) - n_peds

    summary_parts = []
    if n_peds:
        summary_parts.append(f"{n_peds} pedestrian{'s' if n_peds > 1 else ''}")
    if n_vehs:
        summary_parts.append(f"{n_vehs} vehicle{'s' if n_vehs > 1 else ''}")
    summary = ", ".join(summary_parts)

    lines = [_DETECTION_HEADER, f"Detected: {summary}", ""]
    for det in detections:
        label = "Pedestrian" if det["class_name"] == "person" else f"Vehicle ({det['class_name']})"
        x1, y1, x2, y2 = (round(v) for v in det["bbox"])
        lines.append(f"  • {label} [bbox: {x1},{y1},{x2},{y2}]")

    return "\n".join(lines)


def build_tracking_context_from_detections(
    frame_detections: list[list[dict]],
) -> str:
    """Build a tracking context string from live detection results.

    Parameters
    ----------
    frame_detections:
        Ordered list of detection lists — one list per frame in the window,
        as returned by LiveTracker.process_frames().

    Returns
    -------
    Formatted string ready for {tracking_context} injection in the prompt,
    or "" if no detections at all.
    """
    if not frame_detections or not any(frame_detections):
        return ""

    lines: list[str] = [_HEADER, ""]
    for k, detections in enumerate(frame_detections, start=1):
        lines.append(f"Frame {k}: {_format_detections(detections)}")

    return "\n".join(lines)
