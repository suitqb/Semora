from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ultralytics import YOLO
from ultralytics.utils import LOGGER as _YOLO_LOGGER

# Suppress ultralytics banner and per-call verbose output
_YOLO_LOGGER.setLevel(logging.WARNING)

if TYPE_CHECKING:
    from PIL import Image as PILImage

# Bicycle excluded: TITAN does not annotate bicycles as standard vehicles,
# so injecting them inflates false positives on trunk_open / doors_open.
_KEPT_CLASSES = {"person", "car", "truck", "bus", "motorcycle"}
_CONFIDENCE_THRESHOLD = 0.3


def _parse_boxes(result) -> list[dict]:
    """Extract kept-class detections from a single YOLO result."""
    detections: list[dict] = []
    if result.boxes is None or result.boxes.id is None:
        return detections
    boxes = result.boxes
    for i in range(len(boxes)):
        class_name = result.names[int(boxes.cls[i].item())]
        if class_name not in _KEPT_CLASSES:
            continue
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        detections.append({
            "track_id":   int(boxes.id[i].item()),
            "bbox":       [x1, y1, x2, y2],
            "class_name": class_name,
            "confidence": round(float(boxes.conf[i].item()), 4),
        })
    return detections


class LiveTracker:
    """Run YOLOv8l + ByteTrack on PIL images in real time.

    Usage
    -----
    tracker = LiveTracker()
    for clip in clips:
        tracker.reset()                          # fresh state per clip
        for window in windows:
            detections = tracker.process_frames(window.frames)
            context = build_tracking_context_from_detections(detections)
    """

    def __init__(self, model_name: str = "yolov8l.pt") -> None:
        self.model = YOLO(model_name)

    def reset(self) -> None:
        """Reset ByteTrack state for a new clip."""
        self.model = YOLO(self.model.ckpt_path)

    def process_frames(self, frames: list[PILImage]) -> list[list[dict]]:
        """Run tracking on an ordered list of PIL images.

        Returns one detection list per frame (same order as input).
        Tracker state is maintained across calls — call reset() between clips.
        """
        output: list[list[dict]] = []
        for img in frames:
            results = self.model.track(
                source=img,
                persist=True,
                conf=_CONFIDENCE_THRESHOLD,
                tracker="bytetrack.yaml",
                verbose=False,
            )
            output.append(_parse_boxes(results[0]))
        return output

    def detect_frame(self, frame: PILImage) -> list[dict]:
        """Run plain YOLO detection on a single frame (no tracking state).

        Suitable for single-frame analysis (N=1) where ByteTrack is not needed.
        Returns a flat list of detections with bbox and class_name, no track_id.
        """
        results = self.model.predict(
            source=frame,
            conf=_CONFIDENCE_THRESHOLD,
            verbose=False,
        )
        detections: list[dict] = []
        boxes = results[0].boxes
        if boxes is None:
            return detections
        for i in range(len(boxes)):
            class_name = results[0].names[int(boxes.cls[i].item())]
            if class_name not in _KEPT_CLASSES:
                continue
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            detections.append({
                "bbox":       [x1, y1, x2, y2],
                "class_name": class_name,
                "confidence": round(float(boxes.conf[i].item()), 4),
            })
        return detections
