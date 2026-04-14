from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


_KEPT_CLASSES = {"person", "car", "truck", "bus", "motorcycle", "bicycle"}
_CONFIDENCE_THRESHOLD = 0.3


class TrackingPrecomputer:
    """Run YOLOv8l + ByteTrack on an ordered list of frame paths.

    The tracker state is reset for each new clip via a fresh model instance
    (or by calling reset() between clips).
    """

    def __init__(self, model_name: str = "yolov8l.pt") -> None:
        self.model = YOLO(model_name)

    def reset(self) -> None:
        """Reset ByteTrack state so a new clip starts with a clean tracker."""
        # Reinitialising the model is the safest way to flush ByteTrack state.
        self.model = YOLO(self.model.ckpt_path)

    def process_clip(self, frame_paths: list[Path]) -> dict[str, list[dict]]:
        """Run tracking on a single clip.

        Parameters
        ----------
        frame_paths:
            Image paths in chronological order.

        Returns
        -------
        dict mapping frame_id (stem of the image filename, e.g. "00001")
        to a list of detection dicts::

            {
                "track_id":   int,
                "bbox":       [x1, y1, x2, y2],   # absolute pixel coords
                "class_name": str,
                "confidence": float,
            }

        Only detections whose class_name is in the kept-classes set and whose
        confidence is >= 0.3 are included. Frames with no qualifying detections
        are stored as empty lists.
        """
        output: dict[str, list[dict]] = {}

        for path in frame_paths:
            frame_id = path.stem

            results = self.model.track(
                source=str(path),
                persist=True,           # keeps ByteTrack state between frames
                conf=_CONFIDENCE_THRESHOLD,
                tracker="bytetrack.yaml",
                verbose=False,
            )

            detections: list[dict] = []
            result = results[0]  # single image → single result

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes
                for i in range(len(boxes)):
                    class_name = result.names[int(boxes.cls[i].item())]
                    if class_name not in _KEPT_CLASSES:
                        continue

                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    detections.append(
                        {
                            "track_id":   int(boxes.id[i].item()),
                            "bbox":       [x1, y1, x2, y2],
                            "class_name": class_name,
                            "confidence": round(float(boxes.conf[i].item()), 4),
                        }
                    )

            output[frame_id] = detections

        return output
