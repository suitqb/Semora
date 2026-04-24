from __future__ import annotations
from PIL import Image


def build_crops(
    image: Image.Image,
    detections: list[dict],
    padding: float = 0.3,
    max_size: int = 640,
) -> list[tuple[Image.Image, dict]]:
    """Generate padded crops for each detected entity.

    Returns list of (crop_image, detection) pairs.
    Skips degenerate bboxes (zero area after clamping).
    """
    w, h = image.size
    results: list[tuple[Image.Image, dict]] = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        bw, bh = x2 - x1, y2 - y1
        px, py = bw * padding, bh * padding
        cx1 = max(0.0, x1 - px)
        cy1 = max(0.0, y1 - py)
        cx2 = min(float(w), x2 + px)
        cy2 = min(float(h), y2 + py)
        if cx2 - cx1 < 2 or cy2 - cy1 < 2:
            continue
        crop = image.crop((cx1, cy1, cx2, cy2))
        crop.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        results.append((crop, det))
    return results
