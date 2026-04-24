from __future__ import annotations

import json
import re
import time
from PIL import Image

from .base import VLMOutput
from .molmo import Molmo

_LOCALIZE_PROMPT = (
    "Point to each pedestrian visible in this driving scene. "
    "Point to each vehicle visible in this driving scene. "
    "Label each point clearly as 'Pedestrian' or 'Vehicle'."
)

_PED_PROMPT = """\
This crop shows a single pedestrian from a driving scene.
Return ONLY a valid JSON object with exactly these fields:
{
  "atomic_action": "<one of: walking, running, standing, sitting, bending, squatting, laying down, jumping, none of the above>",
  "simple_context": "<one of: walking along the side of the road, walking on the road, waiting to cross street, crossing a street at pedestrian crossing, jaywalking (illegally crossing NOT at pedestrian crossing), entering a building, exiting a building, biking, motorcycling, none of the above>",
  "communicative": "<one of: none of the above, looking into phone, talking on phone, talking in group>",
  "transporting": "<one of: none of the above, carrying with both hands, pushing, pulling>",
  "age": "<one of: child, adult, senior over 65>"
}"""

_VEH_PROMPT = """\
This crop shows a single vehicle from a driving scene.
Return ONLY a valid JSON object with exactly these fields:
{
  "type": "<one of: car, truck, bus, motorcycle, bicycle, other>",
  "motion_status": "<one of: stopped, moving, parked>",
  "trunk_open": "<one of: open, closed>",
  "doors_open": "<one of: open, closed>"
}"""

_POINT_RE = re.compile(
    r'(pedestrian|vehicle)[^<]{0,80}<point[^>]*x=["\']?([0-9.]+)["\']?[^>]*y=["\']?([0-9.]+)["\']?',
    re.IGNORECASE,
)

_TRACK_DIST_THRESHOLD = 100  # pixels — max distance to consider same entity across frames


def _parse_entities(text: str, img_w: int, img_h: int) -> list[tuple[str, int, int]]:
    """Extract (entity_type, px, py) from Molmo pointing output. Coords are 0-100 percent."""
    entities = []
    for m in _POINT_RE.finditer(text):
        etype = m.group(1).lower()
        px = int(float(m.group(2)) / 100.0 * img_w)
        py = int(float(m.group(3)) / 100.0 * img_h)
        entities.append((etype, px, py))
    return entities


def _assign_track_ids(
    per_frame: list[list[tuple[str, int, int]]],
) -> list[list[tuple[str, int, int, int]]]:
    """
    Greedy nearest-neighbor matching across frames to assign stable track_ids.
    Returns per-frame lists of (etype, px, py, track_id).
    """
    next_id = 1
    prev: list[tuple[str, int, int, int]] = []
    result = []

    for entities in per_frame:
        if not prev:
            tracked = [(e, x, y, next_id + i) for i, (e, x, y) in enumerate(entities)]
            next_id += len(entities)
        else:
            used: set[int] = set()
            tracked = []
            for etype, px, py in entities:
                best_id, best_dist = None, _TRACK_DIST_THRESHOLD
                for pe, ppx, ppy, pid in prev:
                    if pid in used or pe != etype:
                        continue
                    d = ((px - ppx) ** 2 + (py - ppy) ** 2) ** 0.5
                    if d < best_dist:
                        best_dist, best_id = d, pid
                if best_id is not None:
                    used.add(best_id)
                    tracked.append((etype, px, py, best_id))
                else:
                    tracked.append((etype, px, py, next_id))
                    next_id += 1
        prev = tracked
        result.append(tracked)

    return result


def _make_crop(img: Image.Image, px: int, py: int) -> Image.Image:
    W, H = img.size
    pad_x = max(int(W * 0.15), 100)
    pad_y = max(int(H * 0.20), 120)
    return img.crop((
        max(0, px - pad_x), max(0, py - pad_y),
        min(W, px + pad_x), min(H, py + pad_y),
    ))


def _extract_json(text: str) -> dict:
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}


class MolmoPointing(Molmo):
    """
    Two-stage Molmo inference:
      1. Point to all pedestrians and vehicles in the scene.
      2. For each pointed entity, extract a crop and query attributes.
    Falls back to standard (parent) extraction if pointing produces no entities.
    """

    def infer(self, frames: list[Image.Image], prompt: str) -> VLMOutput:
        assert self._loaded
        t0 = time.perf_counter()

        W, H = frames[-1].size
        total_prompt_tokens = 0
        total_completion_tokens = 0

        # ── Stage 1: localize entities on each frame → assign stable track_ids ─
        per_frame_raw: list[list[tuple[str, int, int]]] = []
        for frame in frames:
            locate_text, pt, ct = self._generate([frame], _LOCALIZE_PROMPT)
            total_prompt_tokens += pt
            total_completion_tokens += ct
            per_frame_raw.append(_parse_entities(locate_text, W, H))

        tracked_per_frame = _assign_track_ids(per_frame_raw)
        # Use last frame's entities for attribute extraction
        last_frame_entities = tracked_per_frame[-1]

        # ── Fallback: no pointing output → standard extraction ────────────────
        if not last_frame_entities:
            std_text, pt, ct = self._generate(frames, prompt)
            latency = time.perf_counter() - t0
            return VLMOutput(
                model_name=self.name,
                clip_id="", center_frame="", frame_names=[],
                window_size=len(frames),
                raw_text=std_text,
                latency_s=round(latency, 3),
                prompt_tokens=total_prompt_tokens + pt,
                completion_tokens=total_completion_tokens + ct,
            )

        # ── Stage 2: per-entity attribute extraction ──────────────────────────
        pedestrians: list[dict] = []
        vehicles: list[dict] = []

        for etype, px, py, track_id in last_frame_entities:
            crop = _make_crop(frames[-1], px, py)
            entity_prompt = _PED_PROMPT if etype == "pedestrian" else _VEH_PROMPT
            attr_text, pt, ct = self._generate([crop], entity_prompt)
            total_prompt_tokens += pt
            total_completion_tokens += ct

            attrs = _extract_json(attr_text)
            attrs["track_id"] = track_id
            attrs["spatial_relations"] = []

            if etype == "pedestrian":
                pedestrians.append(attrs)
            else:
                vehicles.append(attrs)

        latency = time.perf_counter() - t0

        raw_text = json.dumps({
            "frame_1": {
                "scene_context": {
                    "location": "road",
                    "description": (
                        f"Pointing mode: {len(pedestrians)} pedestrian(s), "
                        f"{len(vehicles)} vehicle(s) detected."
                    ),
                },
                "pedestrians": pedestrians,
                "vehicles": vehicles,
            }
        })

        return VLMOutput(
            model_name=self.name,
            clip_id="", center_frame="", frame_names=[],
            window_size=len(frames),
            raw_text=raw_text,
            latency_s=round(latency, 3),
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
        )
