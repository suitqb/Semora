"""Complexity sampler — selects frames by exact GT entity count for Plan 3."""

from __future__ import annotations
import math
from dataclasses import dataclass

from .clip_loader import TITANClip
from .frame_sampler import FrameWindow, _select_indices


@dataclass
class ComplexityWindow:
    """A FrameWindow tagged with the exact GT entity counts of its center frame."""
    window: FrameWindow
    n_persons_gt: int
    n_vehicles_gt: int
    n_entities_gt: int


def sample_complexity_windows(
    clip: TITANClip,
    window_size: int,
    strategy: str = "uniform",
    max_resolution: tuple[int, int] | None = (1280, 720),
    frames_per_count: int = 3,
    max_entities: int | None = None,
) -> list[ComplexityWindow]:
    """Sample frames grouped by exact GT entity count (persons + vehicles).

    For each distinct entity count found in the clip, up to `frames_per_count`
    frames are selected (evenly spaced within that group). Frames with zero
    annotated entities are ignored.

    Args:
        clip:             loaded TITANClip
        window_size:      N frames per window
        strategy:         window frame selection strategy (uniform | last | center)
        max_resolution:   resize cap applied when loading images
        frames_per_count: max frames sampled per distinct entity count value
        max_entities:     optional upper cap on entity count (None = no limit)

    Returns:
        List of ComplexityWindow sorted by n_entities_gt ascending (easy → hard).
    """
    frame_names = clip.frame_names
    total = len(frame_names)

    # Group annotated frames by exact entity count
    count_frames: dict[int, list[str]] = {}
    frame_meta: dict[str, tuple[int, int]] = {}  # frame_name → (n_persons, n_vehicles)

    for frame_name in frame_names:
        ann = clip.annotations.get(frame_name)
        if ann is None:
            continue
        n_persons  = len(ann.persons)
        n_vehicles = len(ann.vehicles)
        n_entities = n_persons + n_vehicles
        if n_entities == 0:
            continue
        if max_entities is not None and n_entities > max_entities:
            continue
        frame_meta[frame_name] = (n_persons, n_vehicles)
        count_frames.setdefault(n_entities, []).append(frame_name)

    # Sample frames_per_count frames per count (evenly spaced)
    selected: list[tuple[str, int]] = []  # (frame_name, n_entities)
    for n_entities, frames in count_frames.items():
        if len(frames) <= frames_per_count:
            chosen = frames
        else:
            step = len(frames) / frames_per_count
            chosen = [frames[round(i * step)] for i in range(frames_per_count)]
        for fn in chosen:
            selected.append((fn, n_entities))

    # Build ComplexityWindow objects, sorted easy → hard
    selected.sort(key=lambda x: x[1])

    result: list[ComplexityWindow] = []
    for center_name, n_entities in selected:
        center_idx = frame_names.index(center_name)
        indices    = _select_indices(center_idx, total, window_size, strategy)
        sel_names  = [frame_names[i] for i in indices]

        fw = FrameWindow(
            clip_id=clip.clip_id,
            center_frame=center_name,
            frame_names=sel_names,
            frames=clip.get_frames(sel_names, max_resolution),
            annotation=clip.annotations[center_name],
            annotations=[clip.annotations.get(fn) for fn in sel_names],
            window_size=window_size,
        )
        n_persons, n_vehicles = frame_meta[center_name]
        result.append(ComplexityWindow(
            window=fw,
            n_persons_gt=n_persons,
            n_vehicles_gt=n_vehicles,
            n_entities_gt=n_entities,
        ))

    return result
