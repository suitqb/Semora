from __future__ import annotations
from dataclasses import dataclass
from PIL import Image

from .clip_loader import TITANClip, FrameAnnotation


@dataclass
class FrameWindow:
    """A window = N consecutive frames centered on a target frame."""
    clip_id: str
    center_frame: str                        # target frame (reference for judge)
    frame_names: list[str]                   # N frames in chronological order
    frames: list[Image.Image]                # corresponding PIL images
    annotation: FrameAnnotation              # GT for the center frame (used by judge)
    annotations: list[FrameAnnotation | None]  # GT for each frame in the window (None if no GT)
    window_size: int                         # requested N (may differ at clip edges)


def _select_indices(
    center_idx: int,
    total: int,
    n: int,
    strategy: str,
) -> list[int]:
    """Returns the indices of the N frames to include in the window.

    Strategies:
      uniform — N frames spaced regularly around center_idx
      last    — the N frames preceding center_idx (inclusive)
      center  — center_idx in the middle, frames on either side
    """
    if n == 1:
        return [center_idx]

    if strategy == "last":
        start = max(0, center_idx - n + 1)
        indices = list(range(start, center_idx + 1))

    elif strategy == "center":
        half = n // 2
        start = max(0, center_idx - half)
        end   = min(total - 1, start + n - 1)
        start = max(0, end - n + 1)
        indices = list(range(start, end + 1))

    else:  # uniform (default)
        start = max(0, center_idx - n + 1)
        end   = center_idx
        if end - start + 1 < n:
            # not enough frames before, take what we can
            indices = list(range(start, end + 1))
        else:
            step = (end - start) / (n - 1)
            indices = [round(start + i * step) for i in range(n)]
            # guarantees that center_idx is always the last frame
            indices[-1] = center_idx

    return indices


def sample_windows(
    clip: TITANClip,
    window_size: int,
    strategy: str = "uniform",
    max_resolution: tuple[int, int] | None = (1280, 720),
    step: int = 1,
) -> list[FrameWindow]:
    """Generates all possible windows for a clip.

    Args:
        clip:           loaded TITANClip
        window_size:    N (number of frames per window)
        strategy:       uniform | last | center
        max_resolution: resize applied upon loading (native 2704×1520)
        step:           step between two consecutive center_frames

    Returns:
        List of FrameWindow — one per annotated frame of the clip
    """
    frame_names = clip.frame_names
    total = len(frame_names)
    windows: list[FrameWindow] = []

    # Iterate only on frames that have a GT annotation
    annotated = [fn for fn in frame_names if fn in clip.annotations]

    for center_name in annotated[::step]:
        center_idx = frame_names.index(center_name)
        indices    = _select_indices(center_idx, total, window_size, strategy)
        selected   = [frame_names[i] for i in indices]

        windows.append(FrameWindow(
            clip_id=clip.clip_id,
            center_frame=center_name,
            frame_names=selected,
            frames=clip.get_frames(selected, max_resolution),
            annotation=clip.annotations[center_name],
            annotations=[clip.annotations.get(fn) for fn in selected],
            window_size=window_size,
        ))

    return windows
