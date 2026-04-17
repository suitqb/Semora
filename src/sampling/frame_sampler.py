from __future__ import annotations
from dataclasses import dataclass
from PIL import Image

from .clip_loader import TITANClip, FrameAnnotation


@dataclass
class FrameWindow:
    """A single frame with its GT annotation."""
    clip_id: str
    center_frame: str
    frame_names: list[str]
    frames: list[Image.Image]
    annotation: FrameAnnotation
    annotations: list[FrameAnnotation | None]


def sample_windows(
    clip: TITANClip,
    max_resolution: tuple[int, int] | None = (1280, 720),
    step: int = 1,
) -> list[FrameWindow]:
    """Return one FrameWindow per annotated frame (single-frame, no window context).

    Args:
        clip:           loaded TITANClip
        max_resolution: resize applied upon loading (native 2704×1520)
        step:           step between consecutive sampled frames

    Returns:
        List of FrameWindow — one per annotated frame of the clip
    """
    annotated = [fn for fn in clip.frame_names if fn in clip.annotations]

    windows: list[FrameWindow] = []
    for center_name in annotated[::step]:
        windows.append(FrameWindow(
            clip_id=clip.clip_id,
            center_frame=center_name,
            frame_names=[center_name],
            frames=clip.get_frames([center_name], max_resolution),
            annotation=clip.annotations[center_name],
            annotations=[clip.annotations[center_name]],
        ))

    return windows
