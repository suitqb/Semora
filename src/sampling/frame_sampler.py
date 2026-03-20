from __future__ import annotations
from dataclasses import dataclass
from PIL import Image

from .clip_loader import TITANClip, FrameAnnotation


@dataclass
class FrameWindow:
    """Une window = N frames consécutives centrées sur une frame cible."""
    clip_id: str
    center_frame: str           # frame cible (celle scorée contre le GT)
    frame_names: list[str]      # N frames dans l'ordre chronologique
    frames: list[Image.Image]   # images PIL correspondantes
    annotation: FrameAnnotation # GT de la frame cible uniquement
    window_size: int            # N demandé (peut différer en bord de clip)


def _select_indices(
    center_idx: int,
    total: int,
    n: int,
    strategy: str,
) -> list[int]:
    """Retourne les indices des N frames à inclure dans la window.

    Stratégies :
      uniform — N frames espacées régulièrement autour de center_idx
      last    — les N frames précédant center_idx (inclus)
      center  — center_idx au milieu, frames de part et d'autre
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

    else:  # uniform (défaut)
        start = max(0, center_idx - n + 1)
        end   = center_idx
        if end - start + 1 < n:
            # pas assez de frames avant, on prend ce qu'on peut
            indices = list(range(start, end + 1))
        else:
            step = (end - start) / (n - 1)
            indices = [round(start + i * step) for i in range(n)]
            # garantit que center_idx est toujours la dernière frame
            indices[-1] = center_idx

    return indices


def sample_windows(
    clip: TITANClip,
    window_size: int,
    strategy: str = "uniform",
    max_resolution: tuple[int, int] | None = (1280, 720),
    step: int = 1,
) -> list[FrameWindow]:
    """Génère toutes les windows possibles pour un clip.

    Args:
        clip:           TITANClip chargé
        window_size:    N (nombre de frames par window)
        strategy:       uniform | last | center
        max_resolution: resize appliqué au chargement (natif 2704×1520)
        step:           pas entre deux center_frames consécutives

    Returns:
        Liste de FrameWindow — une par frame annotée du clip
    """
    frame_names = clip.frame_names
    total = len(frame_names)
    windows: list[FrameWindow] = []

    # On itère uniquement sur les frames qui ont une annotation GT
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
            window_size=window_size,
        ))

    return windows
