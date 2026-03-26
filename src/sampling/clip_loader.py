from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class FrameAnnotation:
    """GT for all entities of a frame."""
    frame_name: str
    persons: list[dict]   # fields: track_id, Atomic Actions, Simple Context, ...
    vehicles: list[dict]  # fields: track_id, Motion Status, Trunk Open, Doors Open


@dataclass
class TITANClip:
    clip_id: str
    frame_paths: list[Path]                      # sorted chronologically
    annotations: dict[str, FrameAnnotation]      # frame_name → FrameAnnotation

    @property
    def frame_names(self) -> list[str]:
        return [p.name for p in self.frame_paths]

    def get_frame(self, frame_name: str, max_resolution: tuple[int, int] | None = None) -> Image.Image:
        path = next((p for p in self.frame_paths if p.name == frame_name), None)
        if path is None:
            raise FileNotFoundError(f"{frame_name!r} not found in {self.clip_id}")
        img = Image.open(path).convert("RGB")
        # Resize if necessary (native 2704×1520, default down to 1280×720)
        if max_resolution is not None:
            img.thumbnail(max_resolution, Image.Resampling.LANCZOS) # Use LANCZOS algorithm for resizing
        return img

    def get_frames(
        self,
        frame_names: list[str],
        max_resolution: tuple[int, int] | None = None,
    ) -> list[Image.Image]:
        return [self.get_frame(n, max_resolution) for n in frame_names]


# ---------------------------------------------------------------------------
# GT Columns used for scoring
# ---------------------------------------------------------------------------

_PERSON_COLS = [
    "obj_track_id",
    "attributes.Atomic Actions",
    "attributes.Simple Context",
    "attributes.Complex Contextual",
    "attributes.Communicative",
    "attributes.Transporting",
    "attributes.Age",
]

_VEHICLE_COLS = [
    "obj_track_id",
    "attributes.Motion Status",
    "attributes.Trunk Open",
    "attributes.Doors Open",
]


def _parse_csv(csv_path: Path) -> dict[str, FrameAnnotation]:
    df = pd.read_csv(csv_path)

    annotations: dict[str, FrameAnnotation] = {}
    for frame_name, group in df.groupby("frames"):

        persons_df  = group[group["label"] == "person"]
        vehicles_df = group[group["label"] != "person"]

        def to_dicts(sub: pd.DataFrame, cols: list[str]) -> list[dict]:
            available = [c for c in cols if c in sub.columns]
            return (
                sub[available]
                .rename(columns=lambda c: c.replace("attributes.", ""))
                .to_dict(orient="records")
            )

        annotations[str(frame_name)] = FrameAnnotation(
            frame_name=str(frame_name),
            persons=to_dicts(persons_df, _PERSON_COLS),
            vehicles=to_dicts(vehicles_df, _VEHICLE_COLS),
        )

    return annotations


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_clip(clip_cfg: dict, data_root: Path) -> TITANClip:
    """Load a clip from a clips.yaml entry.

    data_root is always the absolute path from clips.yaml,
    independent of the benchmark project location.
    """
    clip_id   = clip_cfg["clip_id"]
    video_dir = data_root / clip_cfg["video_path"]
    ann_path  = data_root / clip_cfg["annotation_path"]

    if not video_dir.exists():
        raise FileNotFoundError(f"Frames folder not found: {video_dir}")
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation CSV not found: {ann_path}")

    frame_paths = sorted(video_dir.glob("*.png"), key=lambda p: int(p.stem))
    if not frame_paths:
        raise ValueError(f"No .png frame in {video_dir}")

    return TITANClip(
        clip_id=clip_id,
        frame_paths=frame_paths,
        annotations=_parse_csv(ann_path),
    )


def load_all_clips(clips_cfg: dict) -> list[TITANClip]:
    """Load all clips from full clips.yaml."""
    data_root = Path(clips_cfg["data_root"])  # absolute path TITAN
    return [load_clip(cfg, data_root) for cfg in clips_cfg["clips"]]
