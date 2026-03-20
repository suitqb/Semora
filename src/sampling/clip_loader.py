from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image


# ---------------------------------------------------------------------------
# Structures de données
# ---------------------------------------------------------------------------

@dataclass
class FrameAnnotation:
    """GT pour toutes les entités d'une frame."""
    frame_name: str
    persons: list[dict]   # champs: track_id, Atomic Actions, Simple Context, ...
    vehicles: list[dict]  # champs: track_id, Motion Status, Trunk Open, Doors Open


@dataclass
class TITANClip:
    clip_id: str
    frame_paths: list[Path]                      # triées chronologiquement
    annotations: dict[str, FrameAnnotation]      # frame_name → FrameAnnotation

    @property
    def frame_names(self) -> list[str]:
        return [p.name for p in self.frame_paths]

    def get_frame(self, frame_name: str, max_resolution: tuple[int, int] | None = None) -> Image.Image:
        path = next((p for p in self.frame_paths if p.name == frame_name), None)
        if path is None:
            raise FileNotFoundError(f"{frame_name!r} introuvable dans {self.clip_id}")
        img = Image.open(path).convert("RGB")
        # Resize si nécessaire (natif 2704×1520, on descend à 1280×720 par défaut)
        if max_resolution is not None:
            img.thumbnail(max_resolution, Image.Resampling.LANCZOS) #Utiliser l'algo LANCZOS pour le resize
        return img

    def get_frames(
        self,
        frame_names: list[str],
        max_resolution: tuple[int, int] | None = None,
    ) -> list[Image.Image]:
        return [self.get_frame(n, max_resolution) for n in frame_names]


# ---------------------------------------------------------------------------
# Colonnes GT utilisées pour le scoring
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
# API publique
# ---------------------------------------------------------------------------

def load_clip(clip_cfg: dict, data_root: Path) -> TITANClip:
    """Charge un clip depuis une entrée de clips.yaml.

    data_root est toujours le chemin absolu issu de clips.yaml,
    indépendant de l'emplacement du projet benchmark.
    """
    clip_id   = clip_cfg["clip_id"]
    video_dir = data_root / clip_cfg["video_path"]
    ann_path  = data_root / clip_cfg["annotation_path"]

    if not video_dir.exists():
        raise FileNotFoundError(f"Dossier frames introuvable : {video_dir}")
    if not ann_path.exists():
        raise FileNotFoundError(f"CSV annotation introuvable : {ann_path}")

    frame_paths = sorted(video_dir.glob("*.png"), key=lambda p: int(p.stem))
    if not frame_paths:
        raise ValueError(f"Aucune frame .png dans {video_dir}")

    return TITANClip(
        clip_id=clip_id,
        frame_paths=frame_paths,
        annotations=_parse_csv(ann_path),
    )


def load_all_clips(clips_cfg: dict) -> list[TITANClip]:
    """Charge tous les clips depuis clips.yaml complet."""
    data_root = Path(clips_cfg["data_root"])  # chemin absolu TITAN
    return [load_clip(cfg, data_root) for cfg in clips_cfg["clips"]]
