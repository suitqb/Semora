from __future__ import annotations
from dataclasses import dataclass

from ..parsing.output_parser import ParsedOutput
from ..sampling.clip_loader import FrameAnnotation


# Champs scorés pour les piétons — alignés sur les colonnes GT TITAN
_PERSON_FIELDS = [
    "atomic_action",
    "simple_context",
    "communicative",
    "transporting",
    "age",
]

# Mapping nom de champ ParsedOutput → nom de colonne GT TITAN
_GT_FIELD_MAP = {
    "atomic_action":  "Atomic Actions",
    "simple_context": "Simple Context",
    "communicative":  "Communicative",
    "transporting":   "Transporting",
    "age":            "Age",
    "motion_status":  "Motion Status",
    "trunk_open":     "Trunk Open",
    "doors_open":     "Doors Open",
}


@dataclass
class FieldScore:
    field: str
    tp: int   # true positives
    fp: int   # false positives
    fn: int   # false negatives

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@dataclass
class FrameScore:
    """Score pour une frame donnée."""
    clip_id: str
    center_frame: str
    model_name: str
    window_size: int
    parse_success: bool
    person_scores: dict[str, FieldScore]   # field → FieldScore
    vehicle_scores: dict[str, FieldScore]


def _score_field(
    pred_values: list[str],
    gt_values: list[str],
    field: str,
) -> FieldScore:
    """Score un champ catégoriel : matching exact sur les valeurs normalisées."""
    pred_set = {v.strip().lower() for v in pred_values if v}
    gt_set   = {v.strip().lower() for v in gt_values   if v}

    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    return FieldScore(field=field, tp=tp, fp=fp, fn=fn)


def score_frame(
    parsed: ParsedOutput,
    annotation: FrameAnnotation,
    model_name: str,
    clip_id: str,
    center_frame: str,
    window_size: int,
) -> FrameScore:
    """Compare ParsedOutput vs FrameAnnotation GT et retourne un FrameScore."""

    person_scores: dict[str, FieldScore] = {}
    vehicle_scores: dict[str, FieldScore] = {}

    if parsed.parse_success:
        # --- Piétons ---
        for field in _PERSON_FIELDS:
            gt_col = _GT_FIELD_MAP[field]
            pred_vals = [p.get(field, "") for p in parsed.pedestrians]
            gt_vals   = [p.get(gt_col, "") for p in annotation.persons]
            person_scores[field] = _score_field(pred_vals, gt_vals, field)

        # --- Véhicules ---
        for field in ["motion_status", "trunk_open", "doors_open"]:
            gt_col = _GT_FIELD_MAP[field]
            pred_vals = [v.get(field, "") for v in parsed.vehicles]
            gt_vals   = [v.get(gt_col, "") for v in annotation.vehicles]
            vehicle_scores[field] = _score_field(pred_vals, gt_vals, field)

    return FrameScore(
        clip_id=clip_id,
        center_frame=center_frame,
        model_name=model_name,
        window_size=window_size,
        parse_success=parsed.parse_success,
        person_scores=person_scores,
        vehicle_scores=vehicle_scores,
    )
