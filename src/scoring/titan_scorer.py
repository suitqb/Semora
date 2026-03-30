from __future__ import annotations
from dataclasses import dataclass

from ..parsing.output_parser import ParsedOutput
from ..sampling.clip_loader import FrameAnnotation


# Fields scored for pedestrians - aligned with TITAN GT columns
_PERSON_FIELDS = [
    "atomic_action",
    "simple_context",
    "communicative",
    "transporting",
    "age",
]

# Mapping of ParsedOutput field names to TITAN GT column names
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
    """Score for a given frame."""
    clip_id: str
    center_frame: str
    model_name: str
    window_size: int
    parse_success: bool
    person_scores: dict[str, FieldScore]   # field -> FieldScore
    vehicle_scores: dict[str, FieldScore]


def _score_field(
    pred_values: list[str],
    gt_values: list[str],
    field: str,
) -> FieldScore:
    """Score a categorical field: exact matching on normalized values."""
    pred_set = {str(v).strip().lower() for v in pred_values if v is not None and str(v).strip()}
    gt_set   = {str(v).strip().lower() for v in gt_values   if v is not None and str(v).strip()}

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
    """Compare ParsedOutput vs FrameAnnotation GT and return a FrameScore.

    Scoring uses only the center frame from the multi-frame ParsedOutput:
      - odd window_size  → middle frame (frames[window_size // 2])
      - even window_size → last frame   (frames[-1])
    """
    person_scores: dict[str, FieldScore] = {}
    vehicle_scores: dict[str, FieldScore] = {}

    if parsed.parse_success:
        center = parsed.center_frame_output(window_size)

        # Pedestrians
        for field in _PERSON_FIELDS:
            gt_col = _GT_FIELD_MAP[field]
            pred_vals = [p.get(field, "") for p in center.pedestrians]
            gt_vals   = [p.get(gt_col, "") for p in annotation.persons]
            person_scores[field] = _score_field(pred_vals, gt_vals, field)

        # Vehicles
        for field in ["motion_status", "trunk_open", "doors_open"]:
            gt_col = _GT_FIELD_MAP[field]
            pred_vals = [v.get(field, "") for v in center.vehicles]
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
