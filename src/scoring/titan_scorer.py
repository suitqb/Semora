from __future__ import annotations
from dataclasses import dataclass

from ..parsing.output_parser import ParsedOutput
from ..sampling.clip_loader import FrameAnnotation
from .fields import PERSON_FIELDS as _PERSON_FIELDS, GT_FIELD_MAP as _GT_FIELD_MAP


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


def score_window(
    parsed: ParsedOutput,
    annotations: list,  # list[FrameAnnotation | None]
    model_name: str,
    clip_id: str,
    frame_names: list[str],
    window_size: int,
) -> list[FrameScore]:
    """Score every frame in the window that has a GT annotation.

    Returns one FrameScore per frame with a non-None annotation.
    Frames where the ParsedOutput has no corresponding FrameOutput are skipped.
    """
    scores: list[FrameScore] = []
    for i, (frame_name, annotation) in enumerate(zip(frame_names, annotations)):
        if annotation is None:
            continue
        if not parsed.parse_success or i >= len(parsed.frames):
            scores.append(FrameScore(
                clip_id=clip_id, center_frame=frame_name,
                model_name=model_name, window_size=window_size,
                parse_success=False,
                person_scores={}, vehicle_scores={},
            ))
            continue

        frame_out = parsed.frames[i]
        person_scores: dict[str, FieldScore] = {}
        vehicle_scores: dict[str, FieldScore] = {}

        for field in _PERSON_FIELDS:
            gt_col = _GT_FIELD_MAP[field]
            pred_vals = [p.get(field, "") for p in frame_out.pedestrians]
            gt_vals   = [p.get(gt_col, "") for p in annotation.persons]
            person_scores[field] = _score_field(pred_vals, gt_vals, field)

        for field in ["motion_status", "trunk_open", "doors_open"]:
            gt_col = _GT_FIELD_MAP[field]
            pred_vals = [v.get(field, "") for v in frame_out.vehicles]
            gt_vals   = [v.get(gt_col, "") for v in annotation.vehicles]
            vehicle_scores[field] = _score_field(pred_vals, gt_vals, field)

        scores.append(FrameScore(
            clip_id=clip_id, center_frame=frame_name,
            model_name=model_name, window_size=window_size,
            parse_success=True,
            person_scores=person_scores, vehicle_scores=vehicle_scores,
        ))
    return scores


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
