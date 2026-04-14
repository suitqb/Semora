from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass

from .titan_scorer import FrameScore
from .llm_judge import JudgeScore


@dataclass
class ModelSummary:
    """Aggregated scores for a model × window_size."""
    model_name: str
    window_size: int
    n_frames: int
    parse_success_rate: float
    f1_context: float
    f1_pedestrians: float
    f1_vehicles: float
    person_fields: dict[str, dict]   # field → {precision, recall, f1}
    vehicle_fields: dict[str, dict]
    avg_latency_s: float
    total_prompt_tokens: int
    total_completion_tokens: int
    # --- Judge Scores (0-1) ---
    avg_judge_completeness: float | None = None
    avg_judge_semantic_richness: float | None = None
    avg_judge_spatial_relations: float | None = None
    avg_judge_overall: float | None = None


def aggregate(
    scores: list[FrameScore],
    latencies: dict[tuple[str, int], list[float]],
    token_counts: dict[tuple[str, int], dict[str, int]],
    judge_scores: list[JudgeScore] | None = None,
) -> list[ModelSummary]:
    """Aggregate FrameScore and JudgeScore by (model_name, window_size)."""
    # Group scores by (model, N)
    groups: dict[tuple[str, int], list[FrameScore]] = defaultdict(list)
    for s in scores:
        groups[(s.model_name, s.window_size)].append(s)

    # Group judge scores by (model, N)
    judge_groups: dict[tuple[str, int], list[JudgeScore]] = defaultdict(list)
    if judge_scores:
        for js in judge_scores:
            judge_groups[(js.model_name, js.window_size)].append(js)

    summaries: list[ModelSummary] = []

    for (model, n), group in groups.items():
        n_frames = len(group)
        n_success = sum(1 for s in group if s.parse_success)

        # Aggregate precision/recall/f1 per field by averaging over frames
        def _agg_field(field_name: str, src: str) -> dict:
            values = [
                getattr(s, f"{src}_scores").get(field_name)
                for s in group
                if s.parse_success and field_name in getattr(s, f"{src}_scores")
            ]
            if not values:
                return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            return {
                "precision": sum(v.precision for v in values) / len(values),
                "recall":    sum(v.recall    for v in values) / len(values),
                "f1":        sum(v.f1        for v in values) / len(values),
            }

        person_fields = {
            f: _agg_field(f, "person")
            for f in ["atomic_action", "simple_context", "communicative", "transporting", "age"]
        }
        vehicle_fields = {
            f: _agg_field(f, "vehicle")
            for f in ["motion_status", "trunk_open", "doors_open"]
        }

        f1_context = person_fields["simple_context"]["f1"]
        f1_pedestrians = sum(f["f1"] for f in person_fields.values()) / len(person_fields) if person_fields else 0.0
        f1_vehicles = sum(f["f1"] for f in vehicle_fields.values()) / len(vehicle_fields) if vehicle_fields else 0.0

        lats = latencies.get((model, n), [])
        toks = token_counts.get((model, n), {})

        # --- Judge Aggregation ---
        avg_comp = None
        avg_sem  = None
        avg_spat = None
        avg_over = None
        
        j_group = judge_groups.get((model, n), [])
        if j_group:
            # Only count frames where the judge did not have an error
            j_valid = [js for js in j_group if js.judge_error is None]
            if j_valid:
                avg_comp = sum(js.completeness for js in j_valid) / len(j_valid)
                avg_sem  = sum(js.semantic_richness for js in j_valid) / len(j_valid)
                avg_spat = sum(js.spatial_relations for js in j_valid) / len(j_valid)
                avg_over = sum(js.overall for js in j_valid) / len(j_valid)

        summaries.append(ModelSummary(
            model_name=model,
            window_size=n,
            n_frames=n_frames,
            parse_success_rate=n_success / n_frames if n_frames > 0 else 0.0,
            f1_context=f1_context,
            f1_pedestrians=f1_pedestrians,
            f1_vehicles=f1_vehicles,
            person_fields=person_fields,
            vehicle_fields=vehicle_fields,
            avg_latency_s=sum(lats) / len(lats) if lats else 0.0,
            total_prompt_tokens=toks.get("prompt", 0),
            total_completion_tokens=toks.get("completion", 0),
            avg_judge_completeness=avg_comp,
            avg_judge_semantic_richness=avg_sem,
            avg_judge_spatial_relations=avg_spat,
            avg_judge_overall=avg_over,
        ))

    return sorted(summaries, key=lambda s: (s.model_name, s.window_size))


def build_scores_payload(
    summaries: list[ModelSummary],
    tracking: bool,
    mode: str = "extraction",
) -> dict:
    """Wrap aggregated summaries with run metadata for serialisation to scores.json."""
    return {
        "meta": {"tracking": tracking, "mode": mode},
        "results": [s.__dict__ for s in summaries],
    }
