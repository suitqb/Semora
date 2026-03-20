from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass

from .titan_scorer import FrameScore


@dataclass
class ModelSummary:
    """Scores agrégés pour un modèle × window_size."""
    model_name: str
    window_size: int
    n_frames: int
    parse_success_rate: float
    person_fields: dict[str, dict]   # field → {precision, recall, f1}
    vehicle_fields: dict[str, dict]
    avg_latency_s: float
    total_prompt_tokens: int
    total_completion_tokens: int


def aggregate(
    scores: list[FrameScore],
    latencies: dict[tuple[str, int], list[float]],
    token_counts: dict[tuple[str, int], dict[str, int]],
) -> list[ModelSummary]:
    """Agrège les FrameScore par (model_name, window_size).

    Args:
        scores:       liste de tous les FrameScore du run
        latencies:    {(model, N): [latency_s, ...]}
        token_counts: {(model, N): {prompt: int, completion: int}}
    """
    # Groupe les scores par (model, N)
    groups: dict[tuple[str, int], list[FrameScore]] = defaultdict(list)
    for s in scores:
        groups[(s.model_name, s.window_size)].append(s)

    summaries: list[ModelSummary] = []

    for (model, n), group in groups.items():
        n_frames = len(group)
        n_success = sum(1 for s in group if s.parse_success)

        # Agrège precision/recall/f1 par field en moyennant sur les frames
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

        lats = latencies.get((model, n), [])
        toks = token_counts.get((model, n), {})

        summaries.append(ModelSummary(
            model_name=model,
            window_size=n,
            n_frames=n_frames,
            parse_success_rate=n_success / n_frames if n_frames > 0 else 0.0,
            person_fields=person_fields,
            vehicle_fields=vehicle_fields,
            avg_latency_s=sum(lats) / len(lats) if lats else 0.0,
            total_prompt_tokens=toks.get("prompt", 0),
            total_completion_tokens=toks.get("completion", 0),
        ))

    return sorted(summaries, key=lambda s: (s.model_name, s.window_size))
