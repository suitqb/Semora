from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Any

from ..parsing.output_parser import ParsedOutput
from ..sampling.clip_loader import FrameAnnotation

_JUDGE_MODEL = "mistral-large-latest"

_SYSTEM_PROMPT = """You are an expert evaluator for autonomous driving perception systems.
You assess the quality of semantic extractions produced by Vision-Language Models (VLMs) on driving scenes.
You always respond with valid JSON only. No explanation, no markdown."""

_USER_TEMPLATE = """A VLM analyzed a driving scene and produced the following extraction:

EXTRACTION:
{extraction}

GROUND TRUTH annotations for the same frame:
{ground_truth}

Evaluate the extraction on the following criteria. For each, give a score from 0.0 to 1.0 and a brief justification.

Respond with this exact JSON schema:
{{
  "completeness": {{
    "score": <float 0-1>,
    "justification": "<one sentence>"
  }},
  "semantic_richness": {{
    "score": <float 0-1>,
    "justification": "<one sentence>"
  }},
  "spatial_relations": {{
    "score": <float 0-1>,
    "justification": "<one sentence>"
  }},
  "overall": {{
    "score": <float 0-1>,
    "justification": "<one sentence>"
  }}
}}"""


@dataclass
class JudgeScore:
    model_name: str
    clip_id: str
    center_frame: str
    window_size: int
    completeness: float
    semantic_richness: float
    spatial_relations: float
    overall: float
    justifications: dict[str, str]
    judge_error: str | None


def _format_extraction(parsed: ParsedOutput) -> str:
    return json.dumps({
        "scene_context": parsed.scene_context,
        "pedestrians":   parsed.pedestrians,
        "vehicles":      parsed.vehicles,
    }, indent=2)


def _format_gt(annotation: FrameAnnotation) -> str:
    return json.dumps({
        "persons":  annotation.persons,
        "vehicles": annotation.vehicles,
    }, indent=2)


def _extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = []
        for chunk in content:
            if hasattr(chunk, "text") and chunk.text:
                texts.append(chunk.text)
            elif isinstance(chunk, dict):
                texts.append(chunk.get("text", ""))
        return "".join(texts).strip()
    return str(content).strip()


def judge(
    parsed: ParsedOutput,
    annotation: FrameAnnotation,
    model_name: str,
    clip_id: str,
    center_frame: str,
    window_size: int,
) -> JudgeScore:
    """Évalue la qualité sémantique d'un ParsedOutput via mistral-large."""

    # Parse failed — pas la peine d'appeler le judge
    if not parsed.parse_success:
        return JudgeScore(
            model_name=model_name, clip_id=clip_id,
            center_frame=center_frame, window_size=window_size,
            completeness=0.0, semantic_richness=0.0,
            spatial_relations=0.0, overall=0.0,
            justifications={},
            judge_error="parse_failed",
        )

    try:
        from mistralai.client import Mistral

        client = Mistral(
            api_key=os.environ["MISTRAL_API_KEY"],
            server_url=os.environ.get("MISTRAL_API_BASE"),
        )

        response = client.chat.complete(
            model=_JUDGE_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _USER_TEMPLATE.format(
                    extraction=_format_extraction(parsed),
                    ground_truth=_format_gt(annotation),
                )},
            ],
            temperature=0.0,
            max_tokens=512,
        )

        raw_content = response.choices[0].message.content
        raw_text = _extract_text(raw_content)
        data = json.loads(raw_text)

        return JudgeScore(
            model_name=model_name, clip_id=clip_id,
            center_frame=center_frame, window_size=window_size,
            completeness=data["completeness"]["score"],
            semantic_richness=data["semantic_richness"]["score"],
            spatial_relations=data["spatial_relations"]["score"],
            overall=data["overall"]["score"],
            justifications={k: v["justification"] for k, v in data.items()},
            judge_error=None,
        )

    except Exception as e:
        return JudgeScore(
            model_name=model_name, clip_id=clip_id,
            center_frame=center_frame, window_size=window_size,
            completeness=0.0, semantic_richness=0.0,
            spatial_relations=0.0, overall=0.0,
            justifications={},
            judge_error=str(e),
        )
