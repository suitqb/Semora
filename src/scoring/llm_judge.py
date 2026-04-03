from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Any, cast

from ..parsing.output_parser import ParsedOutput
from ..sampling.clip_loader import FrameAnnotation
from ..core.utils import extract_vlm_text

from ..core.console import console

_SYSTEM_PROMPT = """You are an expert evaluator for autonomous driving perception systems.
You assess the quality of semantic extractions produced by Vision-Language Models (VLMs) on driving scenes.
You always respond with valid JSON only. No explanation, no markdown.
CRITICAL: If you include quotes in the 'justification' field, you MUST escape them (e.g., \\") or use single quotes."""

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


def _format_extraction(parsed: ParsedOutput, window_size: int) -> str:
    center = parsed.center_frame_output(window_size)
    return json.dumps({
        "scene_context": center.scene_context,
        "pedestrians":   center.pedestrians,
        "vehicles":      center.vehicles,
    }, indent=2)


def _format_gt(annotation: FrameAnnotation) -> str:
    return json.dumps({
        "persons":  annotation.persons,
        "vehicles": annotation.vehicles,
    }, indent=2)


def judge(
    parsed: ParsedOutput,
    annotation: FrameAnnotation,
    model_name: str,
    clip_id: str,
    center_frame: str,
    window_size: int,
    judge_cfg: dict,
) -> JudgeScore:
    """Evaluate the semantic quality of a ParsedOutput using an LLM (OpenAI or Mistral)."""

    # Parse failed — no need to call the judge
    if not parsed.parse_success:
        return JudgeScore(
            model_name=model_name, clip_id=clip_id,
            center_frame=center_frame, window_size=window_size,
            completeness=0.0, semantic_richness=0.0,
            spatial_relations=0.0, overall=0.0,
            justifications={},
            judge_error="parse_failed",
        )

    backend     = judge_cfg.get("backend", "openai_api")
    model_id    = judge_cfg.get("model_id", "gpt-4o")
    temperature = judge_cfg.get("temperature", 0.0)
    max_tokens  = judge_cfg.get("max_tokens", 1024)

    try:
        extraction_text = _format_extraction(parsed, window_size)
        gt_text         = _format_gt(annotation)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_TEMPLATE.format(
                extraction=extraction_text,
                ground_truth=gt_text,
            )},
        ]

        if backend == "openai_api":
            from openai import OpenAI
            raw_key = judge_cfg.get("api_key", "${OPENAI_API_KEY}")
            raw_url = judge_cfg.get("base_url", "${OPENAI_API_BASE}")
            client = OpenAI(
                api_key=os.path.expandvars(raw_key),
                base_url=os.path.expandvars(raw_url),
            )
            
            base_kwargs: dict[str, Any] = {
                "model": model_id,
                "messages": cast(Any, messages),
            }
            if "gpt-4" in model_id or "gpt-4o" in model_id:
                base_kwargs["response_format"] = {"type": "json_object"}

            def _is_param_error(e: Exception) -> bool:
                s = str(e)
                return any(k in s for k in ("unsupported_parameter", "not supported", "extra_forbidden", "unsupported_value"))

            response = None
            for extra in (
                {"max_completion_tokens": max_tokens, "temperature": temperature},
                {"max_tokens": max_tokens,            "temperature": temperature},
                {"max_completion_tokens": max_tokens},
                {"max_tokens": max_tokens},
                {},
            ):
                try:
                    response = client.chat.completions.create(**base_kwargs, **extra)
                    break
                except Exception as e:
                    if _is_param_error(e):
                        continue
                    raise

            if response is None:
                raise RuntimeError("No supported parameter combination found for judge model.")

            raw_content = response.choices[0].message.content

        elif backend == "mistral_api":
            from mistralai.client import Mistral
            client = Mistral(
                api_key=os.environ.get("MISTRAL_API_KEY"),
                server_url=os.environ.get("MISTRAL_API_BASE"),
            )
            response = client.chat.complete(
                model=model_id,
                messages=cast(Any, messages),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw_content = response.choices[0].message.content
        
        else:
            raise ValueError(f"Unsupported Judge Backend: {backend}")

        raw_text = extract_vlm_text(raw_content)
        
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            try:
                import ast
                data = ast.literal_eval(raw_text)
            except Exception as je:
                console.print(f"[bold red]⚠ Judge JSON Error ({model_id}):[/bold red]\n{raw_text}")
                raise je

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
        console.print(f"[bold red]✗ Judge Error on {model_name} ({clip_id}/{center_frame}): {e}[/bold red]")
        return JudgeScore(
            model_name=model_name, clip_id=clip_id,
            center_frame=center_frame, window_size=window_size,
            completeness=0.0, semantic_richness=0.0,
            spatial_relations=0.0, overall=0.0,
            justifications={},
            judge_error=str(e),
        )
