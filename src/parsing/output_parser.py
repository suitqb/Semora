from __future__ import annotations
import json
import re
from dataclasses import dataclass


@dataclass
class ParsedOutput:
    """Parsed result of a VLM inference."""
    scene_context: dict
    pedestrians: list[dict]
    vehicles: list[dict]
    parse_success: bool
    parse_error: str | None


def _extract_json(text: str) -> str:
    """Extract the first valid JSON block from the raw text.

    Some models wrap their response in markdown (```json ... ```)
    despite the opposite instruction in the prompt.
    """
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)

    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return text[start:]


def parse(raw_text: str) -> ParsedOutput:
    """Parse raw VLM text into a usable structure.

    Tolerant: if JSON is malformed, returns an empty ParsedOutput
    with parse_success=False rather than crashing.
    """
    try:
        json_str = _extract_json(raw_text)
        data = json.loads(json_str)
        return ParsedOutput(
            scene_context=data.get("scene_context", {}),
            pedestrians=data.get("pedestrians", []),
            vehicles=data.get("vehicles", []),
            parse_success=True,
            parse_error=None,
        )
    except (json.JSONDecodeError, ValueError) as e:
        return ParsedOutput(
            scene_context={},
            pedestrians=[],
            vehicles=[],
            parse_success=False,
            parse_error=str(e),
        )
