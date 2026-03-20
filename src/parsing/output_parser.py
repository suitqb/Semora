from __future__ import annotations
import json
import re
from dataclasses import dataclass


@dataclass
class ParsedOutput:
    """Résultat parsé d'une inférence VLM."""
    scene_context: dict
    pedestrians: list[dict]
    vehicles: list[dict]
    parse_success: bool
    parse_error: str | None


def _extract_json(text: str) -> str:
    """Extrait le premier bloc JSON valide du texte brut.

    Certains modèles wrappent leur réponse dans du markdown (```json ... ```)
    malgré l'instruction contraire dans le prompt.
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
    """Parse le texte brut d'un VLM en structure exploitable.

    Tolérant : si le JSON est malformé, retourne un ParsedOutput vide
    avec parse_success=False plutôt que de planter.
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
