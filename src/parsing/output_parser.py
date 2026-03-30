from __future__ import annotations
import json
import re
from dataclasses import dataclass


@dataclass
class FrameOutput:
    """Parsed semantic content for a single frame."""
    scene_context: dict
    pedestrians: list[dict]
    vehicles: list[dict]


@dataclass
class ParsedOutput:
    """Parsed result of a VLM inference — one FrameOutput per input frame."""
    frames: list[FrameOutput]
    parse_success: bool
    parse_error: str | None

    def center_frame_output(self, window_size: int) -> FrameOutput:
        """Return the FrameOutput corresponding to the center (scored) frame.

        Convention:
          - odd window_size  → middle frame : frames[window_size // 2]
          - even window_size → last frame   : frames[-1]

        Falls back to frames[-1] on IndexError.
        """
        if not self.frames:
            return FrameOutput(scene_context={}, pedestrians=[], vehicles=[])
        idx = window_size // 2 if window_size % 2 == 1 else -1
        try:
            return self.frames[idx]
        except IndexError:
            return self.frames[-1]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> str:
    """Extract the first valid JSON block from raw text."""
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
                return text[start : i + 1]

    return text[start:]


def _frame_output_from_dict(d: dict) -> FrameOutput:
    return FrameOutput(
        scene_context=d.get("scene_context", {}),
        pedestrians=d.get("pedestrians", []),
        vehicles=d.get("vehicles", []),
    )


def _parse_multi_frame(data: dict, window_size: int) -> ParsedOutput:
    """Parse a v2 response: {"frame_1": {...}, "frame_2": {...}, ...}."""
    frames: list[FrameOutput] = []
    for i in range(1, window_size + 1):
        key = f"frame_{i}"
        if key not in data:
            return ParsedOutput(
                frames=[],
                parse_success=False,
                parse_error=f"Missing key '{key}' in response (expected {window_size} frames)",
            )
        frames.append(_frame_output_from_dict(data[key]))

    return ParsedOutput(frames=frames, parse_success=True, parse_error=None)


def _parse_single_frame(data: dict) -> ParsedOutput:
    """Parse a v1 response: {"scene_context": ..., "pedestrians": ..., "vehicles": ...}."""
    return ParsedOutput(
        frames=[_frame_output_from_dict(data)],
        parse_success=True,
        parse_error=None,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse(raw_text: str, window_size: int = 1) -> ParsedOutput:
    """Parse raw VLM text into a ParsedOutput.

    Handles two response formats:
      - v2 multi-frame: {"frame_1": {...}, "frame_2": {...}, ...}
      - v1 single-frame: {"scene_context": {...}, "pedestrians": [...], ...}
        (wrapped into a one-element frames list for backward compatibility)

    parse_success=False if:
      - JSON is malformed
      - v2 format but parsed frame count != window_size
    """
    _EMPTY = ParsedOutput(frames=[], parse_success=False, parse_error=None)

    try:
        json_str = _extract_json(raw_text)
        data = json.loads(json_str)
    except (json.JSONDecodeError, ValueError) as e:
        return ParsedOutput(frames=[], parse_success=False, parse_error=str(e))

    # Detect format: v2 if any "frame_N" key is present
    if any(k.startswith("frame_") for k in data):
        return _parse_multi_frame(data, window_size)

    # v1 fallback — wrap in single-element list; valid only when window_size == 1
    parsed = _parse_single_frame(data)
    if window_size != 1:
        return ParsedOutput(
            frames=parsed.frames,
            parse_success=False,
            parse_error=f"Got v1 single-frame response but window_size={window_size}",
        )
    return parsed
