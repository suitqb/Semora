from __future__ import annotations
import base64
import os
import time
from io import BytesIO
from PIL import Image
from typing import Any, cast
from .base import BaseVLM, VLMOutput


def _pil_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class Gpt4V(BaseVLM):

    def load(self) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")

        cfg = self.config["openai_api"]
        self._client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_API_BASE"],
            timeout=cfg.get("timeout_s", 60),
        )
        self._model_id = self.config["model_id"]
        self._loaded   = True

    def infer(self, frames: list[Image.Image], prompt: str) -> VLMOutput:
        assert self._loaded

        content: list[dict] = [{"type": "text", "text": prompt}]
        for img in frames:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{_pil_to_b64(img)}",
                    "detail": "high",
                },
            })

        t0 = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self._model_id,
            messages=cast(Any, [{"role": "user", "content": content}]),
            max_tokens=self.config.get("max_new_tokens", 512),
            temperature=self.config.get("temperature", 0.0),
        )
        latency = time.perf_counter() - t0

        raw_text = response.choices[0].message.content or ""
        usage    = response.usage

        return VLMOutput(
            model_name=self.name,
            clip_id="", center_frame="", frame_names=[],
            window_size=len(frames),
            raw_text=raw_text.strip(),
            latency_s=round(latency, 3),
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        )
