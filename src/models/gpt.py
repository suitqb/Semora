from __future__ import annotations
import os
import time
from PIL import Image
from typing import Any, cast
from dotenv import load_dotenv
from .base import BaseVLM, VLMOutput
from ..core.utils import pil_to_b64


class GPT(BaseVLM):

    def load(self) -> None:
        load_dotenv(override=True)
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")

        cfg = self.config.get("openai_api", {})
        
        # Récupération de l'API key : priorité config (si non-template) puis env
        api_key = cfg.get("api_key")
        if not api_key or "${" in api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
            
        base_url = cfg.get("base_url")
        if not base_url or "${" in base_url:
            base_url = os.environ.get("OPENAI_API_BASE")

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
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
                    "url": f"data:image/png;base64,{pil_to_b64(img)}",
                    "detail": "high",
                },
            })

        max_tokens = self.config.get("max_new_tokens", 512)
        base_kwargs: dict[str, Any] = {
            "model": self._model_id,
            "messages": cast(Any, [{"role": "user", "content": content}]),
            "temperature": self.config.get("temperature", 0.0),
        }

        t0 = time.perf_counter()
        # Try max_completion_tokens (newer models), fall back to max_tokens, then no limit
        for extra in (
            {"max_completion_tokens": max_tokens},
            {"max_tokens": max_tokens},
            {},
        ):
            try:
                response = self._client.chat.completions.create(**base_kwargs, **extra)
                break
            except Exception as e:
                if "extra_forbidden" in str(e) or "unsupported_parameter" in str(e) or "not supported" in str(e).lower():
                    continue
                raise
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
