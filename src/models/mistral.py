from __future__ import annotations

import os
import time
from typing import List, Any

from PIL import Image
from dotenv import load_dotenv

from .base import BaseVLM, VLMOutput
from ..core.utils import pil_to_b64, extract_vlm_text, dbg, dbg_retry

_MAX_RETRIES = 3
_RETRY_DELAY = 5  # seconds between retries


def _is_transient(e: Exception) -> bool:
    s = str(e).lower()
    return any(k in s for k in ("timeout", "timed out", "503", "502", "429", "rate limit", "connection"))


class Mistral(BaseVLM):

    def load(self) -> None:
        load_dotenv()
        try:
            from mistralai.client import Mistral as MistralClient
        except ImportError:
            raise ImportError("pip install mistralai")

        cfg = self.config.get("mistral_api", {})

        # Récupération de l'API key : priorité config (si non-template) puis env
        api_key = cfg.get("api_key")
        if not api_key or "${" in api_key:
            api_key = os.environ.get("MISTRAL_API_KEY")

        server_url = cfg.get("base_url") or cfg.get("server_url")
        if not server_url or "${" in server_url:
            server_url = os.environ.get("MISTRAL_API_BASE")

        # Mistral client can be sensitive to empty strings for server_url
        effective_url = server_url.strip() if (server_url and server_url.strip()) else None

        self._client = MistralClient(
            api_key=api_key,
            server_url=effective_url,
        )

        self._model_id = self.config["model_id"]
        self._timeout  = cfg.get("timeout_s", 120)
        self._loaded   = True

    def infer(self, frames: List[Image.Image], prompt: str) -> VLMOutput:
        assert self._loaded

        messages: List[Any] = [
            {
                "role": "user",
                "content": [
                    *[
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{pil_to_b64(img)}"
                            }
                        }
                        for img in frames
                    ],
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        t0 = time.perf_counter()

        response = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.chat.complete(
                    model=self._model_id,
                    messages=messages,
                    max_tokens=self.config.get("max_new_tokens", 512),
                    temperature=self.config.get("temperature", 0.0),
                    timeout_ms=int(self._timeout * 1000),
                )
                break
            except Exception as e:
                if _is_transient(e) and attempt < _MAX_RETRIES - 1:
                    dbg_retry(attempt + 1, _MAX_RETRIES, _RETRY_DELAY * (attempt + 1), str(e))
                    time.sleep(_RETRY_DELAY * (attempt + 1))
                    continue
                raise

        if response is None:
            raise RuntimeError("Mistral: all retries exhausted.")

        latency = time.perf_counter() - t0

        content_resp = response.choices[0].message.content
        raw_text = extract_vlm_text(content_resp)

        usage = getattr(response, "usage", None)

        return VLMOutput(
            model_name=self.name,
            clip_id="",
            center_frame="",
            frame_names=[],
            window_size=len(frames),
            raw_text=raw_text,
            latency_s=round(latency, 3),
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        )
