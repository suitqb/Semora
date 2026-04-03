from __future__ import annotations
import os
import time
from PIL import Image
from typing import Any, cast
from dotenv import load_dotenv
from .base import BaseVLM, VLMOutput
from ..core.utils import pil_to_b64, dbg, dbg_infer, dbg_retry

_MAX_RETRIES = 3
_RETRY_DELAY = 5  # seconds between retries


class GPT(BaseVLM):

    def load(self) -> None:
        load_dotenv(override=True)
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")

        cfg = self.config.get("openai_api") or self.config.get("mistral_api", {})
        
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

        max_tokens  = self.config.get("max_new_tokens", 512)
        temperature = self.config.get("temperature", 0.0)
        base_kwargs: dict[str, Any] = {
            "model": self._model_id,
            "messages": cast(Any, [{"role": "user", "content": content}]),
        }

        def _is_param_error(e: Exception) -> bool:
            s = str(e)
            return any(k in s for k in ("extra_forbidden", "unsupported_parameter", "not supported", "unsupported_value"))

        def _is_transient(e: Exception) -> bool:
            s = str(e).lower()
            return any(k in s for k in ("timeout", "timed out", "503", "502", "429", "rate limit"))

        dbg(f"sending to {self._model_id} · {len(frames)} image(s) · max_tokens={max_tokens}")
        t0 = time.perf_counter()
        param_candidates = (
            {"max_completion_tokens": max_tokens, "temperature": temperature},
            {"max_tokens": max_tokens,            "temperature": temperature},
            {"max_completion_tokens": max_tokens},
            {"max_tokens": max_tokens},
            {},
        )
        response = None
        for extra in param_candidates:
            for attempt in range(_MAX_RETRIES):
                try:
                    response = self._client.chat.completions.create(**base_kwargs, **extra)
                    dbg(f"params OK: {list(extra.keys()) or 'none'}")
                    break  # success
                except Exception as e:
                    if _is_param_error(e):
                        dbg(f"params {list(extra.keys())} rejected → trying next combo")
                        break  # try next param combo
                    if _is_transient(e) and attempt < _MAX_RETRIES - 1:
                        dbg_retry(attempt + 1, _MAX_RETRIES, _RETRY_DELAY * (attempt + 1), str(e))
                        time.sleep(_RETRY_DELAY * (attempt + 1))
                        continue
                    raise
            if response is not None:
                break  # found working combo

        if response is None:
            raise RuntimeError("No supported parameter combination found for this model.")
        latency = time.perf_counter() - t0

        raw_text = response.choices[0].message.content or ""
        usage    = response.usage
        dbg_infer(latency, usage.prompt_tokens if usage else None, usage.completion_tokens if usage else None, len(raw_text))

        return VLMOutput(
            model_name=self.name,
            clip_id="", center_frame="", frame_names=[],
            window_size=len(frames),
            raw_text=raw_text.strip(),
            latency_s=round(latency, 3),
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        )
