from __future__ import annotations

import base64
import os
import time
from io import BytesIO
from typing import List, Any

from PIL import Image
from dotenv import load_dotenv

from .base import BaseVLM, VLMOutput


def _pil_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


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

        self._client = MistralClient(
            api_key=api_key,
            server_url=server_url
        )

        self._model_id = self.config["model_id"]
        self._timeout = cfg.get("timeout_s", 60)
        self._loaded = True

    def _extract_text(self, content) -> str:
        """
        Gère tous les formats possibles de réponse Mistral :
        - str
        - list[ContentChunk]
        - list[dict]
        """
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
                                "url": f"data:image/png;base64,{_pil_to_b64(img)}"
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

        response = self._client.chat.complete(
            model=self._model_id,
            messages=messages,
            max_tokens=self.config.get("max_new_tokens", 512),
            temperature=self.config.get("temperature", 0.0),
        )

        latency = time.perf_counter() - t0

        content_resp = response.choices[0].message.content
        raw_text = self._extract_text(content_resp)

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
