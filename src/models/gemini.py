from __future__ import annotations
import os
import time
from typing import List
from PIL import Image
from dotenv import load_dotenv

from .base import BaseVLM, VLMOutput


class Gemini(BaseVLM):

    def load(self) -> None:
        load_dotenv()
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("pip install google-generativeai")

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY non trouvée dans l'environnement")
            
        genai.configure(api_key=api_key)
        
        self._model_id = self.config["model_id"]
        self._model = genai.GenerativeModel(self._model_id)
        self._loaded = True

    def infer(self, frames: List[Image.Image], prompt: str) -> VLMOutput:
        assert self._loaded

        # Gemini accepte une liste mixant texte et images PIL directement
        content = [prompt] + frames

        t0 = time.perf_counter()
        
        # Generation config
        generation_config = {
            "max_output_tokens": self.config.get("max_new_tokens", 512),
            "temperature": self.config.get("temperature", 0.0),
        }

        response = self._model.generate_content(
            content,
            generation_config=generation_config
        )
        
        latency = time.perf_counter() - t0

        try:
            raw_text = response.text
        except Exception as e:
            # En cas de blocage de sécurité ou autre erreur
            raw_text = f"Error or blocked: {str(e)}"

        # Gemini API ne renvoie pas toujours explicitement les tokens dans l'objet response simple
        # sauf si on utilise count_tokens avant ou si on regarde les métadonnées de réponse.
        # Pour rester simple, on va essayer de les extraire si présents.
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = usage.prompt_token_count if usage else None
        completion_tokens = usage.candidates_token_count if usage else None

        return VLMOutput(
            model_name=self.name,
            clip_id="",
            center_frame="",
            frame_names=[],
            window_size=len(frames),
            raw_text=raw_text.strip(),
            latency_s=round(latency, 3),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
