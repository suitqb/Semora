from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from PIL import Image


@dataclass
class VLMOutput:
    """Sortie brute d'un modèle pour une window de frames."""
    model_name: str
    clip_id: str
    center_frame: str
    window_size: int
    frame_names: list[str]
    raw_text: str              # texte brut avant tout parsing — toujours loggué
    latency_s: float
    prompt_tokens: int | None  # None si non dispo (certains backends locaux)
    completion_tokens: int | None


class BaseVLM(ABC):

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Charge le modèle en mémoire. Appelé une seule fois avant les inférences."""
        ...

    @abstractmethod
    def infer(self, frames: list[Image.Image], prompt: str) -> VLMOutput:
        """Lance une inférence sur une window de frames.

        Args:
            frames: frames PIL dans l'ordre chronologique.
            prompt: prompt fixe lu depuis prompts/extraction_v1.txt.

        Returns:
            VLMOutput avec raw_text et métadonnées de performance.
        """
        ...

    def unload(self) -> None:
        """Libère la mémoire GPU. Optionnel, appelé entre deux modèles."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, loaded={self._loaded})"
