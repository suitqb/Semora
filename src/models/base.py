from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from PIL import Image


@dataclass
class VLMOutput:
    """Raw output of a model for a window of frames."""
    model_name: str
    clip_id: str
    center_frame: str
    window_size: int
    frame_names: list[str]
    raw_text: str              # raw text before any parsing — always logged
    latency_s: float
    prompt_tokens: int | None  # None if not available (some local backends)
    completion_tokens: int | None


class BaseVLM(ABC):

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self._loaded = False
        self.parallel_workers: int = config.get("parallel_workers", 1)

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory. Called once before inferences."""
        ...

    @abstractmethod
    def infer(self, frames: list[Image.Image], prompt: str) -> VLMOutput:
        """Run an inference on a window of frames.

        Args:
            frames: PIL frames in chronological order.
            prompt: fixed prompt read from prompts/extraction_v1.txt.

        Returns:
            VLMOutput with raw_text and performance metadata.
        """
        ...

    def unload(self) -> None:
        """Release GPU memory. Optional, called between two models."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, loaded={self._loaded})"
