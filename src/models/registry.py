from __future__ import annotations
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseVLM

_OVERRIDES: dict[str, tuple[str, str]] = {
    "allenai/Molmo2-4B":                 ("src.models.molmo",   "Molmo"),
    "Qwen/Qwen2.5-VL-7B-Instruct":       ("src.models.qwen",    "Qwen"),
    "llava-hf/llava-v1.6-mistral-7b-hf": ("src.models.llava",   "Llava"),
    "mistral-medium-latest":             ("src.models.mistral", "Mistral"),
    "gpt-4-vision-preview":              ("src.models.gpt",     "Gpt4V"),
}


def _resolve(cfg: dict) -> type[BaseVLM]:
    model_id = cfg.get("model_id", "")
    if model_id not in _OVERRIDES:
        raise ValueError(
            f"model_id={model_id!r} inconnu. "
            f"model_id enregistrés : {list(_OVERRIDES)}"
        )
    module_path, class_name = _OVERRIDES[model_id]
    return getattr(importlib.import_module(module_path), class_name)


def build_models(models_cfg: dict) -> dict[str, BaseVLM]:
    """Instancie tous les modèles activés depuis models.yaml.

    Returns:
        dict {nom: instance BaseVLM} — pas encore loadés en mémoire.
    """
    instances: dict[str, BaseVLM] = {}
    for name, cfg in models_cfg.items():
        if not cfg.get("enabled", True):
            print(f"[registry] skip {name!r}")
            continue
        cls = _resolve(cfg)
        instances[name] = cls(name=name, config=cfg)
        print(f"[registry] {name!r} → {cls.__name__}")
    return instances
