from __future__ import annotations
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseVLM

# Mappage des noms de "backend" dans le YAML vers les classes Python correspondantes
_BACKENDS: dict[str, tuple[str, str]] = {
    "mistral_api":         ("src.models.mistral", "Mistral"),
    "openai_api":          ("src.models.gpt",     "GPT"),
    "local_transformers":  None, # sera géré par model_id pour le moment ou via mapping spécifique
    "gemini_api":          ("src.models.gemini",  "Gemini"),
}

# Ancienne logique de secours par model_id pour compatibilité
_OVERRIDES: dict[str, tuple[str, str]] = {
    "allenai/Molmo2-4B":                 ("src.models.molmo",   "Molmo"),
    "Qwen/Qwen2.5-VL-7B-Instruct":       ("src.models.qwen",    "Qwen"),
    "llava-hf/llava-v1.6-mistral-7b-hf": ("src.models.llava",   "Llava"),
    "mistral-medium-latest":             ("src.models.mistral", "Mistral"),
    "gpt-4-vision-preview":              ("src.models.gpt",     "GPT"),
}


def _resolve(cfg: dict) -> type[BaseVLM]:
    backend = cfg.get("backend")
    model_id = cfg.get("model_id", "")

    # 1. Priorité au backend explicite (plus flexible)
    if backend in _BACKENDS and _BACKENDS[backend] is not None:
        module_path, class_name = _BACKENDS[backend]
        return getattr(importlib.import_module(module_path), class_name)

    # 2. Secours par model_id (pour les modèles Transformers locaux qui ont leur propre classe)
    if model_id in _OVERRIDES:
        module_path, class_name = _OVERRIDES[model_id]
        return getattr(importlib.import_module(module_path), class_name)

    raise ValueError(
        f"Impossible de résoudre le modèle : backend={backend!r}, model_id={model_id!r}. "
        f"Vérifiez votre configuration dans models.yaml."
    )


def build_models(models_cfg: dict) -> dict[str, BaseVLM]:
    """Instancie tous les modèles activés depuis models.yaml."""
    instances: dict[str, BaseVLM] = {}
    for name, cfg in models_cfg.items():
        if not cfg.get("enabled", True):
            print(f"[registry] skip {name!r}")
            continue
        cls = _resolve(cfg)
        instances[name] = cls(name=name, config=cfg)
        print(f"[registry] {name!r} ({cfg.get('model_id')}) → {cls.__name__}")
    return instances
