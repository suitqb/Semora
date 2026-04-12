from __future__ import annotations
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseVLM

# Mapping of "backend" names in YAML to corresponding Python classes
_BACKENDS: dict[str, tuple[str, str]] = {
    "mistral_api":         ("src.models.mistral", "Mistral"),
    "openai_api":          ("src.models.gpt",     "GPT"),
}

# Legacy fallback logic by model_id for compatibility.
# This mechanism handles 'local_transformers' and other models where 
# the implementation is chosen based on the model ID rather than the backend name.
_OVERRIDES: dict[str, tuple[str, str]] = {
    "allenai/Molmo2-8B":                 ("src.models.molmo",   "Molmo"),
    "Qwen/Qwen2.5-VL-7B-Instruct":       ("src.models.qwen",    "Qwen"),
    "llava-hf/llava-v1.6-mistral-7b-hf": ("src.models.llava",   "Llava"),
    "mistral-medium-latest":             ("src.models.mistral", "Mistral"),
    "gpt-4-vision-preview":              ("src.models.gpt",     "GPT"),
}


def _resolve(cfg: dict) -> type[BaseVLM]:
    backend = cfg.get("backend")
    model_id = cfg.get("model_id", "")

    # 1. Priority to explicit backend
    if backend in _BACKENDS:
        module_path, class_name = _BACKENDS[backend]
        return getattr(importlib.import_module(module_path), class_name)

    # 2. Fallback by model_id (for local Transformers models with their own class)
    if model_id in _OVERRIDES:
        module_path, class_name = _OVERRIDES[model_id]
        return getattr(importlib.import_module(module_path), class_name)

    valid_backends = list(_BACKENDS.keys())
    raise ValueError(
        f"Unable to resolve model: backend={backend!r}, model_id={model_id!r}.\n"
        f"Valid backends: {valid_backends}. If using 'local_transformers', "
        f"ensure model_id is in: {list(_OVERRIDES.keys())}"
    )


def build_models(models_cfg: dict) -> dict[str, BaseVLM]:
    """Instantiate all enabled models from models_cfg."""
    instances: dict[str, BaseVLM] = {}
    for name, cfg in models_cfg.items():
        if not cfg.get("enabled", True):
            continue
        cls = _resolve(cfg)
        instances[name] = cls(name=name, config=cfg)
    return instances
