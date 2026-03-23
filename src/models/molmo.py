from __future__ import annotations
import time
from PIL import Image
from .base import BaseVLM, VLMOutput

class Molmo(BaseVLM):

    def load(self) -> None:
        from transformers.models.auto.modeling_auto import AutoModelForImageTextToText
        from transformers.models.auto.processing_auto import AutoProcessor

        model_id = self.config["model_id"]
        print(f"\n[DEBUG Molmo] Chargement de {model_id}...")

        print("[DEBUG Molmo] Initialisation du processor...")
        self._processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True
        )

        print("[DEBUG Molmo] Initialisation du modèle (dtype=auto, device_map=auto)...")
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype="auto",
            device_map="auto"
        )

        print(f"[DEBUG Molmo] Modèle chargé sur : {self._model.device}")
        if hasattr(self._model, "hf_device_map"):
            print(f"[DEBUG Molmo] Device Map : {self._model.hf_device_map}")

        self._model.eval()
        self._loaded = True
        print("[DEBUG Molmo] Prêt.\n")

    def infer(self, frames: list[Image.Image], prompt: str) -> VLMOutput:
        assert self._loaded
        import torch
        print(f"[DEBUG Molmo] Inférence lancée : {len(frames)} image(s)")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img} for img in frames
                ] + [{"type": "text", "text": prompt}]
            }
        ]

        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self._processor(
            images=frames,
            text=text,
            return_tensors="pt",
            padding=True
        )

        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        print("[DEBUG Molmo] Inputs préparés, démarrage de la génération...")

        t0 = time.perf_counter()
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.config.get("max_new_tokens", 512),
                do_sample=False,
            )
        latency = time.perf_counter() - t0
        print(f"[DEBUG Molmo] Génération terminée en {latency:.2f}s")

        generated_ids = output[0, inputs["input_ids"].shape[-1]:]
        raw_text = self._processor.batch_decode([generated_ids], skip_special_tokens=True)[0]

        return VLMOutput(
            model_name=self.name,
            clip_id="", center_frame="", frame_names=[],
            window_size=len(frames),
            raw_text=raw_text.strip(),
            latency_s=round(latency, 3),
            prompt_tokens=int(inputs["input_ids"].shape[-1]),
            completion_tokens=int(generated_ids.shape[-1]),
        )

    def unload(self) -> None:
        import torch
        if hasattr(self, "_model"):
            del self._model
        if hasattr(self, "_processor"):
            del self._processor
        torch.cuda.empty_cache()
        self._loaded = False
