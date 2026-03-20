from __future__ import annotations
import time
from PIL import Image
from .base import BaseVLM, VLMOutput


class Molmo(BaseVLM):

    def load(self) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self.config["model_id"],
            trust_remote_code=True,
            padding_side="left",
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.config["model_id"],
            trust_remote_code=True,
            dtype="auto",
            device_map=self.config.get("device", "cuda"),
        )
        self._model.eval()
        self._loaded = True

    def infer(self, frames: list[Image.Image], prompt: str) -> VLMOutput:
        assert self._loaded
        import torch

        messages = [{
            "role": "user",
            "content": [dict(type="image", image=img) for img in frames]
                     + [dict(type="text", text=prompt)],
        }]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        t0 = time.perf_counter()
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.config.get("max_new_tokens", 512),
                do_sample=False,
            )
        latency = time.perf_counter() - t0

        trimmed = output[:, inputs["input_ids"].shape[-1]:]
        raw_text = self._processor.batch_decode(trimmed, skip_special_tokens=True)[0]

        return VLMOutput(
            model_name=self.name,
            clip_id="", center_frame="", frame_names=[],
            window_size=len(frames),
            raw_text=raw_text.strip(),
            latency_s=round(latency, 3),
            prompt_tokens=int(inputs["input_ids"].shape[-1]),
            completion_tokens=int(trimmed.shape[-1]),
        )

    def unload(self) -> None:
        import torch
        del self._model, self._processor
        torch.cuda.empty_cache()
        self._loaded = False
