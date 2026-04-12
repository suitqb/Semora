from __future__ import annotations
import time
from PIL import Image
from .base import BaseVLM, VLMOutput


class Llava(BaseVLM):

    def load(self) -> None:
        import torch
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

        tf_cfg = self.config.get("transformers", {})
        dtype_str = self.config.get("dtype", "float16")
        dtype = getattr(torch, dtype_str) if hasattr(torch, dtype_str) else torch.float16

        self._processor = LlavaNextProcessor.from_pretrained(
            self.config["model_id"],
            trust_remote_code=tf_cfg.get("trust_remote_code", False),
        )
        self._model = LlavaNextForConditionalGeneration.from_pretrained(
            self.config["model_id"],
            torch_dtype=dtype,
            device_map="auto",
            attn_implementation=tf_cfg.get("attn_implementation", None),
            trust_remote_code=tf_cfg.get("trust_remote_code", False),
        )
        self._model.eval()
        self._loaded = True

    def infer(self, frames: list[Image.Image], prompt: str) -> VLMOutput:
        assert self._loaded
        import torch

        # LLaVA-Next supporte plusieurs images — toutes les frames doivent
        # avoir les mêmes dimensions (garanti par le resize 1280×720 du clip_loader)
        conversation = [{
            "role": "user",
            "content": [{"type": "image"} for _ in frames]
                     + [{"type": "text", "text": prompt}],
        }]
        text = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self._processor(
            images=frames, text=text, return_tensors="pt"
        ).to(self._model.device)

        t0 = time.perf_counter()
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.config.get("max_new_tokens", 512),
                do_sample=False,
            )
        latency = time.perf_counter() - t0

        trimmed = output[:, inputs["input_ids"].shape[-1]:]
        raw_text = self._processor.decode(trimmed[0], skip_special_tokens=True)

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
