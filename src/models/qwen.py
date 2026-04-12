from __future__ import annotations
import time
from PIL import Image
from .base import BaseVLM, VLMOutput


class Qwen(BaseVLM):

    def load(self) -> None:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        tf_cfg = self.config.get("transformers", {})
        dtype_str = self.config.get("dtype", "bfloat16")
        dtype = getattr(torch, dtype_str) if hasattr(torch, dtype_str) else torch.bfloat16

        quant_kwargs = {}
        if self.config.get("load_in_8bit"):
            quant_kwargs["load_in_8bit"] = True
        elif self.config.get("load_in_4bit"):
            quant_kwargs["load_in_4bit"] = True
        else:
            quant_kwargs["torch_dtype"] = dtype

        self._processor = AutoProcessor.from_pretrained(
            self.config["model_id"],
            trust_remote_code=tf_cfg.get("trust_remote_code", False),
        )
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config["model_id"],
            device_map="auto",
            attn_implementation=tf_cfg.get("attn_implementation", None),
            trust_remote_code=tf_cfg.get("trust_remote_code", False),
            **quant_kwargs,
        )
        self._model.eval()
        self._loaded = True

    def infer(self, frames: list[Image.Image], prompt: str) -> VLMOutput:
        assert self._loaded
        import torch
        from qwen_vl_utils import process_vision_info

        # Qwen2.5-VL supporte nativement plusieurs images dans un même message
        image_content = [{"type": "image", "image": img} for img in frames]
        messages = [{
            "role": "user",
            "content": image_content + [{"type": "text", "text": prompt}],
        }]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, _ = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        ).to(self._model.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.config.get("max_new_tokens", 512),
                do_sample=False,
            )
        latency = time.perf_counter() - t0

        trimmed = output[:, inputs["input_ids"].shape[-1]:]
        raw_text = self._processor.batch_decode(
            trimmed, skip_special_tokens=True
        )[0]

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
