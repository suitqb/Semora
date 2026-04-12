from __future__ import annotations
import time
from PIL import Image
from .base import BaseVLM, VLMOutput


class Molmo(BaseVLM):

    def load(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        import transformers.processing_utils as _pu
        import transformers.modeling_rope_utils as _rope

        model_id = self.config["model_id"]

        # transformers ≥5.x removed the 'default' rope type from ROPE_INIT_FUNCTIONS.
        # Molmo2's remote code sets rope_type="default" when rope_scaling is None and
        # then does ROPE_INIT_FUNCTIONS["default"], which raises KeyError.
        # Re-inject the standard (unscaled) RoPE init so the lookup succeeds.
        if "default" not in _rope.ROPE_INIT_FUNCTIONS:
            def _default_rope(config, device=None, seq_len=None, **kw):
                base = config.rope_theta
                factor = getattr(config, "partial_rotary_factor", 1.0)
                head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
                dim = int(head_dim * factor)
                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
                return inv_freq, 1.0
            _rope.ROPE_INIT_FUNCTIONS["default"] = _default_rope

        # transformers ≥5.x ProcessorMixin.__init__ validates kwargs strictly
        # against get_attributes() (modality sub-processors only).  Molmo2's
        # remote processor passes extra optional attrs (e.g. image_use_col_tokens)
        # to super().__init__, which triggers the new validation.  Patch it
        # temporarily so those extras are silently set as instance attributes.
        _orig_init = _pu.ProcessorMixin.__init__

        def _permissive_init(self, *args, **kwargs):
            valid = set(type(self).get_attributes()) | {"chat_template"}
            extra = {k: v for k, v in list(kwargs.items()) if k not in valid}
            kwargs = {k: v for k, v in kwargs.items() if k in valid}
            _orig_init(self, *args, **kwargs)
            for k, v in extra.items():
                setattr(self, k, v)

        _pu.ProcessorMixin.__init__ = _permissive_init
        try:
            self._processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
            )
        finally:
            _pu.ProcessorMixin.__init__ = _orig_init

        quant_kwargs = {}
        if self.config.get("load_in_8bit"):
            quant_kwargs["load_in_8bit"] = True
        elif self.config.get("load_in_4bit"):
            quant_kwargs["load_in_4bit"] = True
        else:
            quant_kwargs["torch_dtype"] = getattr(torch, self.config.get("dtype", "bfloat16"))

        self._model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map="auto",
            **quant_kwargs,
        )

        self._model.eval()
        self._loaded = True

    def infer(self, frames: list[Image.Image], prompt: str) -> VLMOutput:
        assert self._loaded
        import torch

        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": img} for img in frames]
                         + [{"type": "text", "text": prompt}],
            }
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        t0 = time.perf_counter()
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.config.get("max_new_tokens", 512),
                do_sample=False,
            )
        latency = time.perf_counter() - t0

        generated_ids = output[0, inputs["input_ids"].shape[-1]:]
        raw_text = self._processor.batch_decode(
            [generated_ids], skip_special_tokens=True
        )[0]

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
