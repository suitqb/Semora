from __future__ import annotations
import time
import warnings
from PIL import Image
from .base import BaseVLM, VLMOutput


class _SafeRepetitionPenaltyProcessor:
    """
    Repetition penalty that skips token IDs >= vocab_size.
    Molmo2 image-patch tokens appear in input_ids but are outside lm_head range,
    so the standard RepetitionPenaltyLogitsProcessor would crash on them.
    """

    def __init__(self, penalty: float) -> None:
        self.penalty = penalty

    def __call__(self, input_ids, scores):
        import torch
        vocab_size = scores.shape[-1]
        for i in range(input_ids.shape[0]):
            valid = input_ids[i][input_ids[i] < vocab_size]
            if valid.numel() == 0:
                continue
            token_scores = scores[i].gather(0, valid)
            token_scores = torch.where(
                token_scores < 0,
                token_scores * self.penalty,
                token_scores / self.penalty,
            )
            scores[i].scatter_(0, valid, token_scores)
        return scores


class Molmo(BaseVLM):

    def load(self) -> None:
        import os, torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        import transformers.processing_utils as _pu

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        model_id = self.config["model_id"]

        # Molmo2's remote processor passes extra attrs (e.g. image_use_col_tokens)
        # to ProcessorMixin.__init__ which only accepts specific named args.
        # Patch temporarily to silently absorb the extras as instance attributes.
        _orig_init = _pu.ProcessorMixin.__init__

        def _permissive_init(self, *args, **kwargs):
            attrs = getattr(type(self), "attributes", None)
            if attrs is None and hasattr(type(self), "get_attributes"):
                attrs = type(self).get_attributes()
            valid = set(attrs or []) | {"chat_template"}
            extra = {k: v for k, v in list(kwargs.items()) if k not in valid}
            kwargs = {k: v for k, v in kwargs.items() if k in valid}
            _orig_init(self, *args, **kwargs)
            for k, v in extra.items():
                setattr(self, k, v)

        _pu.ProcessorMixin.__init__ = _permissive_init
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                self._processor = AutoProcessor.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                )
        finally:
            _pu.ProcessorMixin.__init__ = _orig_init

        from transformers import BitsAndBytesConfig

        quant_kwargs: dict = {}
        if self.config.get("load_in_4bit"):
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["model.vision_backbone"],
            )
        elif self.config.get("load_in_8bit"):
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=["model.vision_backbone"],
            )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self._model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                **quant_kwargs,
            )

        self._model.eval()
        self._loaded = True

    def _generate(self, frames: list[Image.Image], prompt: str) -> tuple[str, int, int]:
        """Run one generation pass. Returns (text, n_prompt_tokens, n_completion_tokens)."""
        import torch
        from transformers import LogitsProcessorList

        messages = [
            {
                "role": "user",
                "content": [{"type": "image"} for _ in frames]
                         + [{"type": "text", "text": prompt}],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        inputs = self._processor(text=text, images=frames, return_tensors="pt")

        _device = next(
            (p.device for p in self._model.parameters() if p.device.type != "meta"),
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        inputs = {k: v.to(_device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        n_input_tokens = inputs["input_ids"].shape[-1]
        lp = LogitsProcessorList([_SafeRepetitionPenaltyProcessor(3.0)])

        torch.cuda.empty_cache()
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.config.get("max_new_tokens", 512),
                do_sample=False,
                use_cache=True,
                no_repeat_ngram_size=8,
                logits_processor=lp,
            )

        generated_ids = output[:, n_input_tokens:]
        raw_text = self._processor.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        ).strip()
        return raw_text, int(n_input_tokens), int(generated_ids.shape[-1])

    def infer(self, frames: list[Image.Image], prompt: str) -> VLMOutput:
        assert self._loaded
        t0 = time.perf_counter()
        raw_text, n_prompt, n_completion = self._generate(frames, prompt)
        latency = time.perf_counter() - t0

        return VLMOutput(
            model_name=self.name,
            clip_id="", center_frame="", frame_names=[],
            window_size=len(frames),
            raw_text=raw_text,
            latency_s=round(latency, 3),
            prompt_tokens=n_prompt,
            completion_tokens=n_completion,
        )

    def unload(self) -> None:
        import torch
        if hasattr(self, "_model"):
            del self._model
        if hasattr(self, "_processor"):
            del self._processor
        torch.cuda.empty_cache()
        self._loaded = False
