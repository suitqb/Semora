from __future__ import annotations
import time
import warnings
from PIL import Image
from .base import BaseVLM, VLMOutput


class _SafeRepetitionPenaltyProcessor:
    """
    Drop-in replacement for RepetitionPenaltyLogitsProcessor that skips token IDs
    >= vocab_size (Molmo2 image-patch tokens are in input_ids but outside lm_head range).
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
        import transformers.modeling_rope_utils as _rope

        # Reduce CUDA allocator fragmentation before any large allocation.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning,
                                        message=r".*(?:image|video)_processor_class.*deprecated.*")
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
                # The vision encoder must stay unquantized — bitsandbytes would try to cast
                # its bias to the quantised dtype → crash. Full path needed: should_convert_module()
                # checks prefix/suffix, so "vision_backbone" alone won't match "model.vision_backbone.*".
                llm_int8_skip_modules=["model.vision_backbone"],
            )
        elif self.config.get("load_in_8bit"):
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=["model.vision_backbone"],
            )

        with warnings.catch_warnings():
            for msg in (r".*image_processor_class.*", r".*video_processor_class.*",
                        r".*rope_config_validation.*"):
                warnings.filterwarnings("ignore", message=msg, category=FutureWarning)
            self._model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map="auto",
                # Force bfloat16 for all non-quantized layers (embedding, lm_head, vision backbone).
                # Without this, those layers default to float32, causing a dtype mismatch when
                # vision features (float32) are added into language embeddings (bfloat16) → garbage output.
                torch_dtype=torch.bfloat16,
                **quant_kwargs,
            )

        self._model.eval()

        # transformers removed auto-creation of cache_position for remote-code models, but
        # Molmo2's prepare_inputs_for_generation does `cache_position[0]` without a None guard.
        # Patch the instance so it builds cache_position itself when the caller omits it.
        _orig_prepare = self._model.prepare_inputs_for_generation

        def _patched_prepare(input_ids, past_key_values=None, cache_position=None, **kwargs):
            if cache_position is None:
                past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_len, past_len + input_ids.shape[1], device=input_ids.device
                )
            return _orig_prepare(input_ids, past_key_values=past_key_values,
                                 cache_position=cache_position, **kwargs)

        self._model.prepare_inputs_for_generation = _patched_prepare

        self._loaded = True

    def infer(self, frames: list[Image.Image], prompt: str) -> VLMOutput:
        assert self._loaded
        import torch

        # Step 1: render the chat template to a plain string containing <|image|> placeholders.
        # Molmo2's processor.__call__ then replaces each placeholder with the actual image tokens.
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"} for _ in frames]
                         + [{"type": "text", "text": prompt}],
            }
        ]
        text = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Step 2: tokenise + encode images.
        inputs = self._processor(
            text=text,
            images=frames,
            return_tensors="pt",
        )

        # With device_map="auto" + CPU offloading, model.device may be 'meta'.
        # Find the first real (non-meta) device instead.
        _device = next(
            (p.device for p in self._model.parameters() if p.device.type != "meta"),
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        inputs = {k: v.to(_device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        n_input_tokens = inputs["input_ids"].shape[-1]

        from transformers import LogitsProcessorList

        logits_processors = LogitsProcessorList([
            _SafeRepetitionPenaltyProcessor(1.5),
        ])

        torch.cuda.empty_cache()
        t0 = time.perf_counter()
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.config.get("max_new_tokens", 512),
                do_sample=False,
                use_cache=False,
                no_repeat_ngram_size=4,
                logits_processor=logits_processors,
            )
        latency = time.perf_counter() - t0

        generated_ids = output[:, n_input_tokens:]
        # Use the wrapped tokenizer for decoding (processor.tokenizer is always present).
        raw_text = self._processor.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )

        return VLMOutput(
            model_name=self.name,
            clip_id="", center_frame="", frame_names=[],
            window_size=len(frames),
            raw_text=raw_text.strip(),
            latency_s=round(latency, 3),
            prompt_tokens=int(n_input_tokens),
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
