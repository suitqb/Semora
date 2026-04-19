"""Check what token IDs are in the Molmo2 input (esp. image tokens)."""
import os, warnings, torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_ID = "allenai/Molmo2-8B"
IMAGE_PATH = "data/titan/clips/clip_1/images/000006.png"

import transformers.modeling_rope_utils as _rope
import transformers.processing_utils as _pu

if "default" not in _rope.ROPE_INIT_FUNCTIONS:
    def _default_rope(config, device=None, seq_len=None, **kw):
        base = config.rope_theta
        factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * factor)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
        return inv_freq, 1.0
    _rope.ROPE_INIT_FUNCTIONS["default"] = _default_rope

_orig_init = _pu.ProcessorMixin.__init__
def _permissive_init(self, *args, **kwargs):
    valid = set(type(self).get_attributes()) | {"chat_template"}
    extra = {k: v for k, v in list(kwargs.items()) if k not in valid}
    kwargs = {k: v for k, v in kwargs.items() if k in valid}
    _orig_init(self, *args, **kwargs)
    for k, v in extra.items():
        setattr(self, k, v)
_pu.ProcessorMixin.__init__ = _permissive_init

from transformers import AutoProcessor
from PIL import Image

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
_pu.ProcessorMixin.__init__ = _orig_init

image = Image.open(IMAGE_PATH).convert("RGB")
prompt = "What do you see in this image? Answer in 2-3 sentences."
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    inputs = processor(text=text, images=[image], return_tensors="pt")

ids = inputs["input_ids"][0]
print(f"input_ids shape: {ids.shape}")
print(f"vocab_size (tokenizer): {processor.tokenizer.vocab_size}")
print(f"max token ID in input: {ids.max().item()}")
print(f"min token ID in input: {ids.min().item()}")

# Unique values that are suspiciously large or negative
unique = ids.unique().tolist()
vocab = processor.tokenizer.vocab_size
out_of_bounds = [x for x in unique if x < 0 or x >= vocab]
print(f"OOB token IDs (< 0 or >= {vocab}): {out_of_bounds}")

# Show what the image token ID is from config
try:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"config.image_patch_id: {getattr(cfg, 'image_patch_id', 'NOT FOUND')}")
    print(f"config.vocab_size: {getattr(cfg, 'vocab_size', 'NOT FOUND')}")
except Exception as e:
    print(f"config load error: {e}")
