"""
Validation test for Molmo2 fix (SafeRepPenalty + no_repeat_ngram_size=4).
Run quand assez de VRAM (~10 GB libres, ferme Overwatch si besoin).

Usage: python debug_molmo2.py [test_id]
  0 = simple prompt, 1 frame
  1 = JSON prompt, 1 frame
  2 = simple prompt, 3 frames
"""
import os, sys, warnings, time, torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

TEST_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0

MODEL_ID = "allenai/Molmo2-8B"
FRAME_PATHS = [
    "data/titan/clips/clip_1/images/000006.png",
    "data/titan/clips/clip_1/images/000126.png",
    "data/titan/clips/clip_1/images/000246.png",
]

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

from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, LogitsProcessorList
from PIL import Image

class SafeRepPenalty:
    """Repetition penalty that skips token IDs >= vocab_size (Molmo2 image-patch tokens)."""
    def __init__(self, penalty):
        self.penalty = penalty
    def __call__(self, input_ids, scores):
        vocab_size = scores.shape[-1]
        for i in range(input_ids.shape[0]):
            valid = input_ids[i][input_ids[i] < vocab_size]
            if valid.numel() == 0:
                continue
            token_scores = scores[i].gather(0, valid)
            token_scores = torch.where(token_scores < 0,
                token_scores * self.penalty, token_scores / self.penalty)
            scores[i].scatter_(0, valid, token_scores)
        return scores

TESTS = [
    (1, "What do you see in this image? Answer in 2-3 sentences."),
    (1, 'Describe this scene as JSON: {"location": "<road/sidewalk/other>", "description": "<one sentence>"}'),
    (3, "Describe each frame in 1 sentence."),
]

n_frames, prompt = TESTS[TEST_ID]
images = [Image.open(FRAME_PATHS[i]).convert("RGB") for i in range(n_frames)]
print(f"\n[TEST {TEST_ID}] {n_frames} frame(s), use_cache=False + SafeRepPenalty + ngram=4", flush=True)

free_mib = torch.cuda.mem_get_info()[0] // (1024**2)
print(f"VRAM free: {free_mib} MiB (need ~10000 MiB)", flush=True)
if free_mib < 9500:
    print("WARNING: VRAM may be insufficient, close Overwatch/games and retry", flush=True)

print("Loading processor...", flush=True)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
_pu.ProcessorMixin.__init__ = _orig_init

print("Loading model (8-bit)...", flush=True)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, trust_remote_code=True, device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=["model.vision_backbone"],
        ),
    )
model.eval()

_orig_prepare = model.prepare_inputs_for_generation
def _patched_prepare(input_ids, past_key_values=None, cache_position=None, **kwargs):
    if cache_position is None:
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(past_len, past_len + input_ids.shape[1], device=input_ids.device)
    return _orig_prepare(input_ids, past_key_values=past_key_values,
                         cache_position=cache_position, **kwargs)
model.prepare_inputs_for_generation = _patched_prepare

_device = next((p.device for p in model.parameters() if p.device.type != "meta"), torch.device("cuda"))
print(f"Loaded on {_device}", flush=True)

messages = [{"role": "user", "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]}]
text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    inputs = processor(text=text, images=images, return_tensors="pt")
inputs = {k: v.to(_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
n_in = inputs["input_ids"].shape[-1]
print(f"Input tokens: {n_in}", flush=True)

logits_processors = LogitsProcessorList([SafeRepPenalty(1.5)])

torch.cuda.empty_cache()
t0 = time.perf_counter()
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=300, do_sample=False,
                         use_cache=False, no_repeat_ngram_size=4,
                         logits_processor=logits_processors)
latency = time.perf_counter() - t0

text_out = processor.tokenizer.decode(out[0, n_in:], skip_special_tokens=True)
print(f"\n[{latency:.1f}s]\n{text_out}", flush=True)
