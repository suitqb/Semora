# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Semora is a benchmarking framework for evaluating Vision-Language Models (VLMs) on autonomous driving scene understanding tasks using the TITAN dataset. Models are evaluated via two methods: symbolic matching against ground truth annotations (precision/recall/F1) and semantic quality scoring via an LLM judge.

## Setup & Running

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env  # Add MISTRAL_API_KEY, OPENAI_API_KEY, etc.

# Run benchmark (interactive model selection)
python run_benchmark.py

# Run with models from config (non-interactive)
python run_benchmark.py --use-config

# Run with specific models
python run_benchmark.py --models "mistral-medium,gpt-4o-mini"

# Custom config paths
python run_benchmark.py \
  --models-cfg configs/models.yaml \
  --clips-cfg configs/clips.yaml \
  --benchmark-cfg configs/benchmark.yaml
```

There is no build step — pure Python project. Requires TITAN data in `data/titan/` with frames under `clips/` and CSV annotations under `annotations/`.

## Architecture

The pipeline flows: **config load → model selection → clip loading → inference loop → scoring → aggregation → report**.

### Key configs (all in `configs/`)
- `models.yaml` — model definitions, backends, API keys, enabled flags
- `clips.yaml` — TITAN clip references, `window_sizes`, `frame_selection` strategy, `step` (frame sparsity), `max_resolution`
- `benchmark.yaml` — enable/disable `titan_gt` scorer and `llm_judge`, output file settings

### Inference loop (`src/core/pipeline.py`)
For each `(model, clip, window_size)`:
1. `frame_sampler.py` generates temporal windows around each target frame (strategies: `uniform`, `last`, `center`)
2. The VLM backend receives N frames as base64 PNG + the fixed extraction prompt (`prompts/extraction_v1.txt`)
3. `output_parser.py` extracts JSON from the raw response (handles markdown wrapping, malformed JSON)
4. `titan_scorer.py` does set-based matching of predicted fields vs. CSV ground truth
5. `llm_judge.py` (if enabled) calls a judge model to score completeness, semantic_richness, spatial_relations, and overall (0–1 scale, temperature=0.0)

### Model backends (`src/models/`)
All implement `BaseVLM` (`base.py`). Registry (`registry.py`) resolves class by `backend` key in `models.yaml`:
- `mistral_api` → `mistral.py`
- `openai_api` → `gpt.py`
- `llava`, `qwen`, `molmo` → local Transformers backends

To add a new model: implement `BaseVLM`, register in `registry.py`, add entry to `configs/models.yaml`.

### Scoring fields
**Pedestrians**: `atomic_action`, `simple_context`, `communicative`, `transporting`, `age`
**Vehicles**: `motion_status`, `trunk_open`, `doors_open`

### Output
Results saved under `results/runs/YYYYMMDD_HHMMSS/`:
- `scores.json` — aggregated per-model/window_size summaries
- `raw_outputs.jsonl` — raw VLM text responses
- `parsed_outputs.jsonl` — extracted JSON + `parse_success` flag
- `judge_outputs.jsonl` — LLM judge scores + justifications

Agis comme un partenaire intellectuel rigoureux et honnête. Tes principes directeurs sont l'exactitude, la transparence et la concision.
Honnêteté absolue : Si tu ne connais pas la réponse, dis "Je ne sais pas". N'invente jamais de faits, de sources ou de données.
Admets tes limites : Si une requête est ambiguë ou impossible à traiter avec certitude, signale-le immédiatement avant de tenter une réponse.
Pas de flatterie : Évite les formules de politesse excessives, ne cherche pas à aller dans mon sens par défaut. Critique mes idées si elles sont erronées.
Priorité à la précision : Préfère une réponse courte et vérifiée à un long paragraphe spéculatif.
