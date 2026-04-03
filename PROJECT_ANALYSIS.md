# PROJECT_ANALYSIS.md — Cartographie de Semora

## 1. Vue d'ensemble

Semora est un framework de benchmark pour évaluer des VLMs (Vision-Language Models) sur des tâches de compréhension de scènes de conduite autonome. Il s'appuie sur le dataset TITAN, dont les annotations CSV fournissent la vérité terrain par frame et par entité (piétons, véhicules). Pour chaque modèle configuré, le pipeline envoie une fenêtre de N frames en un seul appel, parse la réponse JSON structurée, et calcule deux types de scores : un score symbolique par matching exact contre les annotations GT, et un score sémantique produit par un LLM juge. Les résultats agrégés par modèle et par taille de fenêtre sont sauvegardés dans un dossier horodaté.

---

## 2. Flux de données complet

Le point d'entrée est `clips.yaml`, qui référence les clips TITAN : chemins vers les frames PNG et les CSV d'annotations. `clip_loader.py` lit ces fichiers et construit des objets `TITANClip` (liste de chemins de frames + dictionnaire `frame_name → FrameAnnotation`). Chaque `FrameAnnotation` contient les listes de personnes et de véhicules annotés pour cette frame, avec leurs attributs catégoriels et leur `obj_track_id` stable.

`frame_sampler.py` reçoit un `TITANClip` et génère des `FrameWindow` : pour chaque frame annotée (avec un pas défini par `step`), il sélectionne N frames voisines selon la stratégie d'échantillonnage, charge les images PIL redimensionnées, et désigne la frame centrale comme cible de scoring.

Le backend VLM reçoit la liste d'images PIL et le prompt fixe, envoie la requête à l'API ou au modèle local, et retourne un `VLMOutput` contenant le texte brut de la réponse ainsi que les métriques de performance (latence, tokens). Le `clip_id`, `center_frame` et `frame_names` sont remplis par le pipeline après l'appel.

`output_parser.py` transforme ce texte brut en `ParsedOutput` : si la réponse contient des clés `frame_1`, `frame_2`... (format v2), chaque clé est convertie en `FrameOutput` ; si la réponse est au format v1 (`scene_context` à la racine), elle est encapsulée dans une liste de longueur 1. Le `parse_success` passe à `False` si le JSON est malformé ou si le nombre de frames parsées ne correspond pas au `window_size` attendu.

`titan_scorer.py` extrait la frame centrale du `ParsedOutput` (milieu pour N impair, dernière pour N pair), compare ses listes de valeurs prédites aux valeurs GT de la `FrameAnnotation` correspondante par matching ensembliste exact, et produit un `FrameScore` contenant un `FieldScore` (TP/FP/FN) par champ et par type d'entité. En parallèle, `llm_judge.py` envoie la frame centrale et le GT à un second modèle pour obtenir des scores flottants sur quatre critères sémantiques.

Enfin, `aggregator.py` regroupe tous les `FrameScore` et `JudgeScore` par `(model_name, window_size)`, calcule les moyennes de précision/rappel/F1 par champ, et produit une liste de `ModelSummary` qui est sérialisée dans `scores.json`.

---

## 3. Carte des modules

**`run_benchmark.py`** — Point d'entrée CLI. Gère la sélection interactive des modèles (flèches + espace) ou non-interactive via `--models` / `--use-config`. Délègue entièrement à `src.core.pipeline.run`. Dépend de `pipeline.py` et de `models.yaml`.

**`src/core/pipeline.py`** — Orchestrateur central. Charge les configs, instancie les modèles, boucle sur `(modèle × clip × window_size × frame annotée)`, coordonne inférence → parsing → scoring → journalisation → agrégation → rapport. Dépend de tous les autres modules `src/`.

**`src/core/utils.py`** — Utilitaires partagés. Convertit des images PIL en PNG base64 (`pil_to_b64`) et nettoie les réponses VLM brutes des artefacts markdown et d'échappement JSON (`extract_vlm_text`). Aucune dépendance interne.

**`src/models/base.py`** — Interface abstraite `BaseVLM` avec `load()`, `infer()`, `unload()`, et le dataclass `VLMOutput`. Aucune dépendance interne.

**`src/models/registry.py`** — Résolution dynamique des classes de modèles par `backend` (prioritaire) puis par `model_id` (fallback). Dépend de tous les modules `src/models/`.

**`src/models/mistral.py` / `gpt.py`** — Backends API. Encodent les frames en base64, construisent le message multimodal, appellent l'API respective. Dépendent de `base.py` et `utils.py`.

**`src/models/llava.py` / `qwen.py` / `molmo.py`** — Backends Transformers locaux. Chargent le modèle en mémoire GPU, formatent le message selon les conventions propres à chaque architecture, génèrent la réponse. Dépendent de `base.py`.

**`src/sampling/clip_loader.py`** — Charge les frames PNG et les CSV TITAN. Produit `TITANClip` et `FrameAnnotation`. Dépend de pandas et PIL.

**`src/sampling/frame_sampler.py`** — Génère les `FrameWindow` selon les stratégies `uniform`, `last`, `center`. Dépend de `clip_loader.py`.

**`src/parsing/output_parser.py`** — Détecte le format de réponse (v1 ou v2), extrait et valide le JSON, construit `ParsedOutput` (liste de `FrameOutput`). Aucune dépendance interne.

**`src/scoring/titan_scorer.py`** — Scoring symbolique par matching ensembliste. Dépend de `output_parser.py` et `clip_loader.py`.

**`src/scoring/llm_judge.py`** — Scoring sémantique via LLM externe. Construit le prompt juge à partir de la frame centrale et du GT, parse le JSON de réponse. Dépend de `output_parser.py` et `clip_loader.py`.

**`src/scoring/aggregator.py`** — Agrège `FrameScore` et `JudgeScore` en `ModelSummary`. Dépend de `titan_scorer.py` et `llm_judge.py`.

**`analyze_temporal.py`** — Script post-run d'analyse de cohérence. Lit `parsed_outputs.jsonl`, mesure la stabilité des `track_hint` intra-batch. Aucune dépendance interne.

---

## 4. Format des fichiers de résultats

**`raw_outputs.jsonl`** — Une ligne par inférence réussie.
```json
{"model": "gpt-4o-mini", "N": 2, "clip_id": "clip_1", "center_frame": "000306.png", "raw_text": "...", "latency_s": 3.14}
```

**`parsed_outputs.jsonl`** — Une ligne par inférence, après parsing. Format v2 (actuel) :
```json
{"model": "gpt-4o-mini", "N": 2, "clip_id": "clip_1", "center_frame": "000306.png", "parse_success": true,
 "parsed": {"frames": [{"scene_context": {}, "pedestrians": [...], "vehicles": [...]}, {"scene_context": {}, "pedestrians": [...], "vehicles": [...]}], "parse_success": true, "parse_error": null}}
```
Format v1 (legacy, window_size=1) : `parsed.frames` est une liste de longueur 1, identique à l'ancien `{"scene_context": {}, "pedestrians": [], "vehicles": []}`.

**`judge_outputs.jsonl`** — Une ligne par frame jugée.
```json
{"model": "gpt-4o-mini", "N": 2, "clip_id": "clip_1", "center_frame": "000306.png", "judge_model": "mistral-medium-latest",
 "scores": {"model_name": "...", "clip_id": "...", "center_frame": "...", "window_size": 2,
            "completeness": 0.8, "semantic_richness": 0.7, "spatial_relations": 0.6, "overall": 0.75,
            "justifications": {"completeness": "...", ...}, "judge_error": null}}
```

**`scores.json`** — Liste de `ModelSummary` sérialisés, un par `(model_name, window_size)`, contenant `parse_success_rate`, `f1_context`, `f1_pedestrians`, `f1_vehicles`, les détails par champ (`person_fields`, `vehicle_fields`), `avg_latency_s`, les comptages de tokens, et les scores juge moyens.

---

## 5. Points d'extension identifiés

Pour ajouter un nouveau modèle VLM, il suffit d'implémenter `BaseVLM` dans un nouveau fichier `src/models/monmodele.py`, d'enregistrer son backend dans `_BACKENDS` ou son `model_id` dans `_OVERRIDES` dans `registry.py`, puis d'ajouter une entrée dans `models.yaml`.

Pour ajouter une nouvelle métrique de scoring, il faut créer une fonction dans `src/scoring/`, la brancher dans la boucle d'inférence de `pipeline.py` (entre `score_frame()` et l'agrégation), et ajouter les champs correspondants dans `ModelSummary` et `aggregator.py`. Le pattern est identique à `llm_judge.py`.

Pour ajouter une nouvelle analyse post-run, le modèle est `analyze_temporal.py` : un script autonome à la racine qui lit un fichier JSONL produit par le pipeline et produit un tableau Rich et un CSV.

---

## 6. Problèmes et incohérences détectés

Les fichiers JSONL loggués dans `pipeline.py` utilisent les clés `model` et `N`, mais les dataclasses `FrameScore` et `JudgeScore` utilisent `model_name` et `window_size`. `analyze_temporal.py` lit `rec["model"]` et `rec["N"]` correctement, mais l'ancien `analyze_temporal.py` (avant réécriture) indexait par `model_name` et `window_size`, créant une incompatibilité silencieuse.

Les trois backends locaux (Llava, Qwen, Molmo) ignorent le champ `temperature` de la configuration : ils passent uniquement `do_sample=False` sans lire `self.config.get("temperature")`. La cohérence avec les backends API, qui passent `temperature=0.0`, est donc apparente.

Les dataclasses `VLMOutput` sont initialisées avec `clip_id=""`, `center_frame=""`, `frame_names=[]` dans tous les backends, puis ces champs sont remplis par mutation directe dans `pipeline.py` (ligne 135). C'est un anti-pattern fragile qui rompt l'invariant d'immutabilité des dataclasses et peut causer des bugs si l'ordre des assignations change.

`registry.py` contient dans `_OVERRIDES` des entrées pour `mistral-medium-latest` et `gpt-4-vision-preview` qui dupliquent la logique de `_BACKENDS` : si le backend est correctement spécifié dans le YAML, ces overrides ne sont jamais atteints, mais leur présence est trompeuse.

La docstring de `BaseVLM.infer()` référence encore `prompts/extraction_v1.txt` alors que le pipeline charge désormais le prompt via `benchmark.yaml` et supporte v2.

Le champ `parse_success` est dupliqué dans `ParsedOutput` : il existe à la fois comme attribut de l'objet et comme clé dans le dict sérialisé dans `parsed_outputs.jsonl`, avec deux valeurs qui peuvent théoriquement diverger si la sérialisation via `dataclasses.asdict()` capture un état intermédiaire.

Le `step=50` dans `clips.yaml` avec des clips natifs à 10 fps signifie qu'une seule frame sur 50 est scorée. Sur un clip de 600 frames dont ~400 sont annotées, cela donne environ 8 frames évaluées par modèle par window_size, ce qui est très sparse pour des statistiques robustes.

---

## 7. État actuel de `analyze_temporal.py`

Le script mesure la cohérence **intra-batch** : pour chaque appel VLM individuel (un batch de N frames), il vérifie si les `track_hint` présents dans la frame centrale apparaissent également dans les N-1 autres frames du même batch. La métrique `consistency_rate` capture donc la capacité du modèle à maintenir une description stable d'un même piéton à l'intérieur d'un unique appel.

Ce que le script ne mesure pas est la cohérence **inter-batch** : deux batches distincts dont les fenêtres temporelles se chevauchent (ce qui est le cas avec `step=50` et `window_size=4` où les frames se chevauchent partiellement) peuvent décrire le même piéton physique avec des `track_hint` différents d'un appel à l'autre. Cette incohérence ne serait visible qu'en comparant les track_hints de batches consécutifs portant sur les mêmes frames, ce qui nécessiterait de grouper les batches par frame de chevauchement et de faire tourner l'analyse Jaccard sur les listes de track_hints des batches adjacents plutôt que sur les frames d'un même batch.
