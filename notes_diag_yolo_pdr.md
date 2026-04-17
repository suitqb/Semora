# Diagnostic : Limites de YOLO et impact du format de hint

## Question

Le pipeline YOLO+VLM améliore la PDR par rapport au VLM seul. Deux questions restent ouvertes :

1. **Où est le plafond de YOLO ?** — Combien d'entités GT YOLO rate-t-il, et selon quelle densité ?
2. **Le format du hint change-t-il quelque chose ?** — Les coordonnées bbox `[x1,y1,x2,y2]` et une description textuelle positionnelle (`"person, center-left, mid-distance"`) donnent-elles des PDR équivalentes au VLM ?

---

## Hypothèses

### H1 — YOLO a un plafond de détection propre

YOLO rate des entités selon la densité, l'occlusion, la taille. Ce plafond est mesurable indépendamment du VLM : il suffit de comparer les comptages YOLO aux annotations GT. Si PDR_yolo plafonne à 0.65 à densité=7-10, le VLM ne pourra pas faire mieux même avec un hint parfait.

### H2 — Le format du hint influence la PDR finale

En supposant que YOLO détecte correctement, la façon dont on encode ce résultat pour le VLM peut changer l'utilisation qu'il en fait :

- **Format bbox** (actuel) : `Pedestrian #3 [bbox: 423,187,512,340]`  
  Le VLM doit résoudre un matching spatial texte → image : associer des coordonnées à ce qu'il voit.

- **Format texte** (à tester) : `Pedestrian #3 [center-left, mid-distance, walking]`  
  Description positionnelle en langage naturel, plus alignée avec la façon dont les VLMs traitent les scènes.

Si H2 est actif, on s'attend à voir PDR_vlm_text > PDR_vlm_bbox à iso-détection YOLO.

---

## Mesures à faire

### Mesure 1 — PDR YOLO (plafond du détecteur)

Faire tourner YOLO seul (`detect_frame`) sur les frames des runs complexity existants et compter les entités détectées vs GT.

```
PDR_yolo = min(n_yolo_ped / n_persons_gt, 1.0)
```

Résultat : courbe PDR_yolo par bucket de densité → montre où YOLO décroche.

**Script :** `scripts/diag_yolo_pdr.py --run <run_dir>`

---

### Mesure 2 — PDR VLM avec hint bbox (état actuel)

Déjà disponible dans les `frame_scores.jsonl` des runs avec tracking activé :

```
PDR_vlm_bbox = min(n_persons_pred / n_persons_gt, 1.0)
```

Comparer PDR_vlm_bbox à PDR_yolo indique si le VLM suit le détecteur ou s'en éloigne.

---

### Mesure 3 — PDR VLM avec hint texte (à tester)

Relancer un run complexity avec un format de hint différent : position relative + distance au lieu de coordonnées pixel.

**Format actuel (bbox) :**
```
Pedestrian #3 [bbox: 423,187,512,340]
Vehicle (car) #7 [bbox: 610,201,890,410]
```

**Format texte proposé :**
```
Pedestrian #3 [zone: center-left, distance: mid, size: medium]
Vehicle (car) #7 [zone: center, distance: near, size: large]
```

La zone (left / center-left / center / center-right / right) et la distance (near / mid / far) sont calculées depuis le bbox YOLO — pas d'inférence, juste une transformation du même signal.

**Ce qui change :** le context_builder calcule ces descripteurs depuis le bbox. Le prompt et le pipeline ne changent pas.

---

## Implémentation

### scripts/diag_yolo_pdr.py

Mesure 1 : lit les `frame_scores.jsonl` d'un run existant, recharge les frames, fait tourner YOLO, calcule PDR_yolo par frame et par bucket.

```bash
# Sur un run avec tracking
python scripts/diag_yolo_pdr.py --run runs/complexity/20260415_144802

# Sur un run sans tracking (PDR_yolo seul, pas de comparaison VLM+hint)
python scripts/diag_yolo_pdr.py --run runs/complexity/20260413_101247

# Filtrer un modèle
python scripts/diag_yolo_pdr.py --run runs/complexity/20260415_144802 --model gpt-4o-mini
```

Sortie : `<run>/raw/diag_yolo_pdr.jsonl` + résumé terminal par modèle et par bucket.

---

### Format texte — ce qu'il faut modifier

Pour tester H2, deux changements mineurs :

**1. `src/tracking/context_builder.py`** — ajouter `_format_detections_text()` :

```python
def _zone(x1, x2, img_w=1280):
    cx = (x1 + x2) / 2 / img_w
    if cx < 0.25:   return "left"
    if cx < 0.42:   return "center-left"
    if cx < 0.58:   return "center"
    if cx < 0.75:   return "center-right"
    return "right"

def _distance(y2, img_h=720):
    rel = y2 / img_h
    if rel < 0.45:  return "far"
    if rel < 0.70:  return "mid"
    return "near"

def _format_detections_text(detections):
    parts = []
    for det in detections:
        label = "Pedestrian" if det["class_name"] == "person" else "Vehicle"
        x1, y1, x2, y2 = det["bbox"]
        zone = _zone(x1, x2)
        dist = _distance(y2)
        parts.append(f"{label} #{det['track_id']} [zone: {zone}, distance: {dist}]")
    return ", ".join(parts) if parts else "(no detections)"
```

**2. `configs/benchmark.yaml`** — ajouter un flag `hint_format: text | bbox` pour switcher entre les deux sans toucher au code.

Ce changement est volontairement isolé du reste du pipeline — l'info source (YOLO) est identique, seule la représentation change.

---

## Tableau de résultats

*(À compléter après les runs.)*

### PDR_yolo par densité (Mesure 1)

| Bucket | PDR YOLO | Frames |
|---|---|---|
| 1 | | |
| 2 | | |
| 3-4 | | |
| 5-6 | | |
| 7-10 | | |
| 11-15 | | |
| 16+ | | |

### PDR_yolo vs PDR_vlm_bbox par modèle (Mesures 1+2)

| Modèle | PDR YOLO | PDR VLM+bbox | Δ |
|---|---|---|---|
| gpt-4o-mini | | | |
| gpt-5-mini | | | |
| mistral-large | | | |
| mistral-medium | | | |

### PDR_vlm_bbox vs PDR_vlm_text par modèle (Mesures 2+3)

| Modèle | PDR VLM+bbox | PDR VLM+text | Δ |
|---|---|---|---|
| gpt-4o-mini | | | |
| gpt-5-mini | | | |
| mistral-large | | | |
| mistral-medium | | | |
