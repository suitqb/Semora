## Objectif de ces plans de test

L'objectif des plans de test est de mesurer les performances et les limites des VLMs sur deux aspects :
- Leur capacité de tracking dans une série d'images
- La qualité d'extraction sémantique

Plusieurs plans de test sont mis en place, chacun associé à des métriques adaptées à l'aspect évalué.

## Données

Les données sont issues du dataset [TITAN](https://usa.honda-ri.com/titan), produit par le Honda Research Institute. Les clips ont été sélectionnés en fonction des scènes représentées et de la densité de piétons par scène, afin de comparer les performances des modèles sur différents types de situations.

Tous les clips sélectionnés disposent d'annotations servant de vérité terrain (*ground truth*, GT). Ces annotations sont utilisées pour le calcul des métriques et l'évaluation des performances des différents modèles.

## Modèles utilisés

Plusieurs types de modèles sont évalués afin de comparer leurs performances selon différents axes. Le tableau ci-dessous liste les modèles retenus ainsi que leurs modalités d'implémentation.

| Modèle | Model ID | Taille | Implémentation | Rôle |
|---|---|---|---|---|
| mistral-medium | mistral-medium-latest | — | API (Mistral) | Extraction |
| mistral-large | ADADAS-mistral-large-3 | — | API (Azure) | Extraction |
| molmo2-4b | allenai/Molmo2-4B | 4B | Local (Transformers) | Extraction |
| qwen2-5-vl-7b | Qwen/Qwen2.5-VL-7B-Instruct | 7B | Local (Transformers) | Extraction |
| llava-1-6 | llava-hf/llava-v1.6-mistral-7b-hf | 7B | Local (Transformers) | Extraction |
| gpt-4o-mini | gpt-4o-mini | — | API (Azure) | Extraction |
| gpt-5-mini | ADADAS-gpt-5-mini | — | API (Azure) | Judge |

Cette sélection vise à mettre en évidence plusieurs tendances :

- Les modèles propriétaires (Mistral, OpenAI) présentent-ils un avantage sur les modèles open source en termes de qualité d'extraction ?
- La taille du modèle a-t-elle un impact mesurable sur les performances, le coût et le temps d'inférence ?
- Est-il plus pertinent d'exécuter les modèles en local ou de passer par les APIs déployées sur Azure ?
- Comment Molmo2-4B se positionne-t-il sur le tracking, sachant que son architecture intègre nativement un module dédié à cet usage ?

## Protocole d'évaluation et pipeline

### Pipeline d'inférence

Pour chaque combinaison `(modèle, clip, window_size)`, le pipeline suit les étapes suivantes :

1. Chargement des frames : les frames sont échantillonnées depuis le clip vidéo à un pas fixe (`step=15`), soit environ une frame toutes les 1,5 secondes (vidéos à 10 fps natifs). La résolution maximale est fixée à 1280×720.
2. Construction de fenêtres temporelles : autour de chaque frame cible, une fenêtre de `N` frames est construite selon une stratégie d'échantillonnage (`uniform`). Deux tailles de fenêtre sont testées : `window_size = 2` et `window_size = 4`.
3. Inférence VLM : les `N` frames de la fenêtre sont encodées en base64 PNG et envoyées au modèle avec le prompt d'extraction (`prompts/extraction_v2.txt`). Le modèle retourne une réponse textuelle structurée en JSON.
4. Parsing de la sortie : la réponse est parsée pour en extraire les champs structurés (piétons, véhicules, contexte de scène). Un flag `parse_success` indique si le parsing a réussi.
5. Scoring : deux méthodes de scoring sont appliquées en parallèle (voir section suivante).

![schemas pipeline](assets/schema.svg)

### Métriques

Deux méthodes de scoring sont appliquées en parallèle pour chaque frame :

- Scoring symbolique (GT) : comparaison ensembliste des champs extraits (piétons et véhicules) contre les annotations TITAN. Produit une précision, un rappel et un F1 par champ, agrégés sur l'ensemble des frames et des clips.
- LLM Judge : un modèle juge (`gpt-5-mini`) évalue la qualité sémantique de l'extraction sur 4 critères notés de 0 à 1 — `completeness`, `semantic_richness`, `spatial_relations`, `overall` — avec une justification textuelle pour chaque score.

Le détail des champs, du calcul des métriques et des critères du juge est décrit dans les plans de test concernés.

### Clips utilisés

9 clips du dataset TITAN sont utilisés pour le benchmark : `clip_1`, `clip_75`, `clip_279`, `clip_280`, `clip_544`, `clip_552`, `clip_621`, `clip_761`, `clip_785`. Tous sont annotés et échantillonnés à 10 fps natifs.

## Plans de test

### Plan 1 - Qualité d'extraction des VLM

#### Objectif

Ce plan vise à mesurer la capacité des VLMs à extraire correctement les informations sémantiques d'une scène de conduite à partir d'une ou plusieurs frames. On cherche à répondre aux questions suivantes :

- Quel modèle produit les extractions les plus fidèles au ground truth ?
- Est-ce que donner plus de contexte temporel (window_size plus grand) améliore la qualité d'extraction ?
- Y a-t-il des champs systématiquement mieux ou moins bien extraits (ex. `age` vs `atomic_action`) ?
- Est-ce que le scoring symbolique et le LLM judge convergent, ou mesurent-ils des choses différentes ? (Normalement non)

#### Hypothèses

- Les modèles propriétaires (Mistral, GPT) devraient surpasser les modèles open source locaux sur la qualité brute d'extraction.
- Une fenêtre temporelle plus large (`window_size=4`) devrait améliorer les scores en donnant plus de contexte au modèle pour lever des ambiguïtés (ex. distinguer `walking` de `standing` sur une seule frame peut être difficile).
- Certains champs seront structurellement plus difficiles à prédire : `age` et `communicative` requièrent des indices visuels fins, tandis que `motion_status` (véhicule) devrait être plus simple.

#### Métriques utilisées

##### Scoring symbolique — GT Scorer

Le GT Scorer compare les champs extraits par le modèle aux annotations TITAN frame par frame. C'est une évaluation factuelle et stricte : soit la valeur prédite correspond exactement à celle du GT, soit elle ne correspond pas.

Pourquoi ce scorer ?
Les annotations TITAN sont des labels catégoriels fermés, produits par des annotateurs humains sur chaque frame. C'est la référence la plus fiable dont on dispose. Ce scorer mesure directement la capacité du modèle à produire des labels corrects, ce qui est la définition même d'une bonne extraction structurée.

**Champs évalués — Piétons :**

| Champ | Ce qu'il mesure | Exemple de valeurs |
|---|---|---|
| `atomic_action` | Action physique instantanée du piéton | `walking`, `standing`, `running`, `sitting` |
| `simple_context` | Situation de déplacement dans la scène | `crossing a street at pedestrian crossing`, `waiting to cross street`, `walking along the side of the road` |
| `communicative` | Comportement de communication observable | `none of the above`, `talking on phone`, `looking into phone`, `talking in group` |
| `transporting` | Transport d'un objet | `none of the above`, `carrying with both hands`, `pushing`, `pulling` |
| `age` | Tranche d'âge estimée visuellement | `child`, `adult`, `senior over 65` |

**Champs évalués — Véhicules :**

| Champ | Ce qu'il mesure | Exemple de valeurs |
|---|---|---|
| `motion_status` | État de mouvement du véhicule | `moving`, `stopped`, `parked` |
| `trunk_open` | Coffre ouvert ou fermé | `open`, `closed` |
| `doors_open` | Portes ouvertes ou fermées | `open`, `closed` |

**Comment ça se calcule ?**

Pour une frame donnée, on rassemble toutes les valeurs prédites pour un champ (une par entité détectée) et toutes les valeurs du GT. Le matching est ensembliste : on regarde quelles valeurs sont communes, lesquelles sont en trop, et lesquelles manquent.

```
GT          = { "walking", "standing" }   ← 2 piétons annotés
Prédiction  = { "walking", "running" }    ← 2 piétons prédits

TP = { "walking" }   → prédit ET dans le GT         = 1
FP = { "running" }   → prédit MAIS absent du GT     = 1
FN = { "standing" }  → dans le GT MAIS non prédit   = 1

Précision = TP / (TP + FP) = 1/2 = 0.50  → sur ce que le modèle a prédit, combien est correct ?
Rappel    = TP / (TP + FN) = 1/2 = 0.50  → sur ce qui existe dans la scène, combien a été trouvé ?
F1        = 2 × 0.50 × 0.50 / (0.50 + 0.50) = 0.50
```

Le F1 est la métrique principale car elle équilibre précision et rappel — un modèle qui prédit tout (rappel=1 mais précision basse) ou rien (précision=1 mais rappel bas) sera pénalisé. Les scores sont calculés champ par champ, puis moyennés sur l'ensemble des frames et des clips.

Limite du GT Scorer : il ne mesure que ce que TITAN annote. Un modèle qui décrit très bien le contexte spatial, les interactions entre entités ou des détails fins non couverts par les 8 champs ne sera pas récompensé. C'est pour ça qu'on utilise le LLM Judge en complément.

---

##### Scoring sémantique — LLM as a Judge

Le LLM Judge utilise un modèle de langage (`gpt-5-mini`) comme évaluateur pour mesurer la qualité sémantique globale d'une extraction, au-delà du matching exact de catégories.

**Pourquoi ce scorer ?**

Le GT Scorer est binaire par nature : `walking` prédit alors que le GT dit `standing`, c'est 0. Mais un modèle qui décrit "une personne marchant lentement, semblant hésiter à traverser" capture quelque chose de juste sur la scène même si le label exact diffère. De même, les champs TITAN ne couvrent pas les relations spatiales, les interactions entre entités, ou le contexte global de la scène — autant d'éléments qui distinguent une bonne description d'une mauvaise.

Le juge reçoit à la fois l'extraction complète du modèle et le GT de la frame, et évalue 4 critères indépendants :

| Critère | Ce qu'il mesure | Pourquoi c'est important |
|---|---|---|
| `completeness` | Est-ce que toutes les entités et attributs présents ont été détectés ? | Un modèle qui ignore des piétons ou des véhicules est inutilisable en conduite autonome |
| `semantic_richness` | Les descriptions sont-elles précises, détaillées et pertinentes ? | Mesure la "densité" d'information utile dans l'extraction |
| `spatial_relations` | Les relations spatiales entre entités sont-elles correctement décrites ? | Crucial pour la compréhension de scène : savoir qu'un piéton est "devant le véhicule" vs "sur le trottoir" change tout |
| `overall` | Score de qualité globale de l'extraction | Synthèse des 3 critères précédents, permet une comparaison rapide entre modèles |

Chaque critère est noté de **0 à 1** et accompagné d'une justification textuelle, ce qui permet de comprendre pourquoi un modèle est pénalisé ou récompensé — pas juste un chiffre.

**Exemple de sortie juge :**

```json
{
  "completeness": {
    "score": 0.75,
    "justification": "The model identified 3 out of 4 pedestrians and both vehicles, missing one occluded pedestrian near the building."
  },
  "semantic_richness": {
    "score": 0.85,
    "justification": "Descriptions are detailed with accurate action labels and contextual information."
  },
  "spatial_relations": {
    "score": 0.60,
    "justification": "Relative positions of vehicles are correct but pedestrian-to-vehicle distances are not described."
  },
  "overall": {
    "score": 0.73,
    "justification": "Good overall extraction with minor gaps in completeness and spatial reasoning."
  }
}
```

Le juge opère à température 0 pour garantir la reproductibilité des scores entre les runs.

Limite du LLM Judge : l'évaluation reste subjective par nature — deux juges différents peuvent produire des scores différents pour la même extraction. On utilise un modèle fixe et une température de 0 pour minimiser cette variabilité, mais il faut garder en tête que ces scores ne sont pas aussi déterministes que le F1.

#### Variables testées

| Variable | Valeurs |
|---|---|
| Modèle | mistral-medium, mistral-large, gpt-4o-mini, molmo2-4b, qwen2-5-vl-7b, llava-1-6 |
| Window size | 2, 4, 6 |
| Clips | 9 clips TITAN |

#### Résultats attendus

Pour chaque modèle et chaque `window_size`, on agrège les scores sur l'ensemble des frames et des clips. On compare ensuite :

- Le **F1 moyen global** (piétons + véhicules) par modèle
- Le **F1 par champ** pour identifier les attributs les plus/moins bien extraits
- Les **scores LLM judge** moyens par modèle
- L'**effet du window_size** : delta de performance entre `window_size=2` et `window_size=4`

---

### Plan 2 - Tracking des entités dans la scène

#### Objectif

Ce plan évalue la capacité des VLMs à maintenir une identité cohérente pour chaque entité à travers plusieurs frames consécutives. Extraire correctement les attributs d'une entité sur une frame est une chose ; reconnaître que c'est la même entité sur la frame suivante en est une autre. Cette cohérence temporelle est essentielle pour tout système de perception en conduite autonome.

On cherche à répondre aux questions suivantes :

- Les modèles sont-ils capables d'associer la même entité d'une frame à l'autre via le `track_hint` ?
- Est-ce que la cohérence de tracking s'améliore avec une fenêtre temporelle plus large ?
- Molmo2-4B, qui intègre nativement un module de tracking dans son architecture, se distingue-t-il des autres modèles sur ce critère ?
- Quels types d'entités sont les plus difficiles à tracker (piétons occultés, véhicules en mouvement rapide) ?

#### Contexte : le `track_hint`

Le prompt d'extraction demande au modèle d'associer à chaque entité un `track_hint` — un descripteur visuel court et stable (ex. `"person in blue jacket on the left"`). La règle imposée dans le prompt est stricte : si la même entité est visible dans plusieurs frames, son `track_hint` doit être identique mot pour mot.

C'est ce mécanisme que ce plan de test évalue. Un modèle qui reformule, abrège ou change de descripteur entre deux frames casse le tracking, même si l'extraction de chaque frame est correcte individuellement.

#### Hypothèses

- Les modèles propriétaires devraient avoir une meilleure cohérence de tracking grâce à une meilleure compréhension des instructions du prompt.
- Molmo2-4B devrait se distinguer positivement sur ce plan grâce à son module de tracking intégré, même si ses scores d'extraction brute (plan 1) sont inférieurs aux modèles propriétaires.
- Une fenêtre temporelle plus large devrait aider le tracking : voir plusieurs frames en même temps permet au modèle de construire une représentation plus stable des entités.
- Les piétons seront plus difficiles à tracker que les véhicules (plus nombreux, plus similaires visuellement, plus de mouvement).

#### Métriques utilisées

Le tracking n'est pas couvert par le GT Scorer (TITAN n'annote pas les `track_hint`). L'évaluation repose donc sur deux approches complémentaires :

**Cohérence du `track_hint` (métrique interne)**

Pour chaque fenêtre temporelle, on mesure si le modèle utilise des `track_hint` identiques pour les mêmes entités d'une frame à l'autre. En pratique, on groupe les entités par `track_hint` à travers les frames et on calcule :

- Taux de consistance : proportion de `track_hint` qui restent identiques entre la première et la dernière frame de la fenêtre
- Fragmentation : nombre de `track_hint` uniques par entité supposée (une entité bien trackée = 1 `track_hint` unique sur toute la fenêtre)

**LLM Judge — critère `spatial_relations`**

Le critère `spatial_relations` du LLM Judge est particulièrement pertinent ici : un modèle qui tracke bien une entité devrait maintenir des descriptions spatiales cohérentes et précises sur toute la séquence. Ce score est analysé en comparaison inter-frames pour détecter les incohérences.

#### Résultats attendus

- Taux de consistance par modèle : comparaison directe entre modèles sur leur capacité à maintenir les `track_hint`
- Effet du window_size : est-ce qu'un contexte temporel plus large améliore la cohérence ?
- Molmo2 vs les autres : analyse spécifique des performances de Molmo2-4B sur ce plan
- Piétons vs véhicules : est-ce que certains types d'entités sont intrinsèquement plus difficiles à tracker ?

---

### Plan 3 - Montée en complexité

#### Objectif

Ce plan évalue comment les performances des VLMs évoluent lorsque la complexité de la scène augmente. L'idée est de pousser les modèles vers leurs limites en faisant varier la densité et la difficulté des scènes, afin d'identifier à partir de quel seuil les performances se dégradent significativement.

On cherche à répondre aux questions suivantes :

- Est-ce que les performances (F1, LLM judge) baissent avec l'augmentation du nombre d'entités dans la scène ?
- Certains modèles sont-ils plus robustes que d'autres face à la complexité ?
- Y a-t-il un seuil de densité (nombre de piétons, véhicules) au-delà duquel tous les modèles échouent ?
- Est-ce que le window_size aide à mieux gérer les scènes complexes, ou est-ce que ça devient contre-productif (trop d'information à traiter en même temps) ?

#### Définition de la complexité

La complexité d'une scène est définie par plusieurs dimensions cumulables :

| Dimension | Description | Proxy mesurable |
|---|---|---|
| Densité de piétons | Nombre de piétons présents dans la frame | Nombre d'entrées `persons` dans le GT |
| Densité de véhicules | Nombre de véhicules présents dans la frame | Nombre d'entrées `vehicles` dans le GT |
| Diversité d'actions | Variété des actions effectuées simultanément | Nombre de valeurs `atomic_action` distinctes dans le GT |
| Occlusions | Entités partiellement cachées, plus difficiles à décrire | Détectable via les scores de `completeness` du juge |

Les 9 clips ont été sélectionnés en partie pour leur diversité de densité de scène. Ce plan exploite cette variation pour construire une analyse de sensibilité.

#### Hypothèses

- Les performances F1 devraient baisser avec l'augmentation du nombre d'entités : plus il y a de piétons, plus le risque d'en oublier ou de leur attribuer de mauvais labels augmente.
- Le critère `completeness` du LLM Judge devrait être le premier à se dégrader avec la densité, car c'est lui qui mesure directement les entités manquées.
- Les modèles locaux (7B) pourraient se dégrader plus vite que les modèles propriétaires face à la complexité, leur capacité de raisonnement étant plus limitée.
- Un `window_size` plus grand pourrait aider sur les scènes simples (plus de contexte) mais pénaliser sur les scènes complexes (surcharge d'information).

#### Métriques utilisées

Ce plan réutilise l'ensemble des métriques des plans 1 et 2, mais en les stratifiant par niveau de complexité de la scène plutôt qu'en les agrégeant globalement.

- F1 par champ (GT Scorer) segmenté par tranche de densité (ex. 1-2 entités, 3-5 entités, 6+ entités)
- Scores LLM Judge (`completeness`, `overall`) par tranche de complexité
- Courbe de dégradation : évolution des scores en fonction du nombre d'entités dans la scène

#### Résultats attendus

- Courbes de dégradation par modèle : F1 et `completeness` en fonction du nombre d'entités — permet de voir quel modèle résiste le mieux à la complexité
- Seuil de rupture : à partir de combien d'entités les performances chutent significativement ?
- Effet du window_size sur les scènes complexes : aide ou surcharge ?
- Comparaison des dimensions de complexité : est-ce que la densité de piétons est plus pénalisante que la diversité d'actions ?
