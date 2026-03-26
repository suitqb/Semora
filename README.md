# Semora — VLM Benchmark for Scene Understanding

Semora est un outil de benchmark conçu pour évaluer les capacités des modèles de vision-langage (VLM) sur des tâches de compréhension de scènes routières, en utilisant le jeu de données **TITAN**.

## Pourquoi ce benchmark ?

Dans un pipeline de perception pour la conduite autonome, le choix du VLM est structurant : c'est lui qui produit les extractions sémantiques (attributs piétons, état des véhicules, contexte de scène) qui alimentent les étapes aval — construction de graphe de scène, prédiction d'intention.

Mais tous les VLMs ne se valent pas sur ce type de tâche. Les benchmarks génériques (VQA, captioning) ne capturent pas les propriétés qui importent ici : la précision sur des attributs catégoriels fins, la cohérence temporelle sur une fenêtre vidéo, ou la richesse des relations spatiales entre entités.

**Semora répond à une question simple : quel modèle produit les meilleures extractions pour mon cas d'usage spécifique ?**

Il permet de comparer des modèles hétérogènes (API distantes, modèles locaux) sur les mêmes clips TITAN, avec des métriques adaptées à l'extraction sémantique structurée.

## Pipeline d'évaluation

![Pipeline du Benchmark](assets/schema.svg)

Pour chaque frame, le pipeline :
1. échantillonne une fenêtre temporelle autour d'une frame centrale,
2. soumet les images au VLM évalué avec un prompt standardisé,
3. parse la réponse en une structure `ParsedOutput` (piétons, véhicules, contexte),
4. score l'extraction selon deux méthodes complémentaires décrites ci-dessous.

## Méthodes de scoring

### 1. TITAN GT Scorer — matching symbolique

Le scorer symbolique compare les extractions du modèle aux annotations Ground Truth du dataset TITAN sur des **champs catégoriels définis** :

| Entité | Champs scorés |
|---|---|
| Piétons | `atomic_action`, `simple_context`, `communicative`, `transporting`, `age` |
| Véhicules | `motion_status`, `trunk_open`, `doors_open` |

Pour chaque champ et chaque frame, les valeurs prédites et GT sont converties en ensembles (`set`), puis comparées par **matching ensembliste exact** :

- **TP** — valeurs correctement prédites (pred ∩ GT)
- **FP** — valeurs prédites absentes du GT (pred − GT)
- **FN** — valeurs GT non prédites (GT − pred)

On en déduit précision, rappel et **F1-score par champ**, agrégés ensuite sur l'ensemble des frames.

Ce scorer est rapide, déterministe et directement interprétable. Il est cependant aveugle à tout ce qui n'est pas annoté dans TITAN : la qualité de la description de scène, les relations spatiales, ou la pertinence contextuelle des extractions.

---

### 2. LLM-as-Judge — évaluation sémantique

Pour compenser les angles morts du scorer symbolique, Semora intègre un **juge LLM** (GPT-4o, référence dans la littérature scientifique sur ce pattern) qui évalue la qualité sémantique globale de chaque extraction.

Le judge reçoit en entrée l'extraction complète (`scene_context`, piétons, véhicules) et les annotations GT correspondantes, puis note quatre critères indépendants sur une échelle 0–1 :

| Critère | Ce que ça mesure |
|---|---|
| `completeness` | Tous les éléments GT sont-ils couverts par l'extraction ? |
| `semantic_richness` | L'extraction apporte-t-elle des informations pertinentes au-delà du GT ? |
| `spatial_relations` | Les relations spatiales entre entités sont-elles correctement décrites ? |
| `overall` | Jugement holistique de la qualité de l'extraction |

Chaque score est accompagné d'une justification en une phrase, ce qui permet d'auditer les évaluations.

Le juge est appelé à `temperature=0.0` pour garantir la **reproductibilité** des scores entre runs. Les frames dont le parsing a échoué (`parse_success=False`) sont court-circuitées et reçoivent automatiquement un score nul, sans appel API.

**Limites connues du pattern LLM-as-judge :** le modèle juge voit simultanément l'extraction et le GT, ce qui peut introduire un biais de compensation sur les scènes difficiles. Il convient donc d'interpréter ces scores en complément du GT scorer, et non comme une métrique indépendante.

## Résultats

Les résultats sont produits en JSONL par run, avec un score final agrégé par modèle, taille de fenêtre temporelle et champ. Ils permettent de comparer directement les modèles sur les deux axes : précision catégorielle (GT scorer) et qualité sémantique (judge).
