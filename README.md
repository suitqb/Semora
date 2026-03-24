# Semora: VLM Benchmark for Scene Understanding

Semora est un outil de benchmark conçu pour évaluer les performances des modèles de vision-langage (VLM) sur des tâches de compréhension de scènes routières, en utilisant le jeu de données **TITAN**.

L'objectif principal est de mesurer la capacité des modèles à identifier et caractériser les piétons, les véhicules et le contexte global de la scène à partir de séquences d'images (vidéos).

## 🚀 Objectifs du projet

- **Évaluation de la Perception** : Comparer différents VLMs (Gemini, LLaVA, Molmo, Qwen, GPT-4V, etc.) sur leur précision de détection et de classification.
- **Analyse Temporelle** : Étudier l'impact de la taille de la fenêtre temporelle (*Window Size*) sur la compréhension de la dynamique de la scène.
- **Support Multi-Backend** : Intégration facile de modèles via API (Google, Mistral, OpenAI) ou en local (Transformers).
- **Scoring Automatisé** : Calcul automatique de métriques (F1-score) en comparant les sorties des modèles avec les annotations Ground Truth (GT) de TITAN.

## 🛠️ Schéma du Benchmark

Voici le fonctionnement global du pipeline d'évaluation :

![Pipeline du Benchmark](assets/schema.svg)

## 📦 Installation

1. Clonez le dépôt :
   ```bash
   git clone <repo-url>
   cd Semora
   ```

2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

3. Configurez vos clés API dans un fichier `.env` à la racine :
   ```env
   GEMINI_API_KEY=votre_cle
   MISTRAL_API_KEY=votre_cle
   OPENAI_API_KEY=votre_cle
   ```

## ⚙️ Configuration

Le projet utilise trois fichiers YAML pour la configuration (dans le dossier `configs/`) :

- **`models.yaml`** : Activez ou désactivez les modèles à tester et configurez leurs paramètres (température, max tokens, device, etc.).
- **`clips.yaml`** : Définissez les clips du dataset TITAN à utiliser et la stratégie d'échantillonnage (taille des fenêtres, résolution, etc.).
- **`benchmark.yaml`** : Configurez le prompt à utiliser et le dossier de sortie des résultats.

## 🏃 Comment lancer le benchmark

Pour lancer l'évaluation complète avec les configurations par défaut :

```bash
python run_benchmark.py
```

Vous pouvez également spécifier des fichiers de configuration personnalisés :

```bash
python run_benchmark.py --models configs/my_models.yaml --clips configs/my_clips.yaml
```

## 📂 Structure du projet

- `src/core/` : Logique principale du pipeline de benchmark.
- `src/models/` : Implémentations des différents backends de modèles (BaseVLM, API, Local).
- `src/sampling/` : Chargement des clips et échantillonnage des frames.
- `src/parsing/` : Analyseurs pour transformer les réponses textuelles des VLMs en données structurées.
- `src/scoring/` : Calcul des scores et agrégation des résultats.
- `assets/` : Ressources graphiques (schémas, logos).
- `results/` : Dossier contenant les sorties des runs (JSONL, scores finaux).
