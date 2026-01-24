# Question Answering Fine-Tuning on SQuAD

## Contexte académique

Ce projet est réalisé dans le cadre d’un enseignement de **Traitement Automatique du Langage Naturel (NLP)** au niveau **Master 2**.  
Il a pour objectif d’appliquer les méthodes de **fine-tuning de modèles Transformer** présentées en cours à la tâche de **Question Answering extractif**, en utilisant le dataset **SQuAD v1.1**.

Le projet s’appuie sur l’écosystème **Hugging Face** (`datasets`, `transformers`) et met l’accent sur la **méthodologie**, la **comparaison de modèles** et l’**analyse des résultats**, plutôt que sur le développement d’une infrastructure de production.

---

## Objectifs du projet

Les objectifs principaux sont les suivants :

- Explorer et analyser le dataset SQuAD v1.1
- Fine-tuner plusieurs modèles pré-entraînés pour la tâche de Question Answering extractif
- Évaluer les modèles à l’aide des métriques officielles (Exact Match, F1)
- Comparer les performances, les coûts computationnels et les compromis entre modèles
- Proposer une organisation claire et reproductible du travail expérimental

---

## Dataset

Le dataset utilisé est **SQuAD v1.1 (Stanford Question Answering Dataset)**, chargé via la bibliothèque `datasets` de Hugging Face.

Il se compose de :
- 87 599 exemples d’entraînement
- 10 570 exemples de validation

Chaque exemple contient :
- un contexte (extrait d’article Wikipédia),
- une question,
- une ou plusieurs réponses annotées sous forme de spans de texte.

---

## Modèles étudiés

Les modèles suivants sont fine-tunés et comparés :

- **DistilBERT (distilbert-base-uncased)**  
  Modèle compact et rapide, utilisé comme baseline.
- **BERT (bert-base-uncased)**  
  Modèle de référence pour les tâches de NLP.
- **RoBERTa (roberta-base)**  
  Variante optimisée de BERT, souvent plus performante mais plus coûteuse.

Tous les modèles sont utilisés dans un cadre de **Question Answering extractif** (encoder-only).

---

## Méthodologie

La méthodologie suivie est directement inspirée du pipeline de fine-tuning présenté en cours :

1. Chargement du dataset SQuAD
2. Analyse exploratoire des données (longueurs, distributions, cas difficiles)
3. Tokenisation des paires (question, contexte)
4. Alignement des réponses (caractère → token)
5. Fine-tuning des modèles à l’aide de l’API `Trainer`
6. Évaluation avec les métriques Exact Match et F1
7. Comparaison des modèles

Les choix d’hyperparamètres (longueur maximale, stride, batch size) sont justifiés à partir de l’analyse exploratoire.

---

## Arborescence du projet

```text
qa-finetuning-squad-webapp/
│
├── app/                  # Interface applicative
│
├── data/
│   ├── raw/              # Données brutes 
│   └── processed/        # Données prétraitées / exportées
│
├── deployment/
│   └── hf_spaces/        # Scripts ou configurations pour Hugging Face Spaces
│
├── models/
│   ├── distilbert_squad/ # Modèles fine-tunés + résultats
│   ├── bert_squad/
│   └── roberta_squad/
│
├── notebooks/
│   ├── 01_exploration_donnees.ipynb
│   ├── 02_finetuning_distilbert.ipynb
│   ├── 03_finetuning_bert.ipynb
│   ├── 04_finetuning_roberta.ipynb
│   └── 05_comparaison_modeles.ipynb
│
├── scripts/              # Scripts utilitaires
│
├── requirements.txt      # Dépendances Python
├── README.md             # Documentation du projet
└── .venv/                # Environnement virtuel
