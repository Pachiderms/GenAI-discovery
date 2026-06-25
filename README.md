# GenAI Discovery

## Description

Ce projet a pour objectif d'explorer les principaux concepts de l'IA générative moderne à travers une série d'expérimentations progressives réalisées dans un environnement entièrement local et souverain.

L'objectif n'est pas uniquement de déployer un modèle de langage, mais de comprendre les problématiques associées à l'utilisation de l'IA générative en entreprise :

* performances des modèles ;
* consommation de ressources ;
* souveraineté des données ;
* qualité des réponses ;
* recherche documentaire augmentée (RAG) ;
* interaction utilisateur ;
* intégration dans une application complète.

Le projet a évolué selon trois grandes étapes :

1. Évaluation et comparaison de modèles de langage open-source.
2. Construction d'un système RAG souverain capable d'interroger une base documentaire privée.
3. Développement d'une application web complète permettant d'interagir avec le système.

❕❕ Pour tester les parties 2 et 3 du projet, une archive zip avec des données de test est prévue

```bash
db.zip
```

A extraire 2 fois:
* ./rag/db pour l'étape 2
* ./app/backend/db pour l'étape 3

⚠️ Il y'a 100 fichiers présents dans l'archive, le premier chargement peut prendre un certain temps selon le materiel utilisé.

---

# Objectifs du Projet

## Découvrir l'IA générative

Comprendre :

* le fonctionnement des LLM ;
* les contraintes matérielles ;
* les compromis entre performance et coût ;
* les mécanismes de recherche augmentée.

## Explorer les enjeux de souveraineté

L'ensemble des expérimentations est réalisé localement grâce à des modèles open-source afin de :

* conserver les données sur l'infrastructure locale ;
* éviter l'utilisation d'API tierces ;
* garantir la confidentialité des informations manipulées.

## Construire progressivement une solution complète

Le projet suit une approche incrémentale :

```text
Benchmark LLM
      ↓
Analyse des performances
      ↓
Mise en place d'un RAG local
      ↓
Enrichissement NER
      ↓
Reconnaissance vocale
      ↓
Application Web Fullstack
```

---

# Architecture Globale

```text
                         ┌───────────────┐
                         │ Utilisateur   │
                         └───────┬───────┘
                                 │
                                 ▼
                     ┌──────────────────────┐
                     │ Interface Web        │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │ API FastAPI          │
                     └──────────┬───────────┘
                                │
             ┌──────────────────┼──────────────────┐
             │                  │                  │
             ▼                  ▼                  ▼
      Reconnaissance      Pipeline RAG        Historique
          Vocale             Local

             │                  │
             ▼                  ▼

            Vosk       ChromaDB + NER
                               │
                               ▼

                         Mistral 7B
                               │
                               ▼

                          Réponse
```

---

# Expérimentation 1 : Benchmark des LLM

## Objectif

Comparer différents modèles open-source afin d'évaluer :

* leur vitesse d'inférence ;
* leur consommation de ressources ;
* leur qualité de raisonnement ;
* leur efficacité énergétique.

## Modèles testés

* Mistral 7B
* Phi-3

## Résultats

### Performances

| Modèle     | Tokens/sec |
| ---------- | ---------: |
| Mistral 7B |      90.30 |
| Phi-3      |     151.92 |

Phi-3 se montre nettement plus rapide grâce à son nombre réduit de paramètres.

### Qualité des réponses

Les expérimentations montrent que :

* Mistral fournit des réponses plus stables et plus cohérentes ;
* Phi-3 est plus rapide mais peut rencontrer des difficultés sur les raisonnements complexes.

### Enseignements

Cette première étape met en évidence le compromis classique :

| Critère                  | Mistral         | Phi-3         |
| ------------------------ | --------------- | ------------- |
| Vitesse                  | Moyen           | Élevée        |
| Qualité du raisonnement  | Élevée          | Moyenne       |
| Consommation énergétique | Plus importante | Plus faible   |
| Taille du modèle         | Plus grande     | Plus compacte |

📁 Voir le détail dans :

```text
./benchmark
```

---

# Expérimentation 2 : IA Générative Souveraine et RAG

## Objectif

Permettre à un modèle de répondre à partir de documents privés sans transmettre les données à des services externes.

## Technologies

* Ollama
* Mistral 7B
* LangChain
* ChromaDB
* SpaCy
* Vosk
* SpeechRecognition

## Fonctionnalités

### Indexation documentaire

Support :

* PDF
* DOCX
* TXT

### Recherche sémantique

Utilisation :

* Embeddings Nomic
* ChromaDB
* MMR (Maximal Marginal Relevance)

### Enrichissement NER

Ajout d'une couche de compréhension documentaire grâce à :

* la détection de personnes ;
* la détection d'organisations ;
* la détection de dates ;
* la détection de lieux.

Les résultats du retrieval sont ensuite rerankés afin d'améliorer la pertinence du contexte envoyé au modèle.

### Souveraineté des données

L'ensemble du pipeline fonctionne localement :

| Composant        | Exécution |
| ---------------- | --------- |
| LLM              | Local     |
| Embeddings       | Local     |
| Base vectorielle | Local     |
| NLP              | Local     |
| Speech-To-Text   | Local     |

Aucune donnée n'est envoyée à un fournisseur tiers.

### Enseignements

Cette étape a permis de comprendre :

* les limites des LLM seuls
* l'intérêt du Retrieval-Augmented Generation
* l'importance de la qualité du retrieval
* l'apport du NER dans la recherche documentaire

📁 Voir le détail dans :

```text
./rag
```

---

# Expérimentation 3 : Application Web Fullstack

## Objectif

Transformer le prototype RAG en une application utilisable par un utilisateur final.

## Backend

Développement d'une API FastAPI assurant :

* la communication avec le frontend
* l'orchestration du pipeline RAG
* la gestion de l'historique
* le traitement audio

## Frontend

Développement d'une interface web permettant :

* la saisie de questions
* l'affichage des réponses
* l'interaction vocale
* la consultation de l'historique

## Reconnaissance Vocale

Ajout de :

* Vosk
* SpeechRecognition
* PyAudio

Le système est capable de recevoir une requête vocale et de l'intégrer directement dans le pipeline RAG.

### Enseignements

Cette dernière étape permet de comprendre :

* l'intégration d'un LLM dans une architecture applicative ;
* la gestion d'API IA ;
* l'expérience utilisateur autour d'un assistant IA ;
* les contraintes de déploiement d'une solution d'IA générative.

📁 Voir le détail dans :

```text
./app
```

---

# Technologies Utilisées

## IA Générative

* Ollama
* Mistral 7B
* Phi-3
* Nomic Embed Text

## RAG

* LangChain
* ChromaDB

## NLP

* SpaCy

## Audio

* Vosk
* SpeechRecognition
* PyAudio

## Backend

* Python
* FastAPI

## Frontend

* HTML
* CSS
* JavaScript

## Conteneurisation

* Docker
* Docker Compose

---

# Résultats et Enseignements

Au cours du projet, plusieurs constats ont émergé :

### Les modèles plus petits sont souvent suffisants

Phi-3 démontre qu'un modèle compact peut offrir de très bonnes performances tout en réduisant la consommation énergétique.

### Le RAG est indispensable pour les données métier

Les modèles seuls ne disposent pas de connaissances spécifiques à une organisation.

Le RAG permet d'intégrer ces connaissances de manière contrôlée.

### La qualité du retrieval est critique

L'ajout :

* du MMR ;
* du NER ;
* du reranking contextuel ;

améliore significativement la pertinence des réponses.

### La souveraineté est techniquement réalisable

Grâce aux modèles open-source et aux outils locaux, il est possible de construire une solution complète sans dépendre d'API externes et ce même chez soi avec un setup modeste !

---

# Conclusion

Ce projet constitue un parcours complet de découverte de l'IA générative moderne.

À travers le benchmarking de modèles, la construction d'un système RAG souverain puis le développement d'une application web complète, il permet d'explorer les principaux enjeux techniques, économiques et stratégiques liés à l'intégration des LLM dans un contexte professionnel.

L'ensemble des expérimentations démontre qu'il est aujourd'hui possible de déployer une solution d'IA générative performante, extensible et respectueuse de la souveraineté des données en s'appuyant exclusivement sur des technologies open-source.
