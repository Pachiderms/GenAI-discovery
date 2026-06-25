# Sovereign-RAG

## Description

Ce projet met en œuvre une plateforme souveraine d'IA générative permettant l'exploitation de modèles de langage open-source en environnement local, sans dépendance à des services cloud tiers.

L'objectif principal est de répondre aux enjeux de souveraineté numérique, de confidentialité des données et de maîtrise des infrastructures IA au sein des organisations publiques et privées.

L'ensemble des traitements (indexation documentaire, génération de texte, reconnaissance vocale et recherche sémantique) est exécuté localement afin de garantir qu'aucune donnée sensible ne quitte l'environnement de l'utilisateur.

## Setup
Un fichier pyproject.toml est présent a la racine du dossier. Vous pouvez l'utiliser pour configurer un environnement virutel minimal pour cette étape.

```bash
uv venv
uv sync
```

Vous aurez ensuite besoin de télécharger le modèle pré-entrainé pour le nlp

```bash
uv pip install fr_core_news_sm
```

### Principaux objectifs

* Déploiement local de modèles d'IA générative open-source.
* Conservation intégrale des données sur l'infrastructure de l'organisation.
* Recherche documentaire augmentée (RAG) sur des bases de connaissances privées.
* Reconnaissance vocale locale sans recours à des services externes.
* Architecture extensible pour les usages métier et industriels.
* Réduction de la dépendance aux fournisseurs de services IA propriétaires.

---

# Architecture Technique

## IA Générative Locale

Le projet s'appuie sur Ollama pour l'exécution locale de modèles de langage open-source.

### Modèles utilisés

* Mistral 7B
* Nomic Embed Text

### Avantages

* Aucune transmission de données vers une API externe.
* Contrôle complet du cycle de vie des données.
* Réduction des risques liés à la conformité réglementaire.
* Déploiement possible sur des infrastructures isolées (air-gapped).

---

# Pipeline RAG (Retrieval-Augmented Generation)

Le système repose sur une architecture RAG permettant d'enrichir les réponses du modèle avec des informations issues de documents internes.

## Chargement documentaire

Le système supporte :

* PDF
* DOCX
* TXT

Les documents sont automatiquement :

1. Chargés.
2. Découpés en chunks.
3. Vectorisés.
4. Stockés dans une base vectorielle persistante.

## Base vectorielle

### ChromaDB

Les embeddings sont stockés localement dans ChromaDB afin de :

* Conserver les index entre les redémarrages.
* Réduire les temps de recalcul.
* Faciliter l'ajout incrémental de nouveaux documents.

---

# Recherche Sémantique Avancée

## Embeddings

Le projet utilise :

```bash
nomic-embed-text
```

pour générer des représentations vectorielles des documents.

## Retrieval MMR

La recherche documentaire repose sur la méthode :

```python
retriever = collection.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 10,
        "fetch_k": 20,
        "lambda_mult": 0.25,
        "score_threshold": 0.2
    }
)
```

### Pourquoi le MMR ?

Le Maximal Marginal Relevance permet :

* d'éviter les résultats redondants ;
* de favoriser la diversité documentaire ;
* d'améliorer la qualité du contexte envoyé au LLM.

Cette approche devient particulièrement pertinente lorsque la base documentaire grandit.

---

# Enrichissement NER (Named Entity Recognition)

Afin d'améliorer la pertinence du retrieval, le système exploite également la reconnaissance d'entités nommées via SpaCy.

## Entités extraites

* Personnes
* Organisations
* Dates
* Lieux
* Produits
* Évènements

Exemple :

```text
Emmanuel Macron
Assemblée Nationale
2024
Paris
```

## Indexation des métadonnées

Chaque chunk est enrichi avec les entités détectées :

```python
{
    "people": [...],
    "organizations": [...],
    "dates": [...],
    "location": [...]
}
```

## Reranking contextuel

Lors d'une requête utilisateur :

1. Les entités de la question sont extraites.
2. Une recherche vectorielle est effectuée.
3. Les résultats sont rerankés selon la présence des entités recherchées.
4. Les chunks les plus pertinents sont placés en tête du contexte envoyé au modèle.

Cette approche hybride combine :

* pertinence sémantique ;
* pertinence métier ;
* réduction du bruit documentaire.

---

# Reconnaissance Vocale Locale

Le projet intègre un système de reconnaissance vocale entièrement local.

## Technologies

### Vosk

* Modèle français embarqué.
* Fonctionnement hors ligne.
* Aucune fuite de données vocales.

### SpeechRecognition

Gestion du microphone et de la capture audio.

### PyAudio

Accès bas niveau aux périphériques audio.

---

# Pipeline Conversationnel

## Fonctionnement

1. L'utilisateur pose une question par texte ou voix.
2. Le système reformule et enrichit la requête.
3. Les entités importantes sont extraites.
4. Une recherche vectorielle est lancée dans ChromaDB.
5. Les documents sont rerankés grâce au NER.
6. Le contexte est injecté dans le prompt.
7. Mistral génère une réponse contextualisée.
8. L'historique conversationnel est conservé.

```text
Utilisateur
      ↓
Prétraitement NLP
      ↓
Extraction NER
      ↓
Recherche MMR ChromaDB
      ↓
Reranking NER
      ↓
Construction du contexte
      ↓
Mistral 7B
      ↓
Réponse
```

---

# Souveraineté des Données

## Pourquoi cette architecture est souveraine ?

Contrairement à une architecture SaaS classique, l'ensemble des traitements reste hébergé localement.

| Fonction              | Solution utilisée | Exécution |
| --------------------- | ----------------- | --------- |
| LLM                   | Mistral 7B        | Local     |
| Embeddings            | Nomic Embed Text  | Local     |
| Vector Store          | ChromaDB          | Local     |
| Reconnaissance vocale | Vosk              | Local     |
| Pipeline NLP          | SpaCy             | Local     |

## Aucune dépendance Cloud

Aucune donnée n'est :

* envoyée à OpenAI ;
* envoyée à Google ;
* envoyée à Microsoft ;
* envoyée à Anthropic ;
* stockée sur un service tiers.

Les documents restent sous le contrôle exclusif de l'organisation.

## Conformité et sécurité

Cette architecture répond aux besoins :

* d'administration publique ;
* d'industrie ;
* de défense ;
* de santé ;
* de recherche.

Elle facilite également la mise en conformité avec :

* le RGPD ;
* les politiques internes de sécurité ;
* les exigences de souveraineté numérique.

---

# Performances

## Optimisation des embeddings

La persistance de ChromaDB permet :

* d'éviter la réindexation complète ;
* de réduire les temps de démarrage ;
* d'améliorer l'expérience utilisateur.

## Gestion incrémentale

Lorsqu'un nouveau document est détecté :

1. Le fichier est chargé.
2. Les embeddings sont calculés uniquement pour ce document.
3. Les métadonnées NER sont générées.
4. Les données sont ajoutées à la collection existante.

Cette approche réduit fortement le coût de maintenance de la base documentaire.

---

# Cas d'Usage

## Assistant documentaire interne

* procédures internes ;
* documentation technique ;
* rapports d'audit ;
* documentation qualité.

## Base de connaissances souveraine

* collectivités ;
* administrations ;
* établissements de santé ;
* laboratoires de recherche.

## Assistant métier

* support technique ;
* assistance réglementaire ;
* consultation documentaire spécialisée.

---

# Technologies

## IA Générative

* Ollama
* Mistral 7B
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

## Langage

* Python 3

---

# Avantages et Limites

| Avantages                               | Limites                                        |
| --------------------------------------- | ---------------------------------------------- |
| Souveraineté complète des données       | Nécessite une machine locale adaptée           |
| Aucune facturation API                  | Temps d'inférence dépend du matériel           |
| Confidentialité maximale                | Modèles plus petits que certaines offres cloud |
| Recherche documentaire enrichie par NER | Maintenance locale des modèles                 |
| Fonctionnement hors ligne possible      | Mise à jour manuelle des composants            |
| Architecture extensible                 | Consommation GPU selon le modèle utilisé       |

---

# Conclusion

Cette plateforme démontre qu'il est possible de construire une solution complète d'IA générative souveraine combinant modèles open-source, recherche documentaire augmentée, reconnaissance vocale locale et enrichissement sémantique par NER.

L'approche retenue permet d'obtenir un équilibre entre performance, confidentialité et indépendance technologique tout en garantissant que les données stratégiques demeurent sous le contrôle exclusif de l'organisation.

## Sources

* https://ollama.com/
* https://www.trychroma.com/
* https://docs.langchain.com/
* https://spacy.io/
* https://alphacephei.com/vosk/
* https://pypi.org/project/SpeechRecognition/
* https://people.csail.mit.edu/hubert/pyaudio/
* https://spacy.io/api/entityrecognizer
