# Application Web Fullstack

## Architecture Générale
Déploiement d'une application web complète pour interagir avec le système de RAG local. L'architecture suit un modèle client-serveur avec séparation claire entre le frontend (interface utilisateur) et le backend (logique métier et intégrations).

## 1. Backend Python (FastAPI)

### Composants Principaux
- **main.py** : Serveur FastAPI exposant les endpoints de l'API REST pour la communication avec le frontend.
- **rag_logic.py** : Implémentation du pipeline RAG utilisant LangChain et Ollama (Mistral 7B) avec ChromaDB pour la récupération d'informations contextualisées.
- **file_handler.py** : Gestion des documents (upload, traitement et indexation) et persistance de la base de données vectorielle.
- **audio_manager.py** : Transcription audio en texte via Vosk avec support du français, permettant une interaction vocale avec le système.
- **prompt_templates.py** : Définition des prompts système optimisés pour les tâches de génération et de réponse contextuelle.

### Fonctionnalités Clés
- API REST pour le traitement des requêtes textelles et la gestion de l'historique conversationnel.
- Transcription audio en temps réel avec reconnaissance vocale (Vosk).
- Intégration transparente du RAG avec contexte conversationnel persistant.
- Support CORS pour l'intégration frontend.
- Gestion asynchrone des requêtes pour optimiser la performance.

## 2. Frontend Web

### Composants
- **index.html** : Structure HTML de l'interface utilisateur.
- **app.js** : Logique JavaScript pour la communication avec l'API, gestion de l'interface et interaction utilisateur.
- **style.css** : Styles CSS pour une interface moderne et réactive.
- **app.conf** : Configuration de l'application web.

### Fonctionnalités
- Interface conviviale pour soumettre des questions et recevoir des réponses augmentées par RAG.
- Intégration avec la transcription audio pour une interaction vocale fluide.
- Affichage de l'historique conversationnel.
- Gestion dynamique du contexte utilisateur.

## 3. Déploiement et Orchestration

### Docker et Compose
- **docker-compose.yml** : Orchestration des services (backend, frontend, base de données) dans des conteneurs isolés avec support GPU et volumes persistants.
- **Dockerfile** (backend et frontend) : Images contenerisées pour déploiement consistent.
- **Makefile** : Automatisation des commandes courantes (build, run, clean, logs).
- **pyproject.toml** : Gestion des dépendances Python du projet.

### Avantages
- Isolation complète des services pour une meilleure maintenabilité.
- Support GPU pour accélération des modèles d'IA.
- Déploiement reproducible et facilité de scalabilité.

## 4. Pipeline Complet

Le flux d'interaction utilisateur :
1. Utilisateur soumet une question (textuelle ou vocale).
2. Frontend envoie la requête au backend via API REST.
3. Backend transcrit l'audio si nécessaire (AudioManager).
4. RAG Logic récupère les documents pertinents (ChromaDB).
5. LLM (Mistral 7B) génère une réponse contextualisée.
6. Réponse retournée au frontend avec historique mis à jour.

## <u>**Showcase**
![](images/web00.png)
![](images/web01.png)
![](images/web02.png)
![](images/web03.png)

## Axes d'amélioration
* Diminuer le temps warmup qui est trop long pour un utilisateur réel
* Diminuer le temps de réponse

## Sources

- [MediaStreamRecordingAPI](https://developer.mozilla.org/en-US/docs/Web/API/MediaStream_Recording_API/Using_the_MediaStream_Recording_API)
- [Vosk .wav To text](https://github.com/andrewymin/audio-to-text/blob/master/transcribe.py)
- [Docker GPU support](https://docs.docker.com/compose/how-tos/gpu-support/)