# GenAI-discovery (1 week challenge)
## Description
Ce projet met en place un environnement souverain de test complet pour l'expérimentation de modèles d'IA générative en local tout en garantissant la confidentialité absolue des données.
- Infrastructure : Déploiement de modèles open-source via Ollama (Mistral:7B, Phi3) en local.
- RAG (Retrieval-Augmented Generation): Développement d'un script Python utilisant LangChain et ChromaDB pour indexer et interroger des documents PDF privés.
- Optimisation : Mise en place d'une persistance de base de données vectorielle pour éviter la redondance des calculs d'embeddings.
- Benchmarking : Analyse comparative des performances de génération (vitesse en tokens/sec vs précision du raisonnement).
## Tests
### 1. Benchmark -> ./benchmark
![](images/benchmark_m_phi%20.png)
On remarque une différence très marquée entre les deux modèles notamment sur le nombre de tokens/sec avec 90.30 pour mistral contre 151.92 pour phi3. Cela s'explique par le fait que mistral est un modèle à 7,3 millards de paramètres contre 3,8 milliards pour phi3.

En termes d'usage des ressources de mon GPU, l'usage de la VRAM est identique pour les deux modèles avec une horloge qui plafone à 9846MHz. Par contre, pour l'usage du processuer graphique le modèle mistral plaonne a 98% d'utlisation des ressources contre 92% pour phi3.

Phi3 de par son nombre de paramètres reduit à une efficacité énergétique bien supérieure à celle de mistral car il nécessite moins de calcul pour générer une réponse. Cela peut-être considéré comme un avantage pour une entreprise cherchant à faire des économies. Cela permet aussi à phi3 de pouvoir fonctionner sur des appareils plus limités en ressources comme un smartphone par exemple.

## 2. Efficacité energetique vs performance
![](images/mistral7b_vs_phi3b.png)
- Comme on peut le voir si dessus, mistral cherche une approche mathématiquement correcte et explique de façon structurée et claire. Il a une approche pédagogique comme on pourrait s'y attendre pour un élève de primaire.
- phi3 lui, ne remet pas en question l'assertion de base et se permet de rajouter du contexte pour justifier l'erreur. Le modèle construit des phrases grammaticalement incorrectes (surrement du à une traduction compliquée pour le modèle) et/ou difficles à comprendre pour un élève dde primaire.
De plus, j'ai essayé de faire comprendre au modèle qu'il avait tort mais il est incapable de le reconnaître et se contente de reformuler sa réponse. Après plusieurs tentatives, le modèle étant incapable de répondre génère une réponse hors sujet de plus de 100 lignes et bascule en anglais pour la réponse.

## Conclusion, Résumé
|        | Avantages | Inconvénients |
|:------ |:---------:|:-------------:|
| Mistral | reponses stables, pertinentes, Connaissances gloabale accrue  (langues, logique...) | Plus demandeur en ressources, plus energivore
| Phi3    | Moindre coût et taille, Rapport taille/pertinence des réponses ok, protabilité | Connaissances gloabale moyenne (langues, logique...), style verbeux et répétitif, erreurs si logique complexe dans la requete

## IA Generative locale (RAG) -> ./rag/
### 1. Déploiement d'un Pipeline RAG
Mise en place d'un système capable d'extraire des informations spécifiques depuis des documents volumineux avec langchain et ChromaDB.
### 2. Persistence des données
Optimisation de la base de données pour éviter la redondance d'indexation (Logique de stockage local persistant).
### 3. Protection des données
Approche locale sans appels à une API tierce.
J'ai favorisé l'usage d'un modèle Européen: Mistral, acteur qui respecte les standards de transparence

## Sources
- [HuggingFace](https://huggingface.co/)
- [Ollama](https://ollama.com/)
- [ChromaDB](https://docs.trychroma.com/docs/)
- [Langchain](https://reference.langchain.com/)
- [SpeechRecognition](https://pypi.org/project/SpeechRecognition/)
- [Vosk](https://alphacephei.com/vosk/)
- [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/docs/)
