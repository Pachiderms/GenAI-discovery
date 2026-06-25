# Tests && Benchmarks

## Setup
Un fichier pyproject.toml est présent a la racine du dossier. Vous pouvez l'utiliser pour configurer un environnement virutel minimal pour cette étape.

```bash
uv venv
uv sync
```

## 1. Benchmark -> ./benchmark
![](images/benchmark_m_phi%20.png)
On remarque une différence très marquée entre les deux modèles notamment sur le nombre de tokens/sec avec 90.30 pour mistral contre 151.92 pour phi3. Cela s'explique par le fait que mistral est un modèle à 7,3 millards de paramètres contre 3,8 milliards pour phi3.

En termes d'usage des ressources de mon GPU, l'usage de la VRAM est identique pour les deux modèles avec une horloge qui plafone à 9846MHz. Par contre, pour l'usage du processuer graphique le modèle mistral plaonne a 98% d'utlisation des ressources contre 92% pour phi3.

Phi3 de par son nombre de paramètres reduit à une efficacité énergétique bien supérieure à celle de mistral car il nécessite moins de calcul pour générer une réponse. Cela peut-être considéré comme un avantage pour une entreprise cherchant à faire des économies. Cela permet aussi à phi3 de pouvoir fonctionner sur des appareils plus limités en ressources comme un smartphone par exemple.

# 2. Efficacité energetique vs performance
![](images/mistral7b_vs_phi3b.png)
- Comme on peut le voir si dessus, mistral cherche une approche mathématiquement correcte et explique de façon structurée et claire. Il a une approche pédagogique comme on pourrait s'y attendre pour un élève de primaire.
- phi3 lui, ne remet pas en question l'assertion de base et se permet de rajouter du contexte pour justifier l'erreur. Le modèle construit des phrases grammaticalement incorrectes (surrement du à une traduction compliquée pour le modèle) et/ou difficles à comprendre pour un élève dde primaire.
De plus, j'ai essayé de faire comprendre au modèle qu'il avait tort mais il est incapable de le reconnaître et se contente de reformuler sa réponse. Après plusieurs tentatives, le modèle étant incapable de répondre génère une réponse hors sujet de plus de 100 lignes et bascule en anglais pour la réponse.

# Conclusion, Résumé
|        | Avantages | Inconvénients |
|:------ |:---------:|:-------------:|
| Mistral | reponses stables, pertinentes, Connaissances gloabales accrues  (langues, logique...) | Plus demandeur en ressources, plus energivore
| Phi3    | Moindre coût et taille, Rapport taille/pertinence des réponses ok, portabilité | Connaissances gloabales moyennes (langues, logique...), style verbeux et répétitif, erreurs si logique complexe dans la requete

## Sources

* https://ollama.com/