# Projet 1 : Prévision de Consommation Électrique

## Problématique générale
Prédire la consommation électrique horaire à horizon 24h pour optimiser la production énergétique et réduire les coûts opérationnels. Ce problème de prévision univariée/multivariée nécessite de capturer les patterns cycliques (journaliers, hebdomadaires, saisonniers) et l'influence de variables exogènes (température, jours fériés).

Tâche attendue : Régression - Prévision de séries temporelles multi-horizon
## Tâche attendue
- Régression : prévision de séries temporelles multi-horizon

Pistes d'Analyse
Analyse exploratoire des patterns temporels : saisonnalité, tendance, cycles journaliers/hebdomadaires
## Pistes d'analyse
- Analyse exploratoire des patterns temporels : saisonnalité, tendance, cycles journaliers/hebdomadaires
- Étude de corrélation entre consommation et variables météorologiques
- Comparaison de fenêtres temporelles d'historique (24h, 48h, 7 jours)
- Évaluation de l'impact des features calendaires (jour de la semaine, férié, vacances)
- Benchmarking LSTM vs TCN vs modèles statistiques (SARIMA)
- Analyse des erreurs de prédiction : identification des périodes problématiques (pics de consommation)
- Étude de l'horizon de prédiction optimal selon la métrique choisie
- Visualisation des prédictions multi-horizons avec intervalles de confiance

Étude de corrélation entre consommation et variables météorologiques
## Publications de référence
- Lai et al. (2018) — "Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks" (LSTNet)  
  https://arxiv.org/abs/1703.07015
- Oreshkin et al. (2020) — "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting"  
  https://arxiv.org/abs/1905.10437

Comparaison de fenêtres temporelles d'historique (24h, 48h, 7 jours)
## Sources de données
- UCI Individual Household Electric Power Consumption  
  https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption  
  2,075,259 mesures (déc 2006 - nov 2010), résolution 1 minute, variables : puissance active/réactive, voltage, intensité, sous-compteurs
- Alternative : OpenPowerSystem Data  
  https://open-power-system-data.org/

Évaluation de l'impact des features calendaires (jour de la semaine, férié, vacances)

Benchmarking LSTM vs TCN vs modèles statistiques (SARIMA)

Analyse des erreurs de prédiction : identification des périodes problématiques (pics de consommation)

Étude de l'horizon de prédiction optimal selon la métrique choisie

Visualisation des prédictions multi-horizons avec intervalles de confiance
## Organisation du projet
- README.md : description du projet, instructions d'installation et d'exécution
- requirements.txt : dépendances Python
- scripts/download_data.py : script de téléchargement automatique des données
- notebooks/demo.ipynb : notebook de démonstration avec analyses et visualisations
- src/ : code source organisé  
  - data_processing.py : prétraitement et features engineering  
  - models.py : architectures des modèles  
  - train.py : entraînement  
  - evaluate.py : évaluation et métriques  
  - utils.py : fonctions utilitaires
- data/ : données brutes et prétraitées (gitignored)
- results/ : modèles sauvegardés, graphiques, métriques
- .gitignore
- setup_env.sh : script de création de l'environnement virtuel

Publications de Référence
Lai et al. (2018) - "Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks" (LSTNet)
## Jalons
- Point mi-parcours (4h30) : présentation de l'analyse exploratoire, baseline implémentée, premiers résultats LSTM, axes d'amélioration identifiés
- Présentation finale (15 min + démo) : résultats comparatifs, analyse approfondie des performances, démonstration live du notebook avec prédictions commentées

https://arxiv.org/abs/1703.07015

Oreshkin et al. (2020) - "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting"

https://arxiv.org/abs/1905.10437

Sources de Données
UCI Individual Household Electric Power Consumption

https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

2,075,259 mesures (déc 2006 - nov 2010)

Résolution : 1 minute

Variables : puissance active/réactive, voltage, intensité, sous-compteurs

Alternative : OpenPowerSystem Data

https://open-power-system-data.org/

Organisation du Projet
Structure du Repository GitHub
README.md : description du projet, instructions d'installation et d'exécution

requirements.txt : dépendances Python

scripts/download_data.py : script de téléchargement automatique des données

notebooks/demo.ipynb : notebook de démonstration avec analyses et visualisations

src/ : code source organisé

data_processing.py : prétraitement et features engineering

models.py : architectures des modèles

train.py : entraînement

evaluate.py : évaluation et métriques

utils.py : fonctions utilitaires

data/ : données brutes et prétraitées (gitignored)

results/ : modèles sauvegardés, graphiques, métriques

.gitignore

setup_env.sh : script de création de l'environnement virtuel

Jalons
Point mi-parcours (4h30) : présentation de l'analyse exploratoire, baseline implémentée, premiers résultats LSTM, axes d'amélioration identifiés

Présentation finale (15 min + démo) : résultats comparatifs, analyse approfondie des performances, démonstration live du notebook avec prédictions commentées

Bonnes Pratiques à Respecter
Problématique : définir clairement la tâche (prévision à 24h), les métriques (MAE, RMSE, MAPE), le protocole d'évaluation (train/val/test temporel)

Visualisation : séries temporelles brutes, distributions, décomposition saisonnière, matrice de corrélation

Tableau descriptif : dimensions (nombre d'observations, features), nature (numérique continu), fréquence d'échantillonnage, taux de valeurs manquantes

Code structuré : séparation données/modèles/entraînement, commentaires explicatifs, docstrings

Reproductibilité : environnement virtuel venv, requirements.txt, seed fixé, script d'installation automatique

Baseline : modèle naïf (persistance, moyenne mobile) ou SARIMA pour établir une référence

Méthodes avancées : LSTM multivarié, TCN, comparaison systématique avec ablation studies
## Bonnes pratiques à respecter
- Problématique : définir clairement la tâche (prévision à 24h), les métriques (MAE, RMSE, MAPE), le protocole d'évaluation (train/val/test temporel)
- Visualisation : séries temporelles brutes, distributions, décomposition saisonnière, matrice de corrélation
- Tableau descriptif : dimensions (nombre d'observations, features), nature (numérique continu), fréquence d'échantillonnage, taux de valeurs manquantes
- Code structuré : séparation données/modèles/entraînement, commentaires explicatifs, docstrings
- Reproductibilité : environnement virtuel venv, requirements.txt, seed fixé, script d'installation automatique
- Baseline : modèle naïf (persistance, moyenne mobile) ou SARIMA pour établir une référence
- Méthodes avancées : LSTM multivarié, TCN, comparaison systématique avec ablation studies
