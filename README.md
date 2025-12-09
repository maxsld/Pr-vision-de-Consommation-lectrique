# Prévision de Consommation Électrique

Projet de prévision de la consommation électrique horaire à 24h pour optimiser la production et réduire les coûts, à partir de données historiques, météo et calendaires.

## Contenu
- `sujet.md` : cadrage du projet.
- `notebooks/rapport.ipynb` : version notebook du cadrage/rapport.
- `scripts/download_data.py` : téléchargement et préparation des données (à compléter).
- `src/` : code source (prétraitement, modèles, entraînement, évaluation).
- `data/` : données brutes et prétraitées (ignorées par git).
- `results/` : artefacts de modèles, graphiques, métriques (ignorés par git).

## Données
- OPSD Time Series (charge/énergie, horaire) téléchargée sous `data/consumption_data.csv`. Dataset brut à filtrer/adapter selon le pays ciblé.
- Les dossiers `data/` et `results/` sont ignorés par git ; prévois du stockage local suffisant.
- Les dossiers `data/` et `results/` sont ignorés par git ; prévois du stockage local suffisant.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate   # Windows : .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Lancer les notebooks
```bash
jupyter notebook notebooks/rapport.ipynb
```

## Prochaines étapes suggérées
- Compléter `scripts/download_data.py` pour récupérer OPSD Time Series (charge, vent, solaire) et/ou météo.
- Implémenter le prétraitement dans `src/data_processing.py` (agrégation horaire, features calendrier/météo, split temporel).
- Ajouter une baseline (persistance/moyenne mobile) et un modèle LSTM/TCN dans `src/models.py` et `src/train.py`.
- Rédiger l’évaluation (MAE/RMSE/MAPE, courbes de prédiction) dans `src/evaluate.py` et `results/`.
