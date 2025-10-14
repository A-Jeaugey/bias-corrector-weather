# 🌦️ Bias-Corrector Weather — Prédictions météo locales corrigées automatiquement

![Daily Weather Pipeline](https://github.com/A-Jeaugey/bias-corrector-weather/actions/workflows/daily.yml/badge.svg)

Ce projet corrige automatiquement les **prévisions météo J+1** d’une API publique (Open-Meteo) pour une localisation donnée, en apprenant à partir des **erreurs passées** grâce à “HistGradientBoostingRegressor (sklearn)”.  
Le pipeline tourne chaque jour grâce à GitHub Actions et met à jour les données et les modèles sans intervention manuelle.

---

## ✨ Fonctionnalités

- 📥 Téléchargement automatique des **prévisions J+1** chaque soir  
- 🌡 Récupération automatique des **observations réelles** via Meteostat le lendemain  
- 🧠 Réentraînement quotidien d’un **modèle HGB** pour corriger le biais local  
- 📈 Évaluation rapide sur les 15 derniers jours simulés  
- 🔮 Prédiction corrigée publiée dans `last_prediction.json`  
- ☁️ Automatisation complète via GitHub Actions (aucun PC à laisser allumé)  
- ⚡ Seed initial avec 3 ans d’historique Meteostat pour commencer instantanément

---

## 🗂️ Structure du projet

```
bias-corrector-weather/
├─ data/
│  ├─ forecasts.csv         # prévisions brutes historiques et quotidiennes
│  └─ observations.csv      # observations réelles
├─ models/
│  ├─ HGB_tmax.joblib     # modèle correction Tmax
│  └─ HGB_tmin.joblib     # modèle correction Tmin
├─ src/
│  ├─ config.py             # coordonnées, timezone, chemins
│  ├─ seed_history.py       # seed 3 ans d'historique Meteostat
│  ├─ fetch_forecast.py     # prévision J+1 quotidienne
│  ├─ fetch_obs.py          # observation J-1 quotidienne
│  ├─ features.py           # génération des features saisonnières
│  ├─ train.py              # entraînement HGB sur erreurs
│  └─ predict.py            # prévision corrigée J+1
├─ .github/workflows/
│  └─ daily.yml             # automatisation GitHub Actions (2 runs/jour)
├─ README.md
├─ requirements.txt
└─ .gitignore
```

---

## 🚀 Installation locale

1. Clone le repo :  
```
git clone https://github.com/A-Jeaugey/bias-corrector-weather.git
cd bias-corrector-weather
```

2. Crée un environnement virtuel :  
```
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

3. Installe les dépendances :  
```
pip install -r requirements.txt
```

---

## 🧠 Seed initial (3 ans d'historique)

Avant de lancer la machine, on génère un historique directement :  
```
python src/seed_history.py
```

Ce script :
- Télécharge 3 ans d’archives Meteostat (prévisions proxy)  
- Télécharge 3 ans d’observations Meteostat  
- Remplit `data/forecasts.csv` et `data/observations.csv`

---

## 🛠️ Entraînement + prédiction locale

```
# Entraîne le modèle HGB sur tout l'historique
python src/train.py

# Prédit la prévision corrigée pour demain
python src/predict.py
```

Exemple de sortie :  
```
{
  "date": "2025-10-15",
  "tmax_prev": 22.3,
  "tmax_corr": 23.1,
  "tmin_prev": 12.1,
  "tmin_corr": 12.8
}
```

---

## 🌀 Pipeline quotidien automatisé (GitHub Actions)

Le fichier `.github/workflows/daily.yml` automatise le tout :

| Heure Paris | Étape                                 | Script(s) lancé(s)                       |
|------------|---------------------------------------|------------------------------------------|
| 18:05      | Récupération de la prévision J+1      | `fetch_forecast.py`                      |
| 23:30      | Observation + réentraînement + prédiction | `fetch_obs.py` + `train.py` + `predict.py` |

Les fichiers modifiés (`data/`, `models/`, `last_prediction.json`) sont automatiquement commit & push par le bot GitHub.

---

## 🌍 Changer de localisation

Modifie simplement `LAT`, `LON` et `TIMEZONE` dans `src/config.py`, puis relance le seed :  
```
python src/seed_history.py
python src/train.py
```

⚠️ Si ta nouvelle zone n’a pas de station Meteostat locale, il se peut que le seed retourne 0 ligne.

---

## 📊 Données stockées

- **`data/forecasts.csv`**  
  prévisions brutes quotidiennes (API Open-Meteo)

- **`data/observations.csv`**  
  températures max/min et précipitations réelles (Meteostat)

- **`models/*.joblib`**  
  modèles HGB réentraînés quotidiennement

- **`last_prediction.json`**  
  dernière prévision corrigée, générée chaque nuit par la CI

---

## 📈 Évaluation rapide

À chaque entraînement, `train.py` affiche :
```
MAE(tmax_corr) = 1.25 °C sur les 15 derniers jours simulés.
MAE(tmin_corr) = 0.93 °C sur les 15 derniers jours simulés.
```

👉 MAE = erreur absolue moyenne.  
facilement comparable aux prévisions brutes pour voir le gain.

---

## 🧪 Tech rapide

- **Données** : Open-Meteo Forecast / Meteostat (lib Python officielle)  
- **Features** : saison encodée (`doy_sin`, `doy_cos`) + valeur prévue  
- **Modèle** : HistGradientBoostingRegressor (scikit-learn)
- **Automatisation** : GitHub Actions (2 crons UTC)  
- **Langage** : Python 3.11

---

## 🌟 Pistes d’amélioration

- Ajouter un graphe d’évolution MAE sur 30 jours (Matplotlib)  
- Entraîner des modèles quantiles (p10, p50, p90) pour afficher une **incertitude**  
- Corriger aussi pluie / vent avec des modèles séparés  
- Déployer une petite page web qui lit `last_prediction.json` 📊

---

## 📝 Licence

Projet libre à usage pédagogique et personnel.  
Sources de données :  
- [Open-Meteo](https://open-meteo.com/) (gratuite et sans clé)  
- [Meteostat](https://meteostat.net/) (libre & académique)

---

👉 En résumé :  
> Ce projet ne prédit pas la météo… il **corrige intelligemment** les prévisions existantes pour une ville, avec un pipeline 100 % automatisé 🌍⚡
