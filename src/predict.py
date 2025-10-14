import json
import numpy as np
import pandas as pd
from joblib import load
from config import FORECASTS_CSV


def construire_variables_explicatives(ligne: pd.Series, variable_cible: str) -> pd.DataFrame:
    """
    Construit une ligne de variables explicatives (features) pour le modèle de correction HGB,
    en respectant exactement le même format que lors de l'entraînement :

      - {variable}_prev : valeur prévue brute
      - doy_sin, doy_cos : saisonnalité (jour de l'année)
      - prcp_prev, ws_prev : pluie et vent
      - dow_1..dow_6 : jours de la semaine encodés en one-hot (dow_0 = baseline)
    """
    date_du_jour = pd.to_datetime(ligne["date"])
    jour_annee = date_du_jour.dayofyear
    saison_sin = np.sin(2 * np.pi * jour_annee / 365)
    saison_cos = np.cos(2 * np.pi * jour_annee / 365)

    # Jour de la semaine : 0 = lundi, 6 = dimanche
    jour_semaine = date_du_jour.weekday()
    colonnes_jour = {f"dow_{k}": 0 for k in range(1, 7)}
    if jour_semaine != 0:
        colonnes_jour[f"dow_{jour_semaine}"] = 1

    valeurs = {
        f"{variable_cible}_prev": float(ligne[f"{variable_cible}_prev"]),
        "doy_sin": float(saison_sin),
        "doy_cos": float(saison_cos),
        "prcp_prev": float(ligne.get("prcp_prev", 0.0)),
        "ws_prev": float(ligne.get("ws_prev", 0.0)),
        **colonnes_jour
    }

    colonnes = [
        f"{variable_cible}_prev",
        "doy_sin", "doy_cos",
        "prcp_prev", "ws_prev"
    ] + list(colonnes_jour.keys())

    return pd.DataFrame([valeurs], columns=colonnes)


def main():
    # Chargement du fichier des prévisions
    tableau_previsions = pd.read_csv(FORECASTS_CSV).sort_values("date")
    if tableau_previsions.empty:
        raise RuntimeError("Le fichier forecasts.csv est vide, aucune prédiction possible.")

    # Sélection de la dernière ligne (prévision la plus récente)
    derniere_ligne = tableau_previsions.iloc[-1]

    # Chargement des modèles de correction
    modele_tmax = load("models/hgb_tmax.joblib")
    modele_tmin = load("models/hgb_tmin.joblib")

    # Construction des variables explicatives pour chaque modèle
    variables_tmax = construire_variables_explicatives(derniere_ligne, "tmax")
    variables_tmin = construire_variables_explicatives(derniere_ligne, "tmin")

    # Prédiction des corrections
    correction_tmax = float(modele_tmax.predict(variables_tmax)[0])
    correction_tmin = float(modele_tmin.predict(variables_tmin)[0])

    # Génération de la sortie finale au format JSON
    sortie = {
        "date": str(derniere_ligne["date"]),
        "tmax_prev": round(float(derniere_ligne["tmax_prev"]), 1),
        "tmax_corr": round(float(derniere_ligne["tmax_prev"]) + correction_tmax, 1),
        "tmin_prev": round(float(derniere_ligne["tmin_prev"]), 1),
        "tmin_corr": round(float(derniere_ligne["tmin_prev"]) + correction_tmin, 1),
        "modeles_utilises": {
            "tmax": "hgb_tmax.joblib",
            "tmin": "hgb_tmin.joblib"
        }
    }

    print(json.dumps(sortie, ensure_ascii=False))


if __name__ == "__main__":
    main()
