import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

from config import FORECASTS_CSV


CHEMIN_DERNIER_JSON = Path("last_prediction.json")     # compatibilité CI
CHEMIN_HISTORIQUE_CSV = Path("data/predictions.csv")   # historique cumulatif


def construire_variables_explicatives(ligne_prevision: pd.Series, variable_cible: str) -> pd.DataFrame:
    """
    Construit une ligne de variables explicatives pour le modèle de correction :
      - {variable}_prev : valeur prévue brute
      - doy_sin, doy_cos : saisonnalité (jour de l'année)
      - prcp_prev, ws_prev : pluie et vent
      - dow_1..dow_6 : jours de la semaine one-hot (dow_0 = baseline)
    """
    date_du_jour = pd.to_datetime(ligne_prevision["date"])
    jour_de_l_annee = date_du_jour.dayofyear
    saison_sin = np.sin(2 * np.pi * jour_de_l_annee / 365)
    saison_cos = np.cos(2 * np.pi * jour_de_l_annee / 365)

    jour_semaine = date_du_jour.weekday()  # 0=lundi .. 6=dimanche
    colonnes_jour = {f"dow_{k}": 0 for k in range(1, 7)}
    if jour_semaine != 0:
        colonnes_jour[f"dow_{jour_semaine}"] = 1

    valeurs = {
        f"{variable_cible}_prev": float(ligne_prevision[f"{variable_cible}_prev"]),
        "doy_sin": float(saison_sin),
        "doy_cos": float(saison_cos),
        "prcp_prev": float(ligne_prevision.get("prcp_prev", 0.0)),
        "ws_prev": float(ligne_prevision.get("ws_prev", 0.0)),
        **colonnes_jour,
    }

    ordre_colonnes = [
        f"{variable_cible}_prev",
        "doy_sin", "doy_cos",
        "prcp_prev", "ws_prev",
        *colonnes_jour.keys(),
    ]
    return pd.DataFrame([valeurs], columns=ordre_colonnes)


def sauvegarder_dernier_json(prediction: dict) -> None:
    """Écrit la dernière prédiction dans last_prediction.json (pour la CI)."""
    with open(CHEMIN_DERNIER_JSON, "w", encoding="utf-8") as fichier:
        json.dump(prediction, fichier, ensure_ascii=False, indent=2)


def mettre_a_jour_historique_csv(prediction: dict) -> None:
    """
    Ajoute la prédiction du jour dans data/predictions.csv.
    Si la date existe déjà, on remplace la ligne (pas de doublons).
    """
    CHEMIN_HISTORIQUE_CSV.parent.mkdir(parents=True, exist_ok=True)

    colonnes_utiles = ["date", "tmax_prev", "tmax_corr", "tmin_prev", "tmin_corr"]
    nouvelle_ligne = pd.DataFrame([prediction])[colonnes_utiles]

    if CHEMIN_HISTORIQUE_CSV.exists():
        historique = pd.read_csv(CHEMIN_HISTORIQUE_CSV)
        historique = historique[historique["date"] != prediction["date"]]
        historique = pd.concat([historique, nouvelle_ligne], ignore_index=True)
    else:
        historique = nouvelle_ligne

    historique = historique.sort_values("date")
    historique.to_csv(CHEMIN_HISTORIQUE_CSV, index=False)


def main():
    # 1) Charger la dernière prévision brute
    tableau_previsions = pd.read_csv(FORECASTS_CSV).sort_values("date")
    if tableau_previsions.empty:
        raise RuntimeError("Le fichier forecasts.csv est vide, aucune prédiction possible.")
    derniere_prevision = tableau_previsions.iloc[-1]

    # 2) Charger les modèles de correction
    modele_correction_tmax = load("models/hgb_tmax.joblib")
    modele_correction_tmin = load("models/hgb_tmin.joblib")

    # 3) Construire les features et prédire la correction
    X_tmax = construire_variables_explicatives(derniere_prevision, "tmax")
    X_tmin = construire_variables_explicatives(derniere_prevision, "tmin")

    correction_tmax = float(modele_correction_tmax.predict(X_tmax)[0])
    correction_tmin = float(modele_correction_tmin.predict(X_tmin)[0])

    # 4) Construire l'objet résultat
    prediction = {
        "date": str(derniere_prevision["date"]),
        "tmax_prev": round(float(derniere_prevision["tmax_prev"]), 1),
        "tmax_corr": round(float(derniere_prevision["tmax_prev"]) + correction_tmax, 1),
        "tmin_prev": round(float(derniere_prevision["tmin_prev"]), 1),
        "tmin_corr": round(float(derniere_prevision["tmin_prev"]) + correction_tmin, 1),
        "modeles_utilises": {"tmax": "hgb_tmax.joblib", "tmin": "hgb_tmin.joblib"},
    }

    # 5) Afficher pour les logs/CI + sauvegarder JSON + mettre à jour l'historique CSV
    print(json.dumps(prediction, ensure_ascii=False))
    sauvegarder_dernier_json(prediction)
    mettre_a_jour_historique_csv(prediction)


if __name__ == "__main__":
    main()
