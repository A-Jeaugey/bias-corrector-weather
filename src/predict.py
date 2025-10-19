import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

from config import FORECASTS_CSV, OBS_CSV
from features import prepare_merged


CHEMIN_DERNIER_JSON = Path("last_prediction.json")      # compatibilité CI
CHEMIN_HISTORIQUE_CSV = Path("data/predictions.csv")   # historique cumulatif


def construire_variables_explicatives(ligne_prevision: pd.Series, historique: pd.DataFrame, variable_cible: str) -> pd.DataFrame:
    """
    Construit une ligne de variables explicatives pour le modèle de correction.
    """
    date_du_jour = pd.to_datetime(ligne_prevision["date"])
    jour_annee = date_du_jour.dayofyear
    saison_sin = np.sin(2 * np.pi * jour_annee / 365)
    saison_cos = np.cos(2 * np.pi * jour_annee / 365)
    
    # Récupère la dernière ligne de l'historique pour les features de mémoire
    dernier_jour_historique = historique.iloc[-1]

    valeurs = {
        # Variables de base
        f"{variable_cible}_prev": float(ligne_prevision[f"{variable_cible}_prev"]),
        "doy_sin": float(saison_sin),
        "doy_cos": float(saison_cos),
        "prcp_prev": float(ligne_prevision.get("prcp_prev", 0.0)),
        "ws_prev": float(ligne_prevision.get("ws_prev", 0.0)),
        "rad_prev": float(ligne_prevision.get("rad_prev", 0.0)),
        "sun_prev": float(ligne_prevision.get("sun_prev", 0.0)),
        "cloud_prev": float(ligne_prevision.get("cloud_prev", 0.0)),

        # Features de mémoire basées sur le DERNIER jour de l'historique
        "err_tmax_j1": float(dernier_jour_historique.get("err_tmax", 0.0)), # L'erreur d'hier est l'erreur du dernier jour connu
        "err_tmax_j2": float(dernier_jour_historique.get("err_tmax_j1", 0.0)),
        "err_tmax_moy_7j": float(dernier_jour_historique.get("err_tmax_moy_7j", 0.0)),
        "err_tmax_std_7j": float(dernier_jour_historique.get("err_tmax_std_7j", 0.0)),
        "err_tmin_j1": float(dernier_jour_historique.get("err_tmin", 0.0)),
        "err_tmin_j2": float(dernier_jour_historique.get("err_tmin_j1", 0.0)),
        "err_tmin_moy_7j": float(dernier_jour_historique.get("err_tmin_moy_7j", 0.0)),
        "err_tmin_std_7j": float(dernier_jour_historique.get("err_tmin_std_7j", 0.0)),
    }

    # L'ordre DOIT être le même que celui utilisé dans train.py
    ordre_colonnes = [
        f"{variable_cible}_prev", "doy_sin", "doy_cos", "prcp_prev", "ws_prev",
        "rad_prev", "sun_prev", "cloud_prev", "err_tmax_j1", "err_tmax_j2",
        "err_tmax_moy_7j", "err_tmax_std_7j", "err_tmin_j1", "err_tmin_j2",
        "err_tmin_moy_7j", "err_tmin_std_7j",
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
    # 1) Charger TOUTES les données pour avoir l'historique
    previsions_hist = pd.read_csv(FORECASTS_CSV)
    observations_hist = pd.read_csv(OBS_CSV)
    historique_complet = prepare_merged(previsions_hist, observations_hist)
    historique_complet = historique_complet.sort_values("date")

    if historique_complet.empty:
        raise RuntimeError("L'historique est vide, impossible de calculer les features de mémoire.")

    # On prend la dernière prévision brute à corriger
    derniere_prevision = previsions_hist.sort_values("date").iloc[-1]

    # 2) Charger les modèles de correction
    modele_correction_tmax = load("models/hgb_tmax.joblib")
    modele_correction_tmin = load("models/hgb_tmin.joblib")

    # 3) Construire les features (en passant l'historique) et prédire la correction
    X_tmax = construire_variables_explicatives(derniere_prevision, historique_complet, "tmax")
    X_tmin = construire_variables_explicatives(derniere_prevision, historique_complet, "tmin")

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