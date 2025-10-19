# src/features.py

import pandas as pd
import numpy as np


def ajouter_variables_calendrier(tableau: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les variables de calendrier nécessaires au modèle :
      - doy_sin, doy_cos : encodage saisonnier (jour de l'année) avec années bissextiles
    """
    dates = pd.to_datetime(tableau["date"])
    annees = dates.dt.year
    est_bissextile = (annees % 4 == 0) & ((annees % 100 != 0) | (annees % 400 == 0))
    total_jours = np.where(est_bissextile, 366, 365)
    jour_annee = dates.dt.dayofyear
    angles = 2 * np.pi * (jour_annee - 1) / total_jours

    tableau["doy_sin"] = np.sin(angles)
    tableau["doy_cos"] = np.cos(angles)
    
    # Les features "dow_" (jour de la semaine) ont été supprimées car leur importance était quasi-nulle.
    
    return tableau


def ajouter_variables_memoire(tableau: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute des variables décalées (lagged) et des moyennes glissantes (rolling)
    basées sur les erreurs des jours précédents.
    """
    tableau = tableau.sort_values("date")

    for var in ["tmax", "tmin"]:
        err_col = f"err_{var}"

        # On décale d'abord les données pour n'utiliser que le passé
        erreurs_passees = tableau[err_col].shift(1)

        # Décalages simples : erreurs des jours précédents
        tableau[f"{err_col}_j1"] = erreurs_passees
        tableau[f"{err_col}_j2"] = tableau[err_col].shift(2)

        # Fenêtres glissantes sur les erreurs PASSÉES
        tableau[f"{err_col}_moy_7j"] = erreurs_passees.rolling(window=7, min_periods=1).mean()
        tableau[f"{err_col}_std_7j"] = erreurs_passees.rolling(window=7, min_periods=1).std()

    return tableau


def prepare_merged(previsions: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    """
    Fusionne les prévisions et observations sur la colonne 'date',
    ajoute les variables de calendrier et de mémoire,
    calcule les erreurs à apprendre,
    puis renvoie un tableau propre (sans valeurs manquantes critiques).
    """
    tableau = previsions.merge(observations, on="date", how="inner")
    tableau = ajouter_variables_calendrier(tableau)

    # Erreurs cibles (à apprendre)
    tableau["err_tmax"] = tableau["tmax_obs"] - tableau["tmax_prev"]
    tableau["err_tmin"] = tableau["tmin_obs"] - tableau["tmin_prev"]

    # Ajout des variables décalées et des moyennes glissantes
    tableau = ajouter_variables_memoire(tableau)

    # Remplir les quelques NaN restants (pour l'écart-type au début) avec 0
    tableau = tableau.fillna(0)

    # Filtrage : on enlève les lignes du tout début (NaN dus aux shifts)
    # C'est maintenant géré par .fillna(0) pour ne pas perdre de données
    
    return tableau