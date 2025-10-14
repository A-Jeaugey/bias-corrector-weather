import pandas as pd
import numpy as np

def ajouter_variables_calendrier(tableau: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les variables de calendrier nécessaires au modèle :
      - doy_sin, doy_cos : encodage saisonnier (jour de l'année en sinus/cosinus)
      - dow_*            : jour de la semaine en one-hot (0 = lundi ... 6 = dimanche, baseline dow_0 supprimée)
    """
    dates = pd.to_datetime(tableau["date"])
    jour_annee = dates.dt.dayofyear

    # Saisonnalité (position sur le cycle annuel)
    tableau["doy_sin"] = np.sin(2 * np.pi * jour_annee / 365)
    tableau["doy_cos"] = np.cos(2 * np.pi * jour_annee / 365)

    # Jour de la semaine encodé en one-hot
    tableau["dow"] = dates.dt.weekday  # 0 = lundi ... 6 = dimanche
    tableau = pd.get_dummies(tableau, columns=["dow"], prefix="dow", drop_first=True)

    return tableau


def prepare_merged(previsions: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    """
    Fusionne les prévisions et observations sur la colonne 'date',
    ajoute les variables de calendrier, calcule les erreurs à apprendre,
    puis renvoie un tableau propre (sans valeurs manquantes critiques).

    Colonnes calculées :
      - err_tmax = tmax_obs - tmax_prev
      - err_tmin = tmin_obs - tmin_prev
    """
    tableau = previsions.merge(observations, on="date", how="inner")
    tableau = ajouter_variables_calendrier(tableau)

    # Erreurs cibles pour l’apprentissage (le modèle prédit la correction à appliquer)
    tableau["err_tmax"] = tableau["tmax_obs"] - tableau["tmax_prev"]
    tableau["err_tmin"] = tableau["tmin_obs"] - tableau["tmin_prev"]

    # Filtrage : on enlève les lignes sans valeurs essentielles
    tableau = tableau.dropna(subset=["tmax_prev", "tmin_prev", "tmax_obs", "tmin_obs"])

    return tableau
