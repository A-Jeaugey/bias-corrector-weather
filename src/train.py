import os
import pandas as pd
from joblib import dump
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from config import FORECASTS_CSV, OBS_CSV
from features import prepare_merged


def train_single_target(merged_df: pd.DataFrame, variable: str):
    """
    Entraîne un modèle Ridge pour une variable donnée (tmax ou tmin),
    en apprenant à prédire l'erreur entre la prévision et l'observation.

    Args:
        merged_df (pd.DataFrame): DataFrame fusionné contenant les prévisions,
                                  les observations et les features saisonnières.
        variable (str): "tmax" ou "tmin"

    Returns:
        model (Ridge): modèle entraîné
        mae_holdout (float|None): MAE sur les 15 derniers jours simulés (ou None si pas assez de données)
    """
    error_col = f"err_{variable}"

    # Features utilisées pour prédire l'erreur
    feature_cols = [f"{variable}_prev", "doy_sin", "doy_cos"]
    X = merged_df[feature_cols]
    y = merged_df[error_col]

    # Entraînement principal
    model = Ridge(alpha=1.0)
    model.fit(X, y)

    mae_holdout = None
    if len(merged_df) > 30:
        # Séparation temporelle: tout sauf les 15 derniers jours = train, les 15 derniers jours = test
        train_df = merged_df.iloc[:-15]
        test_df = merged_df.iloc[-15:]

        model_tmp = Ridge(alpha=1.0)
        model_tmp.fit(train_df[feature_cols], train_df[error_col])

        # Prédiction de la température corrigée = prévision brute + correction prédite
        y_pred_corrected = test_df[f"{variable}_prev"] + model_tmp.predict(test_df[feature_cols])
        mae_holdout = mean_absolute_error(test_df[f"{variable}_obs"], y_pred_corrected)

    return model, mae_holdout


def main():
    # Chargement des données prévisions / observations
    forecasts_df = pd.read_csv(FORECASTS_CSV)
    observations_df = pd.read_csv(OBS_CSV)

    # Fusion + ajout des features saisonnières
    merged_df = prepare_merged(forecasts_df, observations_df).sort_values("date")

    # Dossier de sauvegarde des modèles
    os.makedirs("models", exist_ok=True)

    # Entraînement pour tmax et tmin
    tmax_model, tmax_mae = train_single_target(merged_df, "tmax")
    tmin_model, tmin_mae = train_single_target(merged_df, "tmin")

    # Sauvegarde des modèles entraînés
    dump(tmax_model, "models/ridge_tmax.joblib")
    dump(tmin_model, "models/ridge_tmin.joblib")

    print("[OK] Modèles enregistrés avec succès.")
    if tmax_mae is not None:
        print(f"MAE(tmax_corr) = {tmax_mae:.2f} °C sur les 15 derniers jours simulés.")
    if tmin_mae is not None:
        print(f"MAE(tmin_corr) = {tmin_mae:.2f} °C sur les 15 derniers jours simulés.")


if __name__ == "__main__":
    main()
