import os
import pandas as pd
from joblib import dump
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance

from config import FORECASTS_CSV, OBS_CSV
from features import prepare_merged


def entrainer_modele_pour_variable(dataframe_fusionne: pd.DataFrame, variable_cible: str):
    """
    Entraîne un modèle de gradient boosting histogramme pour une variable cible donnée
    (par exemple température maximale ou minimale), en apprenant à prédire l'erreur
    entre la prévision brute et l'observation réelle.

    Args:
        dataframe_fusionne (pd.DataFrame): tableau contenant les prévisions,
                                        les observations et les variables explicatives.
        variable_cible (str): "tmax" ou "tmin"

    Returns:
        modele (HistGradientBoostingRegressor): modèle entraîné
        erreur_holdout (float|None): erreur absolue moyenne sur les 15 derniers jours simulés
                                     ou None s’il n’y a pas assez de données.
    """
    colonne_erreur = f"err_{variable_cible}"

    # Sélection dynamique des colonnes explicatives (features)
    colonnes_jour_semaine = [col for col in dataframe_fusionne.columns if col.startswith("dow_")]
    colonnes_explicatives = [
        f"{variable_cible}_prev",  # valeur prévue brute
        "doy_sin", "doy_cos",      # saisonnalité
        "prcp_prev", "ws_prev"     # pluie et vent
    ] + colonnes_jour_semaine

    X = dataframe_fusionne[colonnes_explicatives]
    y = dataframe_fusionne[colonne_erreur]

    # Entraînement du modèle principal
    modele = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=None,
        random_state=42
    )
    modele.fit(X, y)

    # Importance des variables (permutation)
    resultat_importances = permutation_importance(
        modele, X, y, n_repeats=10, random_state=42, n_jobs=-1
    )
    valeurs_importance = resultat_importances.importances_mean
    indices_tries = valeurs_importance.argsort()[::-1]

    print(f"\n[📈] Importance des variables (permutation) pour {variable_cible} :")
    for indice in indices_tries:
        print(f"  - {colonnes_explicatives[indice]} : {valeurs_importance[indice]:.4f}")

    # Évaluation sur les 15 derniers jours (split temporel)
    erreur_holdout = None
    if len(dataframe_fusionne) > 30:
        donnees_apprentissage = dataframe_fusionne.iloc[:-15]
        donnees_test = dataframe_fusionne.iloc[-15:]

        modele_temporaire = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_depth=None,
            random_state=42
        )
        modele_temporaire.fit(
            donnees_apprentissage[colonnes_explicatives],
            donnees_apprentissage[colonne_erreur]
        )

        prediction_corrigee = (
            donnees_test[f"{variable_cible}_prev"] +
            modele_temporaire.predict(donnees_test[colonnes_explicatives])
        )

        erreur_holdout = mean_absolute_error(
            donnees_test[f"{variable_cible}_obs"],
            prediction_corrigee
        )

    return modele, erreur_holdout


def main():
    # Chargement des données de prévisions et d'observations
    tableau_previsions = pd.read_csv(FORECASTS_CSV)
    tableau_observations = pd.read_csv(OBS_CSV)

    # Fusion et enrichissement avec les variables explicatives
    tableau_fusionne = prepare_merged(tableau_previsions, tableau_observations)
    tableau_fusionne = tableau_fusionne.sort_values("date")

    # Création du dossier de sauvegarde des modèles
    os.makedirs("models", exist_ok=True)

    # Entraînement pour la température maximale
    modele_tmax, erreur_tmax = entrainer_modele_pour_variable(tableau_fusionne, "tmax")

    # Entraînement pour la température minimale
    modele_tmin, erreur_tmin = entrainer_modele_pour_variable(tableau_fusionne, "tmin")

    # Sauvegarde des modèles entraînés
    dump(modele_tmax, "models/hgb_tmax.joblib")
    dump(modele_tmin, "models/hgb_tmin.joblib")

    print("\n[OK] Modèles HistGradientBoosting enregistrés avec succès.")
    if erreur_tmax is not None:
        print(f"Erreur MAE (tmax_corr) = {erreur_tmax:.2f} °C sur les 15 derniers jours simulés.")
    if erreur_tmin is not None:
        print(f"Erreur MAE (tmin_corr) = {erreur_tmin:.2f} °C sur les 15 derniers jours simulés.")


if __name__ == "__main__":
    main()
