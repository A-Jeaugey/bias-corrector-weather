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
        erreur_brute (float|None): MAE de la prévision Open-Meteo brute
        erreur_corrigee (float|None): MAE de la prévision corrigée par HGB
        gain_pct (float|None): Gain en % de la correction
    """
    colonne_erreur = f"err_{variable_cible}"

    # Sélection dynamique des colonnes explicatives (features)
    # On ajoute les nouvelles variables de mémoire et on enlève "dow_"
    colonnes_explicatives = [
        # Variables de base
        f"{variable_cible}_prev",
        "doy_sin", "doy_cos",
        "prcp_prev", "ws_prev",
        "rad_prev", "sun_prev", "cloud_prev",
        
        # Nouvelles variables de mémoire pour tmax
        "err_tmax_j1",
        "err_tmax_j2",
        "err_tmax_moy_7j",
        "err_tmax_std_7j",

        # Nouvelles variables de mémoire pour tmin
        "err_tmin_j1",
        "err_tmin_j2",
        "err_tmin_moy_7j",
        "err_tmin_std_7j",
    ]

    X = dataframe_fusionne[colonnes_explicatives]
    y = dataframe_fusionne[colonne_erreur]

    # Entraînement du modèle principal
    modele = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.05,
        max_leaf_nodes=31,
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

    # --- Évaluation sur les 15 derniers jours (split temporel) ---
    erreur_brute, erreur_corrigee, gain_pct = None, None, None
    
    if len(dataframe_fusionne) > 30:
        donnees_apprentissage = dataframe_fusionne.iloc[:-15]
        donnees_test = dataframe_fusionne.iloc[-15:]

        modele_temporaire = HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=42
        )
        modele_temporaire.fit(
            donnees_apprentissage[colonnes_explicatives],
            donnees_apprentissage[colonne_erreur]
        )

        # 1. Prédiction corrigée
        prediction_corrigee = (
            donnees_test[f"{variable_cible}_prev"] +
            modele_temporaire.predict(donnees_test[colonnes_explicatives])
        )

        # 2. Calcul MAE Brute (Open-Meteo vs Réalité)
        erreur_brute = mean_absolute_error(
            donnees_test[f"{variable_cible}_obs"],   # Réalité
            donnees_test[f"{variable_cible}_prev"]    # Prévision brute
        )

        # 3. Calcul MAE Corrigée (HGB vs Réalité)
        erreur_corrigee = mean_absolute_error(
            donnees_test[f"{variable_cible}_obs"],   # Réalité
            prediction_corrigee                      # Prévision corrigée
        )

        # 4. Calcul du gain
        if erreur_brute > 0:
            gain_pct = ((erreur_brute - erreur_corrigee) / erreur_brute) * 100
        else:
            gain_pct = 0.0 # Cas où l'erreur brute est 0

    return modele, erreur_brute, erreur_corrigee, gain_pct


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
    modele_tmax, mae_brute_tmax, mae_corr_tmax, gain_tmax = entrainer_modele_pour_variable(
        tableau_fusionne, "tmax"
    )

    # Entraînement pour la température minimale
    modele_tmin, mae_brute_tmin, mae_corr_tmin, gain_tmin = entrainer_modele_pour_variable(
        tableau_fusionne, "tmin"
    )

    # Sauvegarde des modèles entraînés
    dump(modele_tmax, "models/hgb_tmax.joblib")
    dump(modele_tmin, "models/hgb_tmin.joblib")

    print("\n[OK] Modèles HistGradientBoosting enregistrés avec succès.")
    
    # --- Affichage des métriques de T° Max ---
    if mae_corr_tmax is not None:
        print(f"\n--- Métriques T° Max (sur 15 jours) ---")
        print(f"  🌡️ MAE Brute (Open-Meteo): {mae_brute_tmax:.2f} °C")
        print(f"  ✨ MAE Corrigée (HGB):   {mae_corr_tmax:.2f} °C")
        print(f"  📊 Amélioration:           {gain_tmax:+.1f} %") # Ajout du '+' pour voir aussi les régressions

    # --- Affichage des métriques de T° Min ---
    if mae_corr_tmin is not None:
        print(f"\n--- Métriques T° Min (sur 15 jours) ---")
        print(f"  🌡️ MAE Brute (Open-Meteo): {mae_brute_tmin:.2f} °C")
        print(f"  ✨ MAE Corrigée (HGB):   {mae_corr_tmin:.2f} °C")
        print(f"  📊 Amélioration:           {gain_tmin:+.1f} %")
    
    


if __name__ == "__main__":
    main()