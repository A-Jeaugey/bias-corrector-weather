# src/plots.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Chemins vers les fichiers de données et le dossier de sortie
FORECASTS_CSV = Path("data/forecasts.csv")
PREDICTIONS_CSV = Path("data/predictions.csv")
OBSERVATIONS_CSV = Path("data/observations.csv")
PLOTS_DIR = Path("plots")


def plot_mae_comparison(data: pd.DataFrame, window_size: int = 30):
    """
    Génère et sauvegarde les graphiques comparant la MAE glissante
    de la prévision brute vs. corrigée pour tmax et tmin.
    """
    print(f"[plots] Génération des graphiques MAE (fenêtre de {window_size} jours)...")
    
    for var in ["tmax", "tmin"]:
        data[f"err_brute_{var}"] = (data[f"{var}_obs"] - data[f"{var}_prev"]).abs()
        data[f"err_corr_{var}"] = (data[f"{var}_obs"] - data[f"{var}_corr"]).abs()

        data[f"mae_brute_{var}"] = data[f"err_brute_{var}"].rolling(window=window_size, min_periods=1).mean()
        # La MAE corrigée sera calculée uniquement sur les données disponibles
        data[f"mae_corr_{var}"] = data[f"err_corr_{var}"].rolling(window=window_size, min_periods=1).mean()
        
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(12, 6))
        
        plt.plot(data["date"], data[f"mae_brute_{var}"], label=f"MAE Brute (Open-Meteo)", color='tomato', linestyle='--')
        # Matplotlib ignore les valeurs NaN, donc la ligne bleue commencera quand les données seront là
        plt.plot(data["date"], data[f"mae_corr_{var}"], label=f"MAE Corrigée (HGB)", color='darkslateblue', linewidth=2)
        
        plt.title(f"Comparaison de l'Erreur Absolue Moyenne (MAE) pour T° {var}", fontsize=16)
        plt.ylabel("Erreur Absolue Moyenne (°C)", fontsize=12)
        plt.xlabel("Date", fontsize=12)
        plt.legend()
        plt.tight_layout()
        
        output_path = PLOTS_DIR / f"mae_comparison_{var}.png"
        plt.savefig(output_path, dpi=120)
        plt.close()
        print(f"  -> Graphique sauvegardé : {output_path}")

def plot_temperature_comparison(data: pd.DataFrame, last_n_days: int = 30):
    """
    Génère et sauvegarde les graphiques comparant les températures sur les N derniers jours.
    """
    print(f"[plots] Génération des graphiques de température ({last_n_days} derniers jours)...")
    
    recent_data = data.tail(last_n_days)
    
    for var in ["tmax", "tmin"]:
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(12, 6))
        
        plt.plot(recent_data["date"], recent_data[f"{var}_obs"], label="Réalité (Meteostat)", color='black', marker='.', linestyle='-')
        plt.plot(recent_data["date"], recent_data[f"{var}_prev"], label="Prévision Brute (Open-Meteo)", color='tomato', linestyle='--')
        plt.plot(recent_data["date"], recent_data[f"{var}_corr"], label="Prévision Corrigée (HGB)", color='darkslateblue', linewidth=2, marker='o')
        
        plt.title(f"Comparaison des Prévisions de T° {var} ({last_n_days} derniers jours)", fontsize=16)
        plt.ylabel("Température (°C)", fontsize=12)
        plt.xlabel("Date", fontsize=12)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        
        output_path = PLOTS_DIR / f"temperature_comparison_{var}.png"
        plt.savefig(output_path, dpi=120)
        plt.close()
        print(f"  -> Graphique sauvegardé : {output_path}")


def main():
    """
    Fonction principale pour charger les données et appeler les fonctions de plotting.
    """
    PLOTS_DIR.mkdir(exist_ok=True)

    try:
        # On charge les 3 fichiers de données
        forecasts = pd.read_csv(FORECASTS_CSV)
        observations = pd.read_csv(OBSERVATIONS_CSV)
        predictions = pd.read_csv(PREDICTIONS_CSV)
    except FileNotFoundError as e:
        print(f"❌ [ERREUR] Fichier de données manquant : {e}.")
        return

    # --- NOUVELLE LOGIQUE DE FUSION ---
    print("[plots] Fusion des données historiques...")
    # 1. On fusionne l'historique brut et les observations
    base_data = pd.merge(forecasts, observations, on="date", how="inner")

    # 2. On fait une jointure GAUCHE pour ajouter les prédictions corrigées
    # Cela garde TOUT l'historique de base, et ajoute les corrections là où elles existent
    data = pd.merge(
        base_data, 
        predictions[['date', 'tmax_corr', 'tmin_corr']], # On ne prend que les colonnes utiles de predictions.csv
        on="date", 
        how="left"
    )

    if data.empty:
        print("❌ [ERREUR] Le tableau de base (`forecasts` + `observations`) est vide.")
        return
        
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date")
    print(f"  -> {len(data)} jours d'historique à tracer.")
    
    plot_mae_comparison(data)
    plot_temperature_comparison(data)
    print("✅ [OK] Tous les graphiques ont été générés.")


if __name__ == "__main__":
    main()