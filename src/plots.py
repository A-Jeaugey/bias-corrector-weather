# src/plots.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Chemins vers les fichiers de données et le dossier de sortie
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
        # Calcul de l'erreur absolue quotidienne
        data[f"err_brute_{var}"] = (data[f"{var}_obs"] - data[f"{var}_prev"]).abs()
        data[f"err_corr_{var}"] = (data[f"{var}_obs"] - data[f"{var}_corr"]).abs()

        # Calcul de la moyenne glissante (MAE)
        data[f"mae_brute_{var}"] = data[f"err_brute_{var}"].rolling(window=window_size, min_periods=1).mean()
        data[f"mae_corr_{var}"] = data[f"err_corr_{var}"].rolling(window=window_size, min_periods=1).mean()
        
        # Création du graphique
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(12, 6))
        
        plt.plot(data["date"], data[f"mae_brute_{var}"], label=f"MAE Brute (Open-Meteo)", color='tomato', linestyle='--')
        plt.plot(data["date"], data[f"mae_corr_{var}"], label=f"MAE Corrigée (HGB)", color='darkslateblue', linewidth=2)
        
        plt.title(f"Comparaison de l'Erreur Absolue Moyenne (MAE) pour T° {var}", fontsize=16)
        plt.ylabel("Erreur Absolue Moyenne (°C)", fontsize=12)
        plt.xlabel("Date", fontsize=12)
        plt.legend()
        plt.tight_layout() # Ajuste le graphique pour que tout soit visible
        
        # Sauvegarde du fichier
        output_path = PLOTS_DIR / f"mae_comparison_{var}.png"
        plt.savefig(output_path, dpi=120)
        plt.close() # Ferme la figure pour libérer la mémoire
        print(f"  -> Graphique sauvegardé : {output_path}")

def plot_temperature_comparison(data: pd.DataFrame, last_n_days: int = 30):
    """
    Génère et sauvegarde les graphiques comparant les températures (réelle, brute, corrigée)
    sur les N derniers jours.
    """
    print(f"[plots] Génération des graphiques de température ({last_n_days} derniers jours)...")
    
    recent_data = data.tail(last_n_days)
    
    for var in ["tmax", "tmin"]:
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(12, 6))
        
        plt.plot(recent_data["date"], recent_data[f"{var}_obs"], label="Réalité (Meteostat)", color='black', marker='o', linestyle='-')
        plt.plot(recent_data["date"], recent_data[f"{var}_prev"], label="Prévision Brute (Open-Meteo)", color='tomato', linestyle='--')
        plt.plot(recent_data["date"], recent_data[f"{var}_corr"], label="Prévision Corrigée (HGB)", color='darkslateblue', linewidth=2)
        
        plt.title(f"Comparaison des Prévisions de T° {var} ({last_n_days} derniers jours)", fontsize=16)
        plt.ylabel("Température (°C)", fontsize=12)
        plt.xlabel("Date", fontsize=12)
        plt.xticks(rotation=45) # Fait pivoter les dates pour une meilleure lisibilité
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
    # S'assurer que le dossier de sortie existe
    PLOTS_DIR.mkdir(exist_ok=True)

    # Charger et fusionner les données
    try:
        predictions = pd.read_csv(PREDICTIONS_CSV)
        observations = pd.read_csv(OBSERVATIONS_CSV)
    except FileNotFoundError as e:
        print(f"[ERREUR] Fichier manquant : {e}. Lance d'abord le seed et le train.")
        return

    data = pd.merge(predictions, observations, on="date", how="inner")
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date")
    
    # Générer les graphiques
    plot_mae_comparison(data)
    plot_temperature_comparison(data)
    print("[OK] Tous les graphiques ont été générés.")


if __name__ == "__main__":
    main()