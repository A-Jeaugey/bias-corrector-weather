import pandas as pd
from meteostat import Point, Daily
from datetime import datetime, timedelta
from dateutil import tz
from config import LAT, LON, TIMEZONE, OBS_CSV


def main():
    """
    Récupère les observations météorologiques réelles d’hier (J-1) via Meteostat
    et les enregistre/met à jour dans OBS_CSV.

    - Si le fichier existe, remplace la ligne d’hier si elle est déjà présente.
    - Si le fichier n’existe pas, le crée.
    - Colonnes enregistrées : date, tmax_obs, tmin_obs, prcp_obs
    """
    # Détermination de la date d’hier dans le fuseau horaire configuré
    date_aujourdhui = datetime.now(tz.gettz(TIMEZONE)).date()
    date_hier = date_aujourdhui - timedelta(days=1)

    # Récupération des observations pour la journée d’hier
    localisation = Point(LAT, LON)
    donnees_journalieres = Daily(localisation, date_hier, date_hier).fetch()  # index DatetimeIndex

    if donnees_journalieres.empty:
        print("[AVERTISSEMENT] Aucune observation disponible pour hier.")
        return

    # Normalisation des colonnes et valeurs manquantes
    ligne = {
        "date": str(date_hier),
        "tmax_obs": float(donnees_journalieres["tmax"].iloc[0]) if pd.notna(donnees_journalieres["tmax"].iloc[0]) else None,
        "tmin_obs": float(donnees_journalieres["tmin"].iloc[0]) if pd.notna(donnees_journalieres["tmin"].iloc[0]) else None,
        "prcp_obs": float(donnees_journalieres["prcp"].iloc[0]) if pd.notna(donnees_journalieres["prcp"].iloc[0]) else 0.0,
    }

    nouveau_tableau = pd.DataFrame([ligne])

    # Fusion avec l’existant en remplaçant la date d’hier si elle est déjà présente
    try:
        tableau_courant = pd.read_csv(OBS_CSV)
        tableau_courant = pd.concat(
            [tableau_courant[~(tableau_courant["date"] == ligne["date"])], nouveau_tableau],
            ignore_index=True
        )
    except FileNotFoundError:
        tableau_courant = nouveau_tableau

    # Sauvegarde
    tableau_courant.to_csv(OBS_CSV, index=False)
    print(f"[OK] Observation enregistrée pour {ligne['date']}.")


if __name__ == "__main__":
    main()
