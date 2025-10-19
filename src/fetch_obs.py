import pandas as pd
from meteostat import Point, Daily
from datetime import datetime, timedelta
from dateutil import tz
from config import LAT, LON, TIMEZONE, OBS_CSV


def main():
    """
    Récupère les observations réelles d’hier (J-1) via Meteostat et les stocke dans OBS_CSV.
    - Remplace la ligne d’hier si elle existe déjà.
    - Crée le fichier si besoin.
    Colonnes : date, tmax_obs, tmin_obs, prcp_obs
    """
    # Hier selon le fuseau configuré (Europe/Paris dans ton projet)
    date_aujourdhui = datetime.now(tz.gettz(TIMEZONE)).date()
    date_hier = date_aujourdhui - timedelta(days=1)

    # Meteostat veut des datetime (pas des date)
    dt_debut = datetime(date_hier.year, date_hier.month, date_hier.day)
    dt_fin = dt_debut  # même jour

    # Récupération des observations
    localisation = Point(LAT, LON)
    donnees_journalieres = Daily(localisation, dt_debut, dt_fin).fetch()

    if donnees_journalieres.empty:
        print("[AVERTISSEMENT] Aucune observation disponible pour hier.")
        return

    # Normalisation des colonnes et gestion des NaN
    tmax = donnees_journalieres["tmax"].iloc[0]
    tmin = donnees_journalieres["tmin"].iloc[0]
    prcp = donnees_journalieres["prcp"].iloc[0]

    ligne = {
        "date": str(date_hier),
        "tmax_obs": float(tmax) if pd.notna(tmax) else None,
        "tmin_obs": float(tmin) if pd.notna(tmin) else None,
        "prcp_obs": float(prcp) if pd.notna(prcp) else 0.0,
    }

    nouveau = pd.DataFrame([ligne])

    try:
        courant = pd.read_csv(OBS_CSV)
        courant = pd.concat([courant[courant["date"] != ligne["date"]], nouveau], ignore_index=True)
    except FileNotFoundError:
        courant = nouveau

    courant.to_csv(OBS_CSV, index=False)
    print(f"[OK] Observation enregistrée pour {ligne['date']}.")


if __name__ == "__main__":
    main()
