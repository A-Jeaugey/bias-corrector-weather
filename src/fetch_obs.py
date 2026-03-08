import pandas as pd
import requests
from datetime import datetime, timedelta
from dateutil import tz
from config import LAT, LON, TIMEZONE, OBS_CSV

# URL de l'API Open-Meteo pour les observations historiques réelles
URL_OBSERVATIONS = "https://archive-api.open-meteo.com/v1/archive"


def main():
    """
    Récupère les observations réelles d'hier (J-1) via Open-Meteo Archive et les stocke dans OBS_CSV.
    - Remplace la ligne d'hier si elle existe déjà.
    - Crée le fichier si besoin.
    Colonnes : date, tmax_obs, tmin_obs, prcp_obs
    """
    # Hier selon le fuseau configuré (Europe/Paris dans ton projet)
    date_aujourdhui = datetime.now(tz.gettz(TIMEZONE)).date()
    date_hier = date_aujourdhui - timedelta(days=1)

    # Paramètres de requête à l'API Open-Meteo Archive
    parametres = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": str(date_hier),
        "end_date": str(date_hier),
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": TIMEZONE,
    }

    reponse = requests.get(URL_OBSERVATIONS, params=parametres, timeout=20)
    reponse.raise_for_status()
    donnees_json = reponse.json()
    donnees_quotidiennes = donnees_json.get("daily", {})

    if not donnees_quotidiennes or not donnees_quotidiennes.get("time"):
        print("[AVERTISSEMENT] Aucune observation disponible pour hier.")
        return

    # Extraction des valeurs
    tmax = donnees_quotidiennes["temperature_2m_max"][0]
    tmin = donnees_quotidiennes["temperature_2m_min"][0]
    prcp = donnees_quotidiennes["precipitation_sum"][0]

    ligne = {
        "date": str(date_hier),
        "tmax_obs": float(tmax) if tmax is not None else None,
        "tmin_obs": float(tmin) if tmin is not None else None,
        "prcp_obs": float(prcp) if prcp is not None else 0.0,
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
