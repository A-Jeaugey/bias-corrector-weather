import requests
import pandas as pd
from datetime import date
from meteostat import Point, Daily

from config import LAT, LON, TIMEZONE, FORECASTS_CSV, OBS_CSV

# URL de l'historique des PRÉVISIONS Open-Meteo (même famille que l'API de prod)
URL_HISTORIQUE_PREVISIONS = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def recuperer_previsions_historiques_openmeteo(latitude: float, longitude: float,
                                               date_debut: str, date_fin: str,
                                               fuseau_horaire: str) -> pd.DataFrame:
    """
    Récupère les prévisions QUOTIDIENNES historiques d'Open-Meteo (même source que la prod),
    sur la période donnée. On les traite comme des 'prévisions brutes' (tmax_prev, tmin_prev...).

    Args:
        latitude (float): latitude de la localisation
        longitude (float): longitude de la localisation
        date_debut (str): début de la période AAAA-MM-JJ
        date_fin (str): fin de la période AAAA-MM-JJ
        fuseau_horaire (str): timezone pour l'API

    Returns:
        pd.DataFrame: colonnes 'date', 'tmax_prev', 'tmin_prev', 'prcp_prev', 'ws_prev', 'source'
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": date_debut,
        "end_date": date_fin,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "windspeed_10m_max"
        ]),
        "timezone": fuseau_horaire
    }

    r = requests.get(URL_HISTORIQUE_PREVISIONS, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()

    # Conversion des dates en AAAA-MM-JJ
    dates = pd.to_datetime(js["daily"]["time"])

    df_prev = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "tmax_prev": js["daily"]["temperature_2m_max"],
        "tmin_prev": js["daily"]["temperature_2m_min"],
        "prcp_prev": js["daily"]["precipitation_sum"],
        "ws_prev":   js["daily"]["windspeed_10m_max"],
    })

    # Marqueur de source pour filtrer/diagnostiquer plus tard si besoin
    df_prev["source"] = "open-meteo-historical"
    return df_prev


def recuperer_observations_meteostat(latitude: float, longitude: float,
                                     date_debut: str, date_fin: str) -> pd.DataFrame:
    """
    Récupère les observations météo quotidiennes réelles depuis Meteostat
    pour une période donnée.

    Returns:
        pd.DataFrame: colonnes 'date', 'tmax_obs', 'tmin_obs', 'prcp_obs'
    """
    loc = Point(latitude, longitude)
    daily = Daily(loc, pd.to_datetime(date_debut), pd.to_datetime(date_fin)).fetch()

    if daily.empty:
        raise RuntimeError(
            "Meteostat n'a renvoyé aucune donnée. Vérifie les coordonnées, la station ou la période."
        )

    df_obs = pd.DataFrame({
        "date": daily.index.date.astype(str),
        "tmax_obs": daily["tmax"].values,
        "tmin_obs": daily["tmin"].values,
        "prcp_obs": daily["prcp"].fillna(0).values
    })

    return df_obs


def main():
    # Période d'historique : 3 ans jusqu'à aujourd'hui
    date_fin = str(date.today())
    date_debut = str(date.today().replace(year=date.today().year - 3))

    print(f"[seed] Téléchargement prévisions historiques Open-Meteo {date_debut} → {date_fin} ...")
    df_prev = recuperer_previsions_historiques_openmeteo(LAT, LON, date_debut, date_fin, TIMEZONE)

    print(f"[seed] Téléchargement observations Meteostat {date_debut} → {date_fin} ...")
    df_obs = recuperer_observations_meteostat(LAT, LON, date_debut, date_fin)

    # Sauvegardes (remplacent l'existant)
    df_prev.to_csv(FORECASTS_CSV, index=False)
    df_obs.to_csv(OBS_CSV, index=False)

    # Résumé
    print(f"[OK] forecasts.csv : {len(df_prev)} lignes (source: open-meteo-historical)")
    print(f"[OK] observations.csv : {len(df_obs)} lignes")
    print("[tip] Enchaîne avec : python src/train.py puis python src/predict.py")


if __name__ == "__main__":
    main()
