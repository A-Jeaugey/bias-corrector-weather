import requests
import pandas as pd
from datetime import date
from meteostat import Point, Daily

from config import LAT, LON, TIMEZONE, FORECASTS_CSV, OBS_CSV

# URL de l'API Open-Meteo pour les données d'archives ERA5
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/era5"


def fetch_era5_daily(lat: float, lon: float, start_date: str, end_date: str, timezone: str) -> pd.DataFrame:
    """
    Récupère les données ERA5 quotidiennes sur une période donnée
    et les formate comme des 'prévisions' (proxy historique).

    Args:
        lat (float): latitude de la localisation
        lon (float): longitude de la localisation
        start_date (str): début de la période AAAA-MM-JJ
        end_date (str): fin de la période AAAA-MM-JJ
        timezone (str): timezone pour l'API

    Returns:
        pd.DataFrame: DataFrame contenant les colonnes 'tmax_prev', 'tmin_prev', 'prcp_prev', 'ws_prev'
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "windspeed_10m_max"
        ]),
        "timezone": timezone
    }

    response = requests.get(ARCHIVE_URL, params=params, timeout=30)
    response.raise_for_status()
    data_json = response.json()

    # Conversion des dates en chaînes formatées AAAA-MM-JJ
    dates = pd.to_datetime(data_json["daily"]["time"])

    era_df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "tmax_prev": data_json["daily"]["temperature_2m_max"],
        "tmin_prev": data_json["daily"]["temperature_2m_min"],
        "prcp_prev": data_json["daily"]["precipitation_sum"],
        "ws_prev":   data_json["daily"]["windspeed_10m_max"],
    })

    era_df["source"] = "era5-archive"
    return era_df


def fetch_observations_daily(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Récupère les observations météo quotidiennes (réelles)
    depuis Meteostat pour une période donnée.

    Args:
        lat (float): latitude
        lon (float): longitude
        start_date (str): début AAAA-MM-JJ
        end_date (str): fin AAAA-MM-JJ

    Returns:
        pd.DataFrame: DataFrame contenant 'tmax_obs', 'tmin_obs', 'prcp_obs'
    """
    location = Point(lat, lon)
    daily_data = Daily(location, pd.to_datetime(start_date), pd.to_datetime(end_date)).fetch()

    if daily_data.empty:
        raise RuntimeError(
            "Meteostat a renvoyé 0 ligne. Vérifie la station, la période ou les coordonnées."
        )

    obs_df = pd.DataFrame({
        "date": daily_data.index.date.astype(str),
        "tmax_obs": daily_data["tmax"].values,
        "tmin_obs": daily_data["tmin"].values,
        "prcp_obs": daily_data["prcp"].fillna(0).values
    })

    return obs_df


def main():
    # On récupère 3 ans d'historique, jusqu'à aujourd'hui
    end_date = str(date.today())
    start_date = str(date.today().replace(year=date.today().year - 3))

    print(f"[seed] Téléchargement ERA5 {start_date} → {end_date} ...")
    era_df = fetch_era5_daily(LAT, LON, start_date, end_date, TIMEZONE)

    print(f"[seed] Téléchargement observations Meteostat {start_date} → {end_date} ...")
    obs_df = fetch_observations_daily(LAT, LON, start_date, end_date)

    # Sauvegarde dans les fichiers CSV (remplace l'existant)
    era_df.to_csv(FORECASTS_CSV, index=False)
    obs_df.to_csv(OBS_CSV, index=False)

    # Résumé rapide
    print(f"[OK] forecasts.csv: {len(era_df)} lignes")
    print(f"[OK] observations.csv: {len(obs_df)} lignes")
    print("[tip] Enchaîne avec : python src/train.py puis python src/predict.py")


if __name__ == "__main__":
    main()
