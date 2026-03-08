import requests
import pandas as pd
from datetime import date

from config import LAT, LON, TIMEZONE, FORECASTS_CSV, OBS_CSV

# URL de l'historique des PRÉVISIONS Open-Meteo (même famille que l'API de prod)
URL_HISTORIQUE_PREVISIONS = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# URL de l'API Open-Meteo pour les observations historiques réelles
URL_OBSERVATIONS = "https://archive-api.open-meteo.com/v1/archive"


def recuperer_previsions_historiques_openmeteo(latitude: float, longitude: float,
                                              date_debut: str, date_fin: str,
                                              fuseau_horaire: str) -> pd.DataFrame:
    """
    Récupère les prévisions QUOTIDIENNES historiques d'Open-Meteo (même source que la prod),
    sur la période donnée. On les traite comme des 'prévisions brutes' (tmax_prev, tmin_prev...).
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
            "windspeed_10m_max",
            "shortwave_radiation_sum",
            "sunshine_duration",
            "cloud_cover_mean"
        ]),
        "timezone": fuseau_horaire
    }

    r = requests.get(URL_HISTORIQUE_PREVISIONS, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()

    dates = pd.to_datetime(js["daily"]["time"])

    df_prev = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "tmax_prev": js["daily"]["temperature_2m_max"],
        "tmin_prev": js["daily"]["temperature_2m_min"],
        "prcp_prev": js["daily"]["precipitation_sum"],
        "ws_prev":   js["daily"]["windspeed_10m_max"],
        "rad_prev":  js["daily"]["shortwave_radiation_sum"],
        "sun_prev":  js["daily"]["sunshine_duration"],
        "cloud_prev":js["daily"]["cloud_cover_mean"],
    })

    df_prev["source"] = "open-meteo-historical"
    return df_prev


def recuperer_observations_openmeteo(latitude: float, longitude: float,
                                     date_debut: str, date_fin: str,
                                     fuseau_horaire: str) -> pd.DataFrame:
    """
    Récupère les observations météo quotidiennes réelles depuis Open-Meteo Archive
    pour une période donnée.

    Returns:
        pd.DataFrame: colonnes 'date', 'tmax_obs', 'tmin_obs', 'prcp_obs'
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": date_debut,
        "end_date": date_fin,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": fuseau_horaire,
    }

    r = requests.get(URL_OBSERVATIONS, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()

    daily = js.get("daily", {})
    if not daily or not daily.get("time"):
        raise RuntimeError(
            "Open-Meteo Archive n'a renvoyé aucune donnée. Vérifie les coordonnées ou la période."
        )

    df_obs = pd.DataFrame({
        "date": daily["time"],
        "tmax_obs": daily["temperature_2m_max"],
        "tmin_obs": daily["temperature_2m_min"],
        "prcp_obs": [p if p is not None else 0.0 for p in daily["precipitation_sum"]],
    })

    return df_obs


def main():
    # Période d'historique : 3 ans jusqu'à aujourd'hui
    date_fin = str(date.today())
    date_debut = str(date.today().replace(year=date.today().year - 3))

    print(f"[seed] Téléchargement prévisions historiques Open-Meteo {date_debut} → {date_fin} ...")
    df_prev = recuperer_previsions_historiques_openmeteo(LAT, LON, date_debut, date_fin, TIMEZONE)

    print(f"[seed] Téléchargement observations Open-Meteo Archive {date_debut} → {date_fin} ...")
    df_obs = recuperer_observations_openmeteo(LAT, LON, date_debut, date_fin, TIMEZONE)

    # Sauvegardes (remplacent l'existant)
    df_prev.to_csv(FORECASTS_CSV, index=False)
    df_obs.to_csv(OBS_CSV, index=False)

    # Résumé
    print(f"[OK] forecasts.csv : {len(df_prev)} lignes (source: open-meteo-historical)")
    print(f"[OK] observations.csv : {len(df_obs)} lignes")


if __name__ == "__main__":
    main()
