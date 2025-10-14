import requests, pandas as pd
from datetime import date
from meteostat import Point, Daily
from config import LAT, LON, TIMEZONE, FORECASTS_CSV, OBS_CSV

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/era5"

def fetch_era5_daily(lat, lon, start, end, tz):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "windspeed_10m_max"
        ]),
        "timezone": tz
    }
    r = requests.get(ARCHIVE_URL, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()

    dates = pd.to_datetime(js["daily"]["time"])
    d = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "tmax_prev": js["daily"]["temperature_2m_max"],
        "tmin_prev": js["daily"]["temperature_2m_min"],
        "prcp_prev": js["daily"]["precipitation_sum"],
        "ws_prev":   js["daily"]["windspeed_10m_max"],
    })
    d["source"] = "era5-archive"
    return d


def fetch_obs_daily(lat, lon, start, end):
    loc = Point(lat, lon)
    df = Daily(loc, pd.to_datetime(start), pd.to_datetime(end)).fetch()
    if df.empty:
        raise RuntimeError("Meteostat a renvoyé 0 ligne. Essaie une autre période ou un autre point.")
    out = pd.DataFrame({
        "date": df.index.date.astype(str),
        "tmax_obs": df["tmax"].values,
        "tmin_obs": df["tmin"].values,
        "prcp_obs": df["prcp"].fillna(0).values
    })
    return out

def main():
    # 3 ans d'historique jusqu’à hier
    end = str(date.today())
    start = str(date.today().replace(year=date.today().year - 3))

    print(f"[seed] Téléchargement ERA5 {start} → {end} ...")
    era = fetch_era5_daily(LAT, LON, start, end, TIMEZONE)

    print(f"[seed] Téléchargement observations Meteostat {start} → {end} ...")
    obs = fetch_obs_daily(LAT, LON, start, end)

    # Sauvegarde brute
    era.to_csv(FORECASTS_CSV, index=False)
    obs.to_csv(OBS_CSV, index=False)

    # Petit résumé
    print(f"[OK] forecasts.csv: {len(era)} lignes")
    print(f"[OK] observations.csv: {len(obs)} lignes")
    print("[tip] Enchaîne avec: python src/train.py puis python src/predict.py")

if __name__ == "__main__":
    main()
