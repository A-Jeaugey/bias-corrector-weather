import os
import requests, pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from dateutil import tz
from config import LAT, LON, TIMEZONE, FORECASTS_CSV

URL = "https://api.open-meteo.com/v1/forecast"

def _safe_read_csv(path):
    try:
        df = pd.read_csv(path)
        # si le fichier existe mais sans colonnes/vidé
        if df.shape[1] == 0:
            return None
        return df
    except FileNotFoundError:
        return None
    except pd.errors.EmptyDataError:
        return None

def main():
    # s'assure que le dossier data/ existe
    Path(FORECASTS_CSV).parent.mkdir(parents=True, exist_ok=True)

    params = {
        "latitude": LAT,
        "longitude": LON,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "windspeed_10m_max"
        ]),
        "timezone": TIMEZONE
    }
    r = requests.get(URL, params=params, timeout=20)
    r.raise_for_status()
    js = r.json()
    daily = js["daily"]

    dates = [pd.to_datetime(d).date() for d in daily["time"]]
    tomorrow = (datetime.now(tz.gettz(TIMEZONE)).date() + timedelta(days=1))
    if tomorrow not in dates:
        raise RuntimeError("Demain pas présent dans la réponse de l'API. Réessaie plus tard.")

    i = dates.index(tomorrow)
    row = {
        "date": str(tomorrow),
        "tmax_prev": float(daily["temperature_2m_max"][i]),
        "tmin_prev": float(daily["temperature_2m_min"][i]),
        "prcp_prev": float(daily.get("precipitation_sum", [None])[i]),
        "ws_prev":   float(daily.get("windspeed_10m_max", [None])[i]),
        "source": "open-meteo"
    }

    df_new = pd.DataFrame([row])
    df_old = _safe_read_csv(FORECASTS_CSV)

    if df_old is None:
        df = df_new
    else:
        # remplace la ligne si la date existe déjà
        df = pd.concat([df_old[df_old["date"] != row["date"]], df_new], ignore_index=True)

    df.to_csv(FORECASTS_CSV, index=False)
    print(f"[OK] Forecast J+1 stockée pour {row['date']}.")

if __name__ == "__main__":
    main()
