import pandas as pd
from meteostat import Point, Daily
from datetime import datetime, timedelta
from dateutil import tz
from config import LAT, LON, TIMEZONE, OBS_CSV

def main():
    # On prend J-1 (hier) comme journée "réelle" à clore
    now = datetime.now(tz.gettz(TIMEZONE)).date()
    yday = now - timedelta(days=1)

    loc = Point(LAT, LON)
    df = Daily(loc, yday, yday).fetch()  # index DateTime
    if df.empty:
        print("[WARN] Pas d'observation dispo pour hier.")
        return

    # Normalise colonnes
    row = {
        "date": str(yday),
        "tmax_obs": float(df["tmax"].iloc[0]) if pd.notna(df["tmax"].iloc[0]) else None,
        "tmin_obs": float(df["tmin"].iloc[0]) if pd.notna(df["tmin"].iloc[0]) else None,
        "prcp_obs": float(df["prcp"].iloc[0]) if pd.notna(df["prcp"].iloc[0]) else 0.0
    }

    df_new = pd.DataFrame([row])
    try:
        cur = pd.read_csv(OBS_CSV)
        cur = pd.concat([cur[~(cur["date"]==row["date"])], df_new], ignore_index=True)
    except FileNotFoundError:
        cur = df_new

    cur.to_csv(OBS_CSV, index=False)
    print(f"[OK] Observation stockée pour {row['date']}.")

if __name__ == "__main__":
    main()
