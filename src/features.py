import pandas as pd, numpy as np

def add_calendar_features(df):
    d = pd.to_datetime(df["date"])
    doy = d.dt.dayofyear
    df["doy_sin"] = np.sin(2*np.pi*doy/365)
    df["doy_cos"] = np.cos(2*np.pi*doy/365)
    return df

def prepare_merged(forecasts, obs):
    df = forecasts.merge(obs, on="date", how="inner")
    df = add_calendar_features(df)
    # Erreurs à apprendre (régression sur l'erreur)
    df["err_tmax"] = df["tmax_obs"] - df["tmax_prev"]
    df["err_tmin"] = df["tmin_obs"] - df["tmin_prev"]
    # Filtrage propre
    return df.dropna(subset=["tmax_prev","tmin_prev","tmax_obs","tmin_obs"])
