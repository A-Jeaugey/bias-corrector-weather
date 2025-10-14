import json
import numpy as np
import pandas as pd
from joblib import load
from config import FORECASTS_CSV

def build_features_for_target(row: pd.Series, target: str) -> pd.DataFrame:
    """
    Construit une ligne de features pour le modèle HGB, en cohérence avec train.py:
      - {target}_prev
      - doy_sin, doy_cos
      - prcp_prev, ws_prev
      - dow_1..dow_6 (one-hot, baseline dow_0 supprimée)
    """
    d = pd.to_datetime(row["date"])
    doy = d.dayofyear
    doy_sin = np.sin(2 * np.pi * doy / 365)
    doy_cos = np.cos(2 * np.pi * doy / 365)

    dow = d.weekday()  # 0=lundi ... 6=dimanche
    dow_cols = {f"dow_{k}": 0 for k in range(1, 7)}
    if dow != 0:
        dow_cols[f"dow_{dow}"] = 1

    feats = {
        f"{target}_prev": float(row[f"{target}_prev"]),
        "doy_sin": float(doy_sin),
        "doy_cos": float(doy_cos),
        "prcp_prev": float(row.get("prcp_prev", 0.0)),
        "ws_prev": float(row.get("ws_prev", 0.0)),
        **dow_cols
    }
    cols = [f"{target}_prev", "doy_sin", "doy_cos", "prcp_prev", "ws_prev"] + list(dow_cols.keys())
    return pd.DataFrame([feats], columns=cols)

def main():
    f = pd.read_csv(FORECASTS_CSV).sort_values("date")
    if f.empty:
        raise RuntimeError("forecasts.csv est vide, impossible de prédire.")
    last = f.iloc[-1]

    m_tmax = load("models/hgb_tmax.joblib")
    m_tmin = load("models/hgb_tmin.joblib")

    X_max = build_features_for_target(last, "tmax")
    X_min = build_features_for_target(last, "tmin")

    corr_max = float(m_tmax.predict(X_max)[0])
    corr_min = float(m_tmin.predict(X_min)[0])

    out = {
        "date": str(last["date"]),
        "tmax_prev": round(float(last["tmax_prev"]), 1),
        "tmax_corr": round(float(last["tmax_prev"]) + corr_max, 1),
        "tmin_prev": round(float(last["tmin_prev"]), 1),
        "tmin_corr": round(float(last["tmin_prev"]) + corr_min, 1),
        "models": {"tmax": "hgb_tmax.joblib", "tmin": "hgb_tmin.joblib"}
    }
    print(json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()
