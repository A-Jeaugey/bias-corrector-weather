import pandas as pd
from joblib import load
from features import add_calendar_features
from config import FORECASTS_CSV

def main():
    f = pd.read_csv(FORECASTS_CSV)
    f = f.sort_values("date")
    last = f.tail(1).copy()
    last = add_calendar_features(last)

    mt = load("models/ridge_tmax.joblib")
    mn = load("models/ridge_tmin.joblib")

    last["tmax_corr"] = last["tmax_prev"] + mt.predict(last[["tmax_prev","doy_sin","doy_cos"]])
    last["tmin_corr"] = last["tmin_prev"] + mn.predict(last[["tmin_prev","doy_sin","doy_cos"]])

    d = last.iloc[0].to_dict()
    print({
        "date": d["date"],
        "tmax_prev": round(float(d["tmax_prev"]),1),
        "tmax_corr": round(float(d["tmax_corr"]),1),
        "tmin_prev": round(float(d["tmin_prev"]),1),
        "tmin_corr": round(float(d["tmin_corr"]),1)
    })

if __name__ == "__main__":
    main()
