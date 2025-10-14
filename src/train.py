import pandas as pd, os
from joblib import dump
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from config import FORECASTS_CSV, OBS_CSV
from features import prepare_merged

def train_one(df, target):
    # target ∈ {"tmax","tmin"}
    err_col = f"err_{target}"
    X = df[[f"{target}_prev","doy_sin","doy_cos"]]
    y = df[err_col]
    m = Ridge(alpha=1.0).fit(X, y)
    # petite éval holdout (15 derniers jours)
    if len(df) > 30:
        tr = df.iloc[:-15]
        te = df.iloc[-15:]
        m_tmp = Ridge(alpha=1.0).fit(tr[[f"{target}_prev","doy_sin","doy_cos"]], tr[err_col])
        pred_corr = te[f"{target}_prev"] + m_tmp.predict(te[[f"{target}_prev","doy_sin","doy_cos"]])
        mae = mean_absolute_error(te[f"{target}_obs"], pred_corr)
    else:
        mae = None
    return m, mae

def main():
    f = pd.read_csv(FORECASTS_CSV)
    o = pd.read_csv(OBS_CSV)
    df = prepare_merged(f, o).sort_values("date")
    os.makedirs("models", exist_ok=True)

    model_tmax, mae_max = train_one(df, "tmax")
    model_tmin, mae_min = train_one(df, "tmin")

    dump(model_tmax, "models/ridge_tmax.joblib")
    dump(model_tmin, "models/ridge_tmin.joblib")

    print("[OK] Modèles enregistrés.")
    if mae_max is not None:
        print(f"MAE(tmax_corr) ~ {mae_max:.2f} °C sur les 15 derniers jours simulés.")
    if mae_min is not None:
        print(f"MAE(tmin_corr) ~ {mae_min:.2f} °C sur les 15 derniers jours simulés.")

if __name__ == "__main__":
    main()
