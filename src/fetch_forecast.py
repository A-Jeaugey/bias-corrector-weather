import os
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from dateutil import tz
from config import LAT, LON, TIMEZONE, FORECASTS_CSV

# URL de l'API Open-Meteo pour les prévisions quotidiennes
URL_PREVISIONS = "https://api.open-meteo.com/v1/forecast"


def lire_csv_sans_echec(chemin_fichier: str) -> pd.DataFrame | None:
    """
    Lit un CSV si possible, sinon renvoie None.
    Gère les cas : fichier absent, vide, ou sans colonnes.
    """
    try:
        tableau = pd.read_csv(chemin_fichier)
        if tableau.shape[1] == 0:
            return None
        return tableau
    except FileNotFoundError:
        return None
    except pd.errors.EmptyDataError:
        return None


def main():
    """
    Récupère la prévision J+1 (demain) via Open-Meteo et
    l’enregistre dans FORECASTS_CSV en remplaçant la ligne
    de la même date si elle existe déjà.
    """
    # S’assure que le dossier data/ existe
    Path(FORECASTS_CSV).parent.mkdir(parents=True, exist_ok=True)

    # Paramètres de requête à l’API
    parametres = {
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

    reponse = requests.get(URL_PREVISIONS, params=parametres, timeout=20)
    reponse.raise_for_status()
    donnees_json = reponse.json()
    donnees_quotidiennes = donnees_json["daily"]

    # Liste de dates renvoyées par l’API
    liste_dates = [pd.to_datetime(d).date() for d in donnees_quotidiennes["time"]]

    # Date de demain dans le fuseau horaire configuré
    date_demain = datetime.now(tz.gettz(TIMEZONE)).date() + timedelta(days=1)
    if date_demain not in liste_dates:
        raise RuntimeError("La date de demain n’est pas présente dans la réponse de l’API. Réessaie plus tard.")

    indice = liste_dates.index(date_demain)

    # Construction de la ligne à enregistrer
    ligne = {
        "date": str(date_demain),
        "tmax_prev": float(donnees_quotidiennes["temperature_2m_max"][indice]),
        "tmin_prev": float(donnees_quotidiennes["temperature_2m_min"][indice]),
        "prcp_prev": float(donnees_quotidiennes.get("precipitation_sum", [None])[indice]),
        "ws_prev":   float(donnees_quotidiennes.get("windspeed_10m_max", [None])[indice]),
        "source": "open-meteo"
    }

    nouveau_tableau = pd.DataFrame([ligne])
    ancien_tableau = lire_csv_sans_echec(FORECASTS_CSV)

    if ancien_tableau is None:
        tableau_final = nouveau_tableau
    else:
        # Remplace la ligne si la date existe déjà
        tableau_final = pd.concat(
            [ancien_tableau[ancien_tableau["date"] != ligne["date"]], nouveau_tableau],
            ignore_index=True
        )

    tableau_final.to_csv(FORECASTS_CSV, index=False)
    print(f"[OK] Prévision J+1 enregistrée pour {ligne['date']}.")


if __name__ == "__main__":
    main()
