"""Feature engineering onda — condiviso tra training (alisee_onda) e previsione."""
import numpy as np

FEAT = ["wave_height", "wave_period", "wave_direction",
        "swell_wave_height", "swell_wave_period", "swell_wave_peak_period",
        "wind_wave_height", "wind_wave_period", "wind_wave_peak_period",
        "swell_frac", "hs_sq", "steep", "dir_sin", "dir_cos",
        "mon_sin", "mon_cos",
        "hs_nwp_lag1", "hs_nwp_lag3", "hs_nwp_lag6",
        "hs_nwp_lead1", "hs_nwp_lead3", "hs_nwp_lead6"]

MARINE_HOURLY = ("wave_height,wave_period,wave_direction,"
                 "swell_wave_height,swell_wave_period,swell_wave_peak_period,"
                 "wind_wave_height,wind_wave_period,wind_wave_peak_period")


def build_features(df):
    """Aggiunge le colonne derivate. df deve avere le variabili MARINE_HOURLY
    + colonna 'date' (datetime). Ritorna df con tutte le FEAT presenti."""
    df = df.copy()
    df["swell_frac"] = (df["swell_wave_height"].fillna(0) /
                        df["wave_height"].clip(lower=0.05))
    df["hs_sq"] = df["wave_height"] ** 2
    df["steep"] = df["wave_height"] / df["wave_period"].clip(lower=1)
    df["dir_sin"] = np.sin(np.radians(df["wave_direction"].fillna(0)))
    df["dir_cos"] = np.cos(np.radians(df["wave_direction"].fillna(0)))
    df["month"] = df["date"].dt.month
    df["mon_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["mon_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    for lag in (1, 3, 6):
        df[f"hs_nwp_lag{lag}"] = df["wave_height"].shift(lag)
        df[f"hs_nwp_lead{lag}"] = df["wave_height"].shift(-lag)
    return df.bfill().ffill()


def onda_cardinale(d):
    dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
            'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    return dirs[round(float(d) / 22.5) % 16]
