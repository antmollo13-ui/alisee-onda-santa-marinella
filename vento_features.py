"""Feature engineering VENTO — condiviso tra training (alisee_vento) e previsione.
Stessa filosofia dell'onda: l'NWP entra come feature, la stazione RMN e' la verita'.
Ricalca il set che in v21 dava il grosso dello skill (Stage 1)."""
import numpy as np

MODELLI = ["icon_seamless", "gfs_seamless"]

# Variabili orarie NWP (atmosferiche) richieste a open-meteo
NWP_HOURLY = ("wind_speed_10m,wind_direction_10m,wind_gusts_10m,"
              "temperature_2m,surface_pressure,cloud_cover")
NWP_850 = "wind_speed_850hPa,wind_direction_850hPa,temperature_850hPa"

FEAT = [
    "hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
    "h_sin_x_m_sin", "h_cos_x_m_cos",
    "wind_model_mean", "wind_model_spread", "dir_sin", "dir_cos",
    "gust_mean", "gust_ratio",
    "press", "press_trend_3h", "press_trend_6h", "press_accel_3h",
    "temp", "cloud", "sst", "thermal_power", "thermal_trend_3h",
    "brezza_index", "grecale_kn", "libeccio_kn",
    "u_850", "v_850", "shear_850_10m", "delta_dir_sin", "delta_dir_cos",
    "wind_lag_1h", "wind_lag_3h", "wind_lag_6h", "wind_lag_24h",
    "wind_roll_3h", "wind_roll_6h", "wind_trend", "wind_accel_3h",
]

# Assi dello spot: offshore = grecale (da terra, pettina l'onda), onshore = libeccio
DIR_OFFSHORE = 45
DIR_ONSHORE = 225


def _uv(speed, direction):
    a = np.radians(np.asarray(direction, dtype=float))
    s = np.asarray(speed, dtype=float)
    return s * np.cos(a), s * np.sin(a)


def build_features(df):
    """df deve avere: date, le variabili NWP_HOURLY per ogni modello di MODELLI,
    le 850hPa (icon) e sst. Ritorna df con tutte le FEAT presenti."""
    df = df.copy()
    spd = [f"wind_speed_10m_{m}" for m in MODELLI if f"wind_speed_10m_{m}" in df.columns]
    df["wind_model_mean"] = df[spd].mean(axis=1)
    df["wind_model_spread"] = (df[spd].max(axis=1) - df[spd].min(axis=1)).fillna(0)

    dir_icon = df.get("wind_direction_10m_icon_seamless",
                      df.get("wind_direction_10m", 0)).fillna(0)
    df["dir_sin"] = np.sin(np.radians(dir_icon))
    df["dir_cos"] = np.cos(np.radians(dir_icon))

    g = [f"wind_gusts_10m_{m}" for m in MODELLI if f"wind_gusts_10m_{m}" in df.columns]
    df["gust_mean"] = df[g].mean(axis=1) if g else df["wind_model_mean"]
    df["gust_ratio"] = df["gust_mean"] / df["wind_model_mean"].clip(lower=0.5)

    df["press"] = df.get("surface_pressure_icon_seamless", df.get("surface_pressure", 1013))
    df["press_trend_3h"] = df["press"].diff(3).fillna(0)
    df["press_trend_6h"] = df["press"].diff(6).fillna(0)
    df["press_accel_3h"] = df["press_trend_3h"].diff(3).fillna(0)

    df["temp"] = df.get("temperature_2m_icon_seamless", df.get("temperature_2m", 20))
    df["cloud"] = df.get("cloud_cover_icon_seamless", df.get("cloud_cover", 0))
    if "sst" not in df.columns:
        df["sst"] = df.get("sea_surface_temperature", np.nan)
    df["sst"] = df["sst"].ffill().bfill().fillna(18.0)
    # Motore della brezza: gradiente terra-mare pesato dalla copertura nuvolosa
    df["thermal_power"] = (df["temp"] - df["sst"]) * (1 - df["cloud"] / 100.0)
    df["thermal_trend_3h"] = df["thermal_power"].diff(3).fillna(0)

    spd_850 = df.get("wind_speed_850hPa", 0)
    spd_850 = spd_850.fillna(0) if hasattr(spd_850, "fillna") else spd_850
    dir_850 = df.get("wind_direction_850hPa", 0)
    dir_850 = dir_850.fillna(0) if hasattr(dir_850, "fillna") else dir_850
    df["u_850"], df["v_850"] = _uv(spd_850, dir_850)
    df["shear_850_10m"] = spd_850 - df["wind_model_mean"]
    delta = (dir_850 - dir_icon + 180) % 360 - 180
    df["delta_dir_sin"] = np.sin(np.radians(delta))
    df["delta_dir_cos"] = np.cos(np.radians(delta))
    df["brezza_index"] = df["thermal_power"] / (1.0 + np.clip(spd_850, 0, None))

    # Proiezioni sugli assi dello spot
    df["grecale_kn"] = (df["wind_model_mean"] *
                        np.cos(np.radians(dir_icon - DIR_OFFSHORE))).clip(lower=0)
    df["libeccio_kn"] = (df["wind_model_mean"] *
                         np.cos(np.radians(dir_icon - DIR_ONSHORE))).clip(lower=0)

    h = df["date"].dt.hour
    m = df["date"].dt.month
    doy = df["date"].dt.dayofyear
    df["hour_sin"], df["hour_cos"] = np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24)
    df["month_sin"], df["month_cos"] = np.sin(2*np.pi*m/12), np.cos(2*np.pi*m/12)
    df["doy_sin"], df["doy_cos"] = np.sin(2*np.pi*doy/365), np.cos(2*np.pi*doy/365)
    df["h_sin_x_m_sin"] = df["hour_sin"] * df["month_sin"]
    df["h_cos_x_m_cos"] = df["hour_cos"] * df["month_cos"]

    ref = df["wind_model_mean"]
    for lag in (1, 3, 6, 24):
        df[f"wind_lag_{lag}h"] = ref.shift(lag)
    df["wind_roll_3h"] = ref.rolling(3, min_periods=1).mean()
    df["wind_roll_6h"] = ref.rolling(6, min_periods=1).mean()
    df["wind_trend"] = df["wind_roll_3h"] - ref.rolling(12, min_periods=1).mean()
    df["wind_accel_3h"] = ref.diff(3).fillna(0)

    return df.bfill().ffill()
