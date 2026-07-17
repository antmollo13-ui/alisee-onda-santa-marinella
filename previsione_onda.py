"""
ALISEE — previsione onda + vento 72h per Santa Marinella. Il prodotto.
Carica modello_onda.pkl e modello_vento.pkl, scarica i forecast (boa RON per l'onda,
punto spot per il vento), applica la calibrazione e RIGENERA a ogni run:
  - dashboard.html : pagina completa
  - widget.html    : LA STESSA pagina, trasparente e senza cornice, per l'iframe
                     sul sito del cliente (una miniatura compressa faceva schifo:
                     il cliente incorpora la pagina buona, non una versione ridotta)
"""
import os, pickle, datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests
from onda_features import MARINE_HOURLY, build_features, onda_cardinale
from vento_features import (FEAT as FEAT_V, MODELLI, NWP_HOURLY, NWP_850,
                            build_features as feat_vento, DIR_OFFSHORE)

# Fuso ancorato: sul cloud (GitHub Actions) il sistema e' in UTC — senza questo
# la dashboard mostrerebbe l'ora UTC spacciandola per ora italiana.
ROMA = ZoneInfo("Europe/Rome")
# Ore UTC del cron (allineate all'uscita delle run del modello d'onda)
CRON_UTC = [5, 11, 17, 23]

# Percorso portabile: la cartella dello script (funziona su Windows e su Linux/cloud)
BASE = os.path.dirname(os.path.abspath(__file__))
LAT_BOA, LON_BOA = 42.05, 11.70      # boa RON (onda)
LAT_SPOT, LON_SPOT = 42.034, 11.849  # spot (vento)
SPOT = "Santa Marinella"

# Co-branding. Si accende con la variabile d'ambiente PARTNER (es. "SurfCam Italia").
# Tenuto SPENTO di default: la pagina e' pubblica e il nome di un partner non va
# esposto finche' non c'e' un accordo. Per la demo:  set PARTNER=SurfCam Italia
PARTNER = os.environ.get("PARTNER", "").strip()

# Skill validata OOS in UNITA' REALI (non percentuali astratte).
# Fonte: alisee_onda.py (test 2026, 4.666 ore) e alisee_vento.py (test 2025, 8.571 ore),
# riproducibili con benchmark_baseline.py. Errore medio assoluto vs strumento.
SKILL = [
    # (cosa, errore ALISEE, errore standard, unita', decimali, % errore in meno)
    ("altezza onda", 12,   17,   "cm", 0, 28),
    ("periodo",      0.54, 0.86, "s",  2, 37),
    ("vento",        1.16, 1.71, "kn", 2, 32),
]
SKILL_ORE = "13.000"   # ore di confronto totali (onda 4.666 + vento 8.571)

# Fascia probabile MISURATA (verita_intervalli.py, test 2026): per ogni fascia di
# valore PREVISTO, dove sta il mare reale 8 volte su 10 (quantili osservati alla
# boa, NON dedotti dal MAE — il MAE medio sottostima l'incertezza sulle mareggiate).
_BANDA_PRED = [0.20, 0.50, 0.70, 0.90, 1.15, 1.50, 1.95, 2.60]
_BANDA_P10  = [0.13, 0.32, 0.46, 0.62, 0.83, 1.11, 1.36, 2.01]
_BANDA_P90  = [0.40, 0.62, 0.83, 1.08, 1.44, 1.78, 2.43, 3.25]

STATI = {  # stato -> (etichetta, colore). "piatto" e "piccolo" restano etichette
    # distinte nelle card, ma nel grafico hanno LO STESSO grigio: per chi guarda
    # sono la stessa notizia (niente da surfare), due grigi diversi erano rumore.
    "piatto":      ("piatto",    "#484f58"),
    "piccolo":     ("piccolo",   "#484f58"),
    "mosso/corto": ("mosso",     "#d29922"),
    "surfabile":   ("surfabile", "#58a6ff"),
    "BUONO":       ("buono",     "#3fb950"),
}
# Legenda del grafico: solo le classi che si VEDONO (4 colori, non 5 stati)
_LEGENDA = [("buono", "#3fb950"), ("surfabile", "#58a6ff"),
            ("mosso", "#d29922"), ("niente da surfare", "#484f58")]


_GG = {0: "lun", 1: "mar", 2: "mer", 3: "gio", 4: "ven", 5: "sab", 6: "dom"}


def gg(ts):
    """Giorno abbreviato in italiano (%a dipende dal locale di sistema)."""
    return _GG[pd.Timestamp(ts).weekday()]


def orari_run():
    """(ultima run, prossima run) in ora italiana. L'ultima e' adesso (la run in
    corso); la prossima si ricava dal cron UTC del workflow."""
    ora = datetime.datetime.now(ROMA)
    u = datetime.datetime.now(datetime.timezone.utc)
    cand = []
    for giorno in (0, 1):
        for h in CRON_UTC:
            t = (u + datetime.timedelta(days=giorno)).replace(
                hour=h, minute=0, second=0, microsecond=0)
            if t > u:
                cand.append(t)
    prossima = min(cand).astimezone(ROMA) if cand else None
    return ora, prossima


def comp_offshore(w_dir):
    """+1 = vento da terra (pettina l'onda), -1 = da mare (la rovina)."""
    return float(np.cos(np.radians(float(w_dir or 0) - DIR_OFFSHORE)))


def giudizio(hs, tp, w_kn=0.0, w_dir=0.0):
    """Qualita' surf: onda (size + periodo) MODULATA dal vento.
    L'onshore teso rovina anche un'onda formata; l'offshore la pulisce."""
    if hs < 0.5:  return "piatto"
    if hs < 0.8:  return "piccolo"
    off = comp_offshore(w_dir)
    if w_kn >= 12 and off < -0.3:  return "mosso/corto"   # onshore teso: chop
    if tp < 5:                      return "mosso/corto"   # mare corto da vento
    if hs >= 1.2 and tp >= 6 and (w_kn < 8 or off > 0.3):
        return "BUONO"                                     # formata + vento amico
    return "surfabile"


def _get(url, params):
    r = requests.get(url, params=params, timeout=90)
    r.raise_for_status()
    d = pd.DataFrame(r.json()["hourly"]).rename(columns={"time": "date"})
    d["date"] = pd.to_datetime(d["date"])
    return d


def scarica_e_prevedi():
    with open(os.path.join(BASE, "modello_onda.pkl"), "rb") as f:
        MO = pickle.load(f)
    with open(os.path.join(BASE, "modello_vento.pkl"), "rb") as f:
        MV = pickle.load(f)

    # ── ONDA (boa) — MARINE_HOURLY = variabili del modello; la SST e' display
    dm = _get("https://marine-api.open-meteo.com/v1/marine",
              {"latitude": LAT_BOA, "longitude": LON_BOA,
               "hourly": MARINE_HOURLY + ",sea_surface_temperature",
               "timezone": "Europe/Rome", "forecast_days": 3})
    df = build_features(dm)
    df["hs_alisee"] = np.clip(MO["hs"].predict(df[MO["feat"]]), 0, None)
    df["tp_alisee"] = np.clip(MO["tp"].predict(df[MO["feat"]]), 0, None)
    tot = (df["swell_wave_height"].fillna(0) + df["wind_wave_height"].fillna(0)).clip(lower=0.01)
    df["swell_pct"] = (df["swell_wave_height"].fillna(0) / tot * 100).clip(0, 100)

    # ── VENTO (spot) — atmosferico + 850hPa + SST della boa
    da = _get("https://api.open-meteo.com/v1/forecast",
              {"latitude": LAT_SPOT, "longitude": LON_SPOT, "hourly": NWP_HOURLY,
               "models": ",".join(MODELLI), "wind_speed_unit": "kn",
               "timezone": "Europe/Rome", "forecast_days": 3})
    d8 = _get("https://api.open-meteo.com/v1/forecast",
              {"latitude": LAT_SPOT, "longitude": LON_SPOT, "hourly": NWP_850,
               "models": "icon_seamless", "wind_speed_unit": "kn",
               "timezone": "Europe/Rome", "forecast_days": 3})
    dv = da.merge(d8, on="date", how="left").merge(
        dm[["date", "sea_surface_temperature"]].rename(
            columns={"sea_surface_temperature": "sst"}), on="date", how="left")
    dv = feat_vento(dv)
    dv["vento_kn"] = np.clip(MV["m"].predict(dv[MV["feat"]]), 0, None)
    dv["vento_dir"] = dv["wind_direction_10m_icon_seamless"].fillna(0)
    dv["vento_icon"] = dv["wind_speed_10m_icon_seamless"]

    df = df.merge(dv[["date", "vento_kn", "vento_dir", "vento_icon"]],
                  on="date", how="left")
    df[["vento_kn", "vento_dir"]] = df[["vento_kn", "vento_dir"]].ffill().bfill()
    df["stato"] = [giudizio(h, t, w, d) for h, t, w, d in
                   zip(df.hs_alisee, df.tp_alisee, df.vento_kn, df.vento_dir)]

    # ── Alba/tramonto: le ore di buio non si surfano — in grafico si scuriscono
    # e le finestre si calcolano solo sulle ore di luce.
    try:
        rs = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": LAT_SPOT, "longitude": LON_SPOT,
            "daily": "sunrise,sunset", "timezone": "Europe/Rome",
            "forecast_days": 4}, timeout=90).json()["daily"]
        sole = {pd.Timestamp(t).date(): (pd.Timestamp(a), pd.Timestamp(b))
                for t, a, b in zip(rs["time"], rs["sunrise"], rs["sunset"])}
        df["luce"] = [bool(sole.get(t.date()) and
                           sole[t.date()][0] <= t <= sole[t.date()][1])
                      for t in df.date]
    except Exception:
        df["luce"] = df.date.dt.hour.between(6, 20)   # ripiego prudente

    # ── Fascia probabile per ogni ora (interpolazione dei quantili misurati)
    df["hs_p10"] = np.minimum(np.interp(df.hs_alisee, _BANDA_PRED, _BANDA_P10),
                              df.hs_alisee)
    df["hs_p90"] = np.maximum(np.interp(df.hs_alisee, _BANDA_PRED, _BANDA_P90),
                              df.hs_alisee)

    ora = pd.Timestamp.now().floor("h")
    return df[df.date >= ora].reset_index(drop=True)


def finestre(df, soglia=0.8):
    """Finestre surfabili DIURNE: onda >= soglia nelle ore di luce. Una finestra
    alle 3 di notte non serve a nessuno — non va annunciata."""
    surf = df[(df.hs_alisee >= soglia) & (df.luce)].copy()
    out = []
    if not surf.empty:
        surf["blk"] = (surf["date"].diff() > pd.Timedelta("1h")).cumsum()
        for _, g in surf.groupby("blk"):
            imax = g.hs_alisee.idxmax()
            out.append((g.date.min(), g.date.max(), g.loc[imax, "hs_alisee"],
                        g.loc[imax, "tp_alisee"],
                        onda_cardinale(g.loc[imax, "wave_direction"]),
                        g.loc[imax, "hs_p10"], g.loc[imax, "hs_p90"]))
    return out


def _bussola(direzione, size=64):
    """Rosa con freccia nel verso di propagazione (l'onda VIENE da `direzione`)."""
    d = float(direzione or 0)
    return f"""<svg viewBox="0 0 64 64" width="{size}" height="{size}">
  <circle cx="32" cy="32" r="27" fill="none" stroke="#30363d" stroke-width="1"/>
  <text x="32" y="10" text-anchor="middle" fill="#6e7681" font-size="8">N</text>
  <text x="32" y="60" text-anchor="middle" fill="#6e7681" font-size="8">S</text>
  <text x="58" y="35" text-anchor="middle" fill="#6e7681" font-size="8">E</text>
  <text x="6"  y="35" text-anchor="middle" fill="#6e7681" font-size="8">O</text>
  <g transform="rotate({d + 180:.0f} 32 32)">
    <path d="M32 14 L38 42 L32 36 L26 42 Z" fill="#58a6ff"/>
  </g></svg>"""


def _chart(df, W=940):
    """Grafico 72h su due tracce: ONDA sopra (barre per stato + fascia probabile
    misurata + linea NWP) e VENTO sotto (barre verdi=amico / ambra=contrario).
    Le ore di buio sono scurite: di notte non si surfa."""
    n = len(df)
    pl, pr, pt = 44, 14, 22
    h_wave, gap, h_wind, pb = 148, 36, 44, 24
    H = pt + h_wave + gap + h_wind + pb
    bw = (W - pl - pr) / max(n, 1)
    hs_max = max(1.0, float(df.hs_p90.max()) * 1.08)
    wk_max = max(12.0, float(df.vento_kn.max()) * 1.15)
    y_wave0 = pt + h_wave                     # base traccia onda
    y_wind0 = pt + h_wave + gap + h_wind      # base traccia vento

    def y_of(v): return pt + h_wave * (1 - v / hs_max)
    def wy(v):   return y_wind0 - h_wind * (min(v, wk_max) / wk_max)

    # ── Notte: colonne scurite su entrambe le tracce
    notte, i = "", 0
    while i < n:
        if not bool(df.luce.iloc[i]):
            j = i
            while j < n and not bool(df.luce.iloc[j]):
                j += 1
            x0, x1 = pl + i * bw, pl + j * bw
            notte += (f'<rect x="{x0:.1f}" y="{pt}" width="{x1-x0:.1f}" '
                      f'height="{y_wind0-pt:.1f}" fill="#010409" opacity="0.5"/>')
            i = j
        else:
            i += 1

    # ── Griglia
    grid, step, v = "", (0.5 if hs_max <= 2.5 else 1.0), 0.0
    while v <= hs_max:
        yy = y_of(v)
        grid += (f'<line x1="{pl}" y1="{yy:.1f}" x2="{W-pr}" y2="{yy:.1f}" '
                 f'stroke="#ffffff14" stroke-width="1"/>'
                 f'<text x="{pl-5}" y="{yy+3:.1f}" text-anchor="end" fill="#6e7681" '
                 f'font-size="11">{v:.1f}</text>')
        v += step
    if wk_max > 10:
        grid += (f'<line x1="{pl}" y1="{wy(10):.1f}" x2="{W-pr}" y2="{wy(10):.1f}" '
                 f'stroke="#ffffff10" stroke-width="1"/>'
                 f'<text x="{pl-5}" y="{wy(10)+3:.1f}" text-anchor="end" fill="#6e7681" '
                 f'font-size="10">10</text>')

    # ── Fascia probabile misurata: area morbida dietro le barre
    xs  = [pl + (i + 0.5) * bw for i in range(n)]
    su  = [f"{x:.1f},{y_of(float(p)):.1f}" for x, p in zip(xs, df.hs_p90)]
    giu = [f"{x:.1f},{y_of(float(p)):.1f}" for x, p in zip(xs, df.hs_p10)]
    banda = (f'<polygon points="{" ".join(su + giu[::-1])}" fill="#58a6ff" '
             f'opacity="0.12"/>')

    bars, wbars, nwp, daysep, daylab = "", "", [], "", ""
    last_day = None
    for i, (_, r) in enumerate(df.iterrows()):
        x = pl + i * bw
        dim = "" if bool(r.luce) else ' fill-opacity="0.5"'
        yb = y_of(float(r.hs_alisee))
        bars += (f'<rect x="{x+bw*0.12:.1f}" y="{yb:.1f}" width="{bw*0.76:.1f}" '
                 f'height="{y_wave0-yb:.1f}" fill="{STATI[r.stato][1]}" rx="1"{dim}>'
                 f'<title>{r.date:%d/%m %H:%M} — onda {r.hs_alisee:.1f} m '
                 f'(probabile {r.hs_p10:.1f}–{r.hs_p90:.1f}) · {r.tp_alisee:.0f}s · '
                 f'vento {r.vento_kn:.0f} kn {onda_cardinale(r.vento_dir)}</title></rect>')
        _, wcol = _vento_label(float(r.vento_kn), float(r.vento_dir))
        yw = wy(float(r.vento_kn))
        wbars += (f'<rect x="{x+bw*0.12:.1f}" y="{yw:.1f}" width="{bw*0.76:.1f}" '
                  f'height="{y_wind0-yw:.1f}" fill="{wcol}" rx="1"{dim}>'
                  f'<title>{r.date:%d/%m %H:%M} — vento {r.vento_kn:.0f} kn '
                  f'{onda_cardinale(r.vento_dir)}</title></rect>')
        nwp.append(f"{x+bw/2:.1f},{y_of(float(r.wave_height)):.1f}")
        d = r.date.date()
        if d != last_day:
            if last_day is not None:
                daysep += (f'<line x1="{x:.1f}" y1="{pt}" x2="{x:.1f}" y2="{y_wind0}" '
                           f'stroke="#30363d" stroke-width="1" stroke-dasharray="3 3"/>')
            daylab += (f'<text x="{x+3:.1f}" y="{y_wind0+16:.0f}" fill="#8b949e" '
                       f'font-size="11">{gg(r.date)} {r.date:%d/%m}</text>')
            last_day = d

    nwp_line = (f'<polyline points="{" ".join(nwp)}" fill="none" stroke="#8b949e" '
                f'stroke-width="1.2" stroke-dasharray="4 3" opacity="0.65"/>')
    capt = (f'<text x="{pl}" y="{pt-8}" fill="#8b949e" font-size="11">onda (m)</text>'
            f'<text x="{pl}" y="{y_wind0-h_wind-9}" fill="#8b949e" font-size="11">'
            f'vento (kn) — <tspan fill="#3fb950">verde: amico</tspan> · '
            f'<tspan fill="#d29922">ambra: contrario</tspan></text>')
    return (f'<svg viewBox="0 0 {W} {H}" width="100%">'
            f'{notte}{grid}{banda}{daysep}{bars}{nwp_line}{wbars}{daylab}{capt}</svg>')


def _giorni(df):
    """Riepilogo per giorno: max Hs, stato al picco, finestra migliore."""
    out = []
    for d, g in df.groupby(df.date.dt.date):
        imax = g.hs_alisee.idxmax()
        r = g.loc[imax]
        surf = g[(g.hs_alisee >= 0.8) & (g.luce)]        # solo ore di luce
        win = (f"{surf.date.min():%H:%M}–{surf.date.max():%H:%M}"
               if not surf.empty else "—")
        out.append({"data": pd.Timestamp(d), "hs": r.hs_alisee, "tp": r.tp_alisee,
                    "dir": onda_cardinale(r.wave_direction), "stato": r.stato,
                    "win": win, "vento": r.vento_kn,
                    "vdir": onda_cardinale(r.vento_dir)})
    return out[:3]


def _accuratezza():
    """Blocco VERIFICA in linguaggio semplice: quanto si sbaglia in media (unita'
    reali), e quanto errore in meno rispetto al modello standard. Niente ±, niente
    gergo: 'sbaglia di 12 cm' lo capisce chiunque."""
    def it(v, dec):
        """Numero con la virgola decimale italiana."""
        return f"{v:.{dec}f}".replace(".", ",")

    righe = ""
    for cosa, ali, std, u, dec, migl in SKILL:
        quota = ali / std * 100        # barra: il nostro errore rispetto al loro
        righe += (
            f'<div class="ac">'
            f'<div class="acn">{cosa} <span class="tag">{migl}% di errore in meno</span></div>'
            f'  <div class="acr"><span class="acl">ALISEE</span>'
            f'    <i style="width:{quota:.0f}%;background:#3fb950"></i>'
            f'    <b>{it(ali, dec)} {u}</b></div>'
            f'  <div class="acr"><span class="acl">standard</span>'
            f'    <i style="width:100%;background:#484f58"></i>'
            f'    <b style="color:#8b949e">{it(std, dec)} {u}</b></div>'
            f'</div>')
    return righe


def _legenda():
    return "".join(f'<span class="lg"><i style="background:{c}"></i>{lab}</span>'
                   for lab, c in _LEGENDA)


# In modalita' embed la pagina vive dentro un iframe sul sito del cliente:
# sfondo trasparente (si adatta alla loro pagina) e niente padding esterno.
CSS_EMBED = """
body{background:transparent;padding:0}
.wrap{max-width:none}
"""

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#e6edf3;font-family:-apple-system,Segoe UI,Roboto,Helvetica,sans-serif;
     padding:24px;line-height:1.5;-webkit-font-smoothing:antialiased}
.wrap{max-width:1000px;margin:0 auto}
.top{display:flex;justify-content:space-between;align-items:center;margin-bottom:18px;flex-wrap:wrap;gap:8px}
h1{font-size:19px;font-weight:600;letter-spacing:-.01em}
h1 span{color:#58a6ff}
h1 .x{color:#6e7681;font-weight:400;margin:0 2px}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:#3fb950;margin-right:6px}
.upd{font-size:12px;color:#6e7681}
.hero{display:grid;grid-template-columns:1.25fr 1fr;gap:14px;margin-bottom:14px}
.card{background:#161b22;border:1px solid #21262d;border-radius:12px;padding:16px 18px}
.k{font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px}
.now{display:flex;align-items:center;gap:18px}
.hs{font-size:40px;font-weight:600;line-height:1;letter-spacing:-.02em}
.meta{font-size:13px;color:#8b949e;margin-top:6px}
.pill{display:inline-block;font-size:11px;font-weight:600;padding:2px 10px;border-radius:20px;color:#0d1117}
.mini{display:flex;gap:10px;margin-top:14px;padding-top:12px;border-top:1px solid #21262d}
.mini div{flex:1}
.mini .v{font-size:15px;font-weight:600}
.mini .l{font-size:11px;color:#6e7681}
.bar{height:5px;border-radius:3px;background:#30363d;overflow:hidden;margin-top:5px}
.bar i{display:block;height:100%;background:#58a6ff}
.big{font-size:26px;font-weight:600;line-height:1.15}
.sub{font-size:13px;color:#8b949e;margin-top:3px}
.chart{background:#161b22;border:1px solid #21262d;border-radius:12px;padding:14px 14px 8px;margin-bottom:14px}
.ct{display:flex;justify-content:space-between;font-size:12px;color:#8b949e;margin:0 2px 10px}
.legend{display:flex;gap:14px;flex-wrap:wrap;padding:8px 2px 0;font-size:11px;color:#8b949e}
.lg i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px}
.days{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:14px}
.day .d{font-size:12px;color:#8b949e;text-transform:capitalize;margin-bottom:6px}
.day .h{font-size:22px;font-weight:600}
.day .w{font-size:12px;color:#6e7681;margin-top:4px}
.foot{font-size:11px;color:#6e7681;border-top:1px solid #21262d;padding-top:12px;line-height:1.6}
.brand{margin-top:10px;font-size:12px;color:#6e7681}
.brand b{color:#58a6ff;letter-spacing:.03em}
.acc{background:#161b22;border:1px solid #21262d;border-radius:12px;padding:14px 18px;margin-bottom:14px}
.acch{font-size:13px;font-weight:600;margin-bottom:3px}
.accs{font-size:11px;color:#6e7681;margin-bottom:12px}
.acgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}
.acn{font-size:11px;color:#8b949e;margin-bottom:6px;text-transform:uppercase;letter-spacing:.04em}
.tag{display:inline-block;background:#1b2c20;color:#3fb950;font-size:10px;font-weight:600;
     padding:1px 7px;border-radius:10px;margin-left:6px;text-transform:none;letter-spacing:0}
.acr{display:flex;align-items:center;gap:7px;margin-bottom:4px;font-size:11px}
.acl{color:#6e7681;width:48px;flex:none}
.acr i{display:block;height:7px;border-radius:3px;max-width:110px}
.acr b{font-size:12px;font-weight:600;white-space:nowrap}
.acex{margin-top:14px;padding-top:11px;border-top:1px solid #21262d;font-size:12px;color:#8b949e}
.acex b{color:#e6edf3;font-weight:600}
@media(max-width:720px){.acgrid{grid-template-columns:1fr}}
@media(max-width:720px){.hero,.days{grid-template-columns:1fr}}
"""


def _vento_label(w_kn, w_dir):
    """Etichetta + colore del vento in ottica surf (non la sua forza in se')."""
    off = comp_offshore(w_dir)
    if w_kn < 5:                 return "calmo", "#3fb950"
    if off > 0.3:                return "offshore", "#3fb950"    # pulisce l'onda
    if off < -0.3 and w_kn >= 12: return "onshore teso", "#d29922"
    if off < -0.3:               return "onshore", "#8b949e"
    return "laterale", "#8b949e"


def build_dashboard(df, wins, embed=False):
    """Genera la pagina. embed=False -> dashboard.html (pagina completa).
    embed=True -> widget.html: STESSA pagina (e' quella che funziona), ma
    trasparente e senza cornice, pronta per l'iframe sul sito del cliente."""
    now = df.iloc[0]
    pk = df.loc[df.hs_alisee.idxmax()]
    lbl, col = STATI[now.stato]
    sst = float(now.get("sea_surface_temperature", float("nan")))
    sst_txt = f"{sst:.0f}°" if sst == sst else "—"
    swp = float(now.swell_pct)
    v_lbl, v_col = _vento_label(float(now.vento_kn), float(now.vento_dir))

    if wins:
        a, b, hs, tp, dr, w10, w90 = wins[0]
        win_html = (f'<div class="big">{hs:.1f} m</div>'
                    f'<div class="sub">{gg(a)} {a:%d} · {a:%H:%M}–{b:%H:%M} · {tp:.0f}s {dr}'
                    f'<br>probabile tra {w10:.1f} e {w90:.1f} m</div>')
    else:
        win_html = ('<div class="big" style="color:#6e7681">—</div>'
                    '<div class="sub">nessuna onda ≥0,8 m nelle ore di luce delle prossime 72h</div>')

    giorni = "".join(
        f'<div class="card day"><div class="d">{gg(g["data"])} {g["data"]:%d/%m}</div>'
        f'<div class="h">{g["hs"]:.1f} m <span class="pill" style="background:'
        f'{STATI[g["stato"]][1]};font-size:10px">{STATI[g["stato"]][0]}</span></div>'
        f'<div class="w">picco {g["tp"]:.0f}s {g["dir"]} · vento {g["vento"]:.0f}kn '
        f'{g["vdir"]} · finestra {g["win"]}</div></div>'
        for g in _giorni(df))

    chart = _chart(df)
    ultima, prossima = orari_run()
    pross_txt = (f" · prossima {gg(prossima)} {prossima:%H:%M}" if prossima else "")

    # Marchio: solo ALISEE, oppure "ALISEE × Partner" se il co-branding e' acceso.
    marchio = (f'ALISEE <span class="x">×</span> {PARTNER}' if PARTNER
               else 'ALISEE <span>Onda</span>')

    # Sulla pagina del cliente il nome dello spot e' gia' nel titolo della loro
    # pagina: qui conta la previsione, e il marchio.
    titolo = (f'<h1><span class="dot"></span>{marchio}</h1>' if embed
              else f'<h1><span class="dot"></span>{marchio} · {SPOT}</h1>')
    sotto_embed = (f'previsione onda e vento · 72h · aggiornata {gg(ultima)} {ultima:%H:%M}')
    firma = (f'<div class="upd">{sotto_embed}</div>' if embed else
             f'<div class="upd">ultima run <b style="color:#8b949e">{gg(ultima)} '
             f'{ultima:%d/%m %H:%M}</b> (ora italiana){pross_txt} · boa RON Civitavecchia'
             f' · previsione 72h</div>')

    # Precisione: blocco completo sulla dashboard; nel widget del cliente una riga
    # sola (il surfista vuole l'onda, la prova estesa serve alla trattativa).
    if embed:
        vals = " · ".join(
            f"{c} {f'{a:.{d}f}'.replace('.', ',')} {u} "
            f"(standard: {f'{s:.{d}f}'.replace('.', ',')})"
            for c, a, s, u, d, m in SKILL)
        acc_html = (f'<div class="acc"><div class="accs" style="margin:0">'
                    f'Verificata sul mare vero — errore medio su ~{SKILL_ORE} ore di '
                    f'confronto con boa e stazione: {vals}.</div></div>')
    else:
        acc_html = f"""<div class="acc">
  <div class="acch">Verificata sul mare vero</div>
  <div class="accs">Abbiamo confrontato ~{SKILL_ORE} ore di previsioni con quello che la boa
    e la stazione hanno poi misurato davvero. Qui sotto: di quanto si sbaglia in media.
    Barra più corta = previsione più fedele.</div>
  <div class="acgrid">{_accuratezza()}</div>
  <div class="acex">Un esempio concreto: quando la previsione dice <b>1,5 m</b>, 8 volte su 10
    il mare misurato è tra <b>1,1 e 1,8 m</b> — è la "fascia probabile" che vedi nel grafico.</div>
</div>"""

    doc = f"""<!doctype html><html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="1800">
<title>ALISEE Onda · {SPOT}</title>
<style>{CSS}{CSS_EMBED if embed else ""}</style></head><body><div class="wrap">
<div class="top">
  {titolo}
  {firma}
</div>

<div class="hero">
  <div class="card">
    <div class="k">adesso</div>
    <div class="now">
      {_bussola(now.wave_direction, 72)}
      <div>
        <div class="hs">{now.hs_alisee:.1f} <span style="font-size:18px;color:#8b949e">m</span></div>
        <div class="meta">{now.tp_alisee:.0f}s · da {onda_cardinale(now.wave_direction)}
          &nbsp;<span class="pill" style="background:{col}">{lbl}</span></div>
      </div>
    </div>
    <div class="mini">
      <div><div class="v" style="color:{v_col}">{now.vento_kn:.0f} kn</div>
        <div class="l">vento {onda_cardinale(now.vento_dir)} · {v_lbl}</div></div>
      <div><div class="v">{sst_txt}</div><div class="l">acqua</div></div>
      <div><div class="v">{swp:.0f}%</div><div class="l">mare lungo</div>
        <div class="bar"><i style="width:{swp:.0f}%"></i></div></div>
      <div><div class="v">{pk.hs_alisee:.1f} m</div><div class="l">picco · {gg(pk.date)} {pk.date:%H}h
        · prob. {pk.hs_p10:.1f}–{pk.hs_p90:.1f}</div></div>
    </div>
  </div>
  <div class="card">
    <div class="k">prossima finestra surfabile</div>
    {win_html}
    <div class="mini"><div><div class="l" style="line-height:1.6">Finestra = onda ≥0,8 m
      nelle ore di luce. "Probabile tra X e Y" = dove il mare reale è stato 8 volte su 10
      quando la previsione diceva così (misurato alla boa, non stimato).</div></div></div>
  </div>
</div>

<div class="chart">
  <div class="ct"><span>onda e vento · prossime 72 ore</span>
    <span>— — modello standard</span></div>
  {chart}
  <div class="legend">{_legenda()}
    <span class="lg"><i style="background:#58a6ff;opacity:.3"></i>fascia probabile</span>
    <span class="lg"><i style="background:#010409;border:1px solid #30363d"></i>notte</span></div>
</div>

<div class="days">{giorni}</div>

{acc_html}

<div class="foot">
  Misurato su ~{SKILL_ORE} ore di confronto con la <b>boa ondametrica RON</b> (onda) e la
  <b>stazione RMN di Civitavecchia</b> (vento), entrambe a ~8 km dallo spot, su periodi che il
  modello non aveva mai visto in addestramento. "Standard" = il modello meteo pubblico da cui
  partono i siti di previsione. I valori sono l'errore sull'analisi: su una previsione a 72 ore
  di anticipo entrambi sbagliano di più.
</div>
<div class="brand"><b>ALISEE</b> · weather intelligence — previsioni calibrate su strumenti reali</div>
</div></body></html>"""
    nome = "widget.html" if embed else "dashboard.html"
    with open(os.path.join(BASE, nome), "w", encoding="utf-8") as f:
        f.write(doc)


if __name__ == "__main__":
    df = scarica_e_prevedi()
    wins = finestre(df)

    print("=" * 60)
    print(f"  ALISEE ONDA — {SPOT} (boa RON Civitavecchia)  |  72h")
    print("=" * 60)
    print(f"{'quando':16s} {'Hs':>7s} {'Tp':>6s} {'dir':>5s}  stato")
    for _, r in df.iloc[::2].iterrows():
        print(f"{r['date']:%a %d/%m %H:%M} {r.hs_alisee:5.2f} m {r.tp_alisee:5.1f}s "
              f"{onda_cardinale(r.wave_direction):>5s}  {r.stato}")
    print("\n--- FINESTRE DIURNE (Hs>=0.8m, ore di luce) ---")
    if wins:
        for a, b, hs, tp, d, p10, p90 in wins:
            print(f"  {a:%a %d/%m %H:%M} -> {b:%H:%M}  fino a {hs:.1f}m "
                  f"(prob {p10:.1f}-{p90:.1f}) {tp:.0f}s {d}")
    else:
        print("  nessuna nelle prossime 72h")

    df[["date", "hs_alisee", "hs_p10", "hs_p90", "tp_alisee", "wave_direction",
        "wave_height", "swell_pct", "vento_kn", "vento_dir", "luce", "stato"]] \
        .to_csv(os.path.join(BASE, "previsione_onda_72h.csv"), index=False)
    build_dashboard(df, wins)                  # dashboard.html — pagina completa
    build_dashboard(df, wins, embed=True)      # widget.html — stessa pagina, per iframe
    print("\nprevisione_onda_72h.csv + dashboard.html + widget.html aggiornati.")
