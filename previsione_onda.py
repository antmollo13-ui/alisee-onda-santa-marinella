"""
ALISEE — previsione onda + vento 72h per Santa Marinella. Il prodotto.
Carica modello_onda.pkl e modello_vento.pkl, scarica i forecast (boa per l'onda,
punto spot per il vento), applica la calibrazione e RIGENERA a ogni run:
  - dashboard.html : pagina completa
  - widget.html    : LA STESSA pagina, trasparente e senza cornice, per l'iframe
                     sul sito del cliente (una miniatura compressa faceva schifo:
                     il cliente incorpora la pagina buona, non una versione ridotta)
"""
import os, json, pickle, datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests
from onda_features import MARINE_HOURLY, build_features, onda_cardinale
from vento_features import (FEAT as FEAT_V, MODELLI, NWP_HOURLY, NWP_850,
                            build_features as feat_vento, DIR_OFFSHORE)
from stile import (STATI, TINT, _LEGENDA, CSS, CSS_EMBED, SFONDO,
                   JS_CHART, _bussola, _sparkline, _legenda)

# Fuso ancorato: sul cloud (GitHub Actions) il sistema e' in UTC — senza questo
# la dashboard mostrerebbe l'ora UTC spacciandola per ora italiana.
ROMA = ZoneInfo("Europe/Rome")
# Ore UTC del cron (allineate all'uscita delle run del modello d'onda)
CRON_UTC = [5, 11, 17, 23]

# Percorso portabile: la cartella dello script (funziona su Windows e su Linux/cloud)
BASE = os.path.dirname(os.path.abspath(__file__))
LAT_BOA, LON_BOA = 42.05, 11.70      # boa di Civitavecchia (onda)
LAT_SPOT, LON_SPOT = 42.034, 11.849  # spot (vento)
SPOT = "Santa Marinella"

# Co-branding. Si accende con la variabile d'ambiente PARTNER (es. "SurfCam Italia").
# Tenuto SPENTO di default: la pagina e' pubblica e il nome di un partner non va
# esposto finche' non c'e' un accordo. Per la demo:  set PARTNER=SurfCam Italia
PARTNER = os.environ.get("PARTNER", "").strip()

# Ganci commerciali opzionali (env var, come PARTNER — spenti di default):
#   CAM_URL -> bottone "Guarda la cam live": il momento buono spinge al live,
#              che e' il prodotto premium della piattaforma.
#   SPONSOR -> "previsione offerta da X": slot che la piattaforma puo' vendere
#              ai supporter locali = linea di ricavo nuova.
CAM_URL = os.environ.get("CAM_URL", "").strip()
SPONSOR = os.environ.get("SPONSOR", "").strip()
#   PREMIUM_URL -> pagina abbonamento della piattaforma: e' il bersaglio del
#                  widget FREEMIUM (widget-free.html), dove la previsione
#                  completa sta dietro il loro Premium = il forecast diventa
#                  un motivo per abbonarsi, cioe' un asset di guadagno.
PREMIUM_URL = os.environ.get("PREMIUM_URL", "").strip()
#   UPSELL_URL -> tier superiore (previsioni a lungo termine + alert personalizzati):
#                 la card "a pagamento" oltre il forecast base, nuova linea di ricavo.
UPSELL_URL = os.environ.get("UPSELL_URL", "").strip()
#   CF_BEACON  -> token Cloudflare Web Analytics: quante volte il widget viene
#                 caricato e DA QUALE SITO (referrer). Gratis, SENZA COOKIE e
#                 senza tracciamento individuale -> nessun banner consenso.
#                 E' l'unico modo per sapere se il widget viene davvero usato.
CF_BEACON = os.environ.get("CF_BEACON", "").strip()
#   EVENTI_URL -> endpoint (opzionale) per contare i click sui bottoni: il widget
#                 manda un ping "cam"/"premium"/"upsell" con sendBeacon.
EVENTI_URL = os.environ.get("EVENTI_URL", "").strip()


def _utm(url):
    """Aggiunge il tracciamento: ogni click dal widget e' attribuibile ad ALISEE.
    E' la prova, nei LORO analytics, di quanto il widget converte."""
    if not url:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}utm_source=alisee&utm_medium=widget"


def _analytics():
    """Misura d'uso del widget. Cloudflare Web Analytics: nessun cookie, nessun
    profilo individuale -> niente banner consenso, utilizzabile anche dentro il
    sito di un cliente. Piu' (opzionale) il ping dei click sui bottoni."""
    out = ""
    if CF_BEACON:
        out += ('<script defer src="https://static.cloudflareinsights.com/beacon.min.js" '
                f"data-cf-beacon='{{\"token\":\"{CF_BEACON}\"}}'></script>")
    if EVENTI_URL:
        out += ("<script>document.addEventListener('click',function(e){"
                "var a=e.target.closest('a[data-ev]');if(!a)return;"
                "try{navigator.sendBeacon('" + EVENTI_URL + "',"
                "JSON.stringify({ev:a.dataset.ev,spot:'" + SPOT + "',t:Date.now()}));}"
                "catch(_){}}); </script>")
    return out

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

# Scala misurata con una REGRESSIONE sulle stesse ore (regressione.py): la
# pendenza dice se il modello rispetta le proporzioni. 1 = scala giusta;
# sotto 1 = comprime, cioe' piu' il mare cresce piu' lo sottostima.
SCALA = [("onda",  0.875, 0.985), ("vento", 0.840, 1.011)]

# ── Regole dello spot (dalle schede SurfCamItalia di Santa Marinella:
#    Banzai "swell da W, SW e NW"; Supertubos "swell ideale da W e SW,
#    con mare molto grande diventa impegnativo").
SWELL_SETTORE = (190, 320)   # da SSW a NW: il settore di swell che accende gli spot
HS_GROSSO     = 2.8          # m: sopra, tecnico/per esperti -> niente piu' "buono"

# Fascia probabile MISURATA (verita_intervalli.py, test 2026): per ogni fascia di
# valore PREVISTO, dove sta il mare reale 8 volte su 10 (quantili osservati alla
# boa, NON dedotti dal MAE — il MAE medio sottostima l'incertezza sulle mareggiate).
_BANDA_PRED = [0.20, 0.50, 0.70, 0.90, 1.15, 1.50, 1.95, 2.60]
_BANDA_P10  = [0.13, 0.32, 0.46, 0.62, 0.83, 1.11, 1.36, 2.01]
_BANDA_P90  = [0.40, 0.62, 0.83, 1.08, 1.44, 1.78, 2.43, 3.25]

_RANK = {"piatto": 0, "piccolo": 1, "mosso/corto": 2, "surfabile": 3, "BUONO": 4}


def momento_migliore(df):
    """Il momento surfabile migliore nelle ore di luce (rank stato, poi Hs).
    E' il 'quando vale la pena' — riempie la card destra con un verdetto vero."""
    luce = df[df.luce & (df.hs_alisee >= 0.6)]
    if luce.empty:
        return None
    luce = luce.assign(_r=luce.stato.map(_RANK))
    best = luce.sort_values(["_r", "hs_alisee"], ascending=False).iloc[0]
    if best._r < 2:        # niente di meglio di "piccolo": non e' un consiglio
        return None
    return best


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


def _swell_giusto(sw_dir):
    """True se lo swell arriva dal settore che accende gli spot (W/SW/NW)."""
    if sw_dir is None or sw_dir != sw_dir:
        return True                       # direzione ignota: non penalizzare
    lo, hi = SWELL_SETTORE
    return lo <= float(sw_dir) % 360 <= hi


def giudizio(hs, tp, w_kn=0.0, w_dir=0.0, sw_dir=None):
    """Qualita' surf: onda (size + periodo + DIREZIONE dello swell) modulata dal
    vento. "BUONO" richiede tutto insieme: taglia giusta (ne' piccola ne' da
    esperti), mare formato, swell dal settore giusto, vento amico."""
    if hs < 0.5:  return "piatto"
    if hs < 0.8:  return "piccolo"
    off = comp_offshore(w_dir)
    if w_kn >= 12 and off < -0.3:  return "mosso/corto"   # onshore teso: chop
    if tp < 5:                      return "mosso/corto"   # mare corto da vento
    if (1.2 <= hs <= HS_GROSSO and tp >= 6 and _swell_giusto(sw_dir)
            and (w_kn < 8 or off > 0.3)):
        return "BUONO"
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
    # 5 giorni: 72h piene in grafico + giorni 4-5 come "tendenza" (motivo per
    # tornare domani a controllare come evolve = visite ricorrenti).
    dm = _get("https://marine-api.open-meteo.com/v1/marine",
              {"latitude": LAT_BOA, "longitude": LON_BOA,
               "hourly": MARINE_HOURLY + ",sea_surface_temperature",
               "timezone": "Europe/Rome", "forecast_days": 5})
    df = build_features(dm)
    df["hs_alisee"] = np.clip(MO["hs"].predict(df[MO["feat"]]), 0, None)
    df["tp_alisee"] = np.clip(MO["tp"].predict(df[MO["feat"]]), 0, None)
    tot = (df["swell_wave_height"].fillna(0) + df["wind_wave_height"].fillna(0)).clip(lower=0.01)
    df["swell_pct"] = (df["swell_wave_height"].fillna(0) / tot * 100).clip(0, 100)

    # ── VENTO (spot) — atmosferico + 850hPa + SST della boa
    da = _get("https://api.open-meteo.com/v1/forecast",
              {"latitude": LAT_SPOT, "longitude": LON_SPOT, "hourly": NWP_HOURLY,
               "models": ",".join(MODELLI), "wind_speed_unit": "kn",
               "timezone": "Europe/Rome", "forecast_days": 5})
    d8 = _get("https://api.open-meteo.com/v1/forecast",
              {"latitude": LAT_SPOT, "longitude": LON_SPOT, "hourly": NWP_850,
               "models": "icon_seamless", "wind_speed_unit": "kn",
               "timezone": "Europe/Rome", "forecast_days": 5})
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
    df["stato"] = [giudizio(h, t, w, d, s) for h, t, w, d, s in
                   zip(df.hs_alisee, df.tp_alisee, df.vento_kn, df.vento_dir,
                       df.wave_direction)]

    # ── Alba/tramonto: le ore di buio non si surfano — in grafico si scuriscono
    # e le finestre si calcolano solo sulle ore di luce.
    try:
        rs = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": LAT_SPOT, "longitude": LON_SPOT,
            "daily": "sunrise,sunset", "timezone": "Europe/Rome",
            "forecast_days": 6}, timeout=90).json()["daily"]
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


def archivia_previsioni(df):
    """Scrive le previsioni di QUESTA run nell'archivio (run_time, target_time,
    lead_h, valori ALISEE + NWP grezzo come benchmark). E' la meta' "promesse"
    del confronto con la verita': senza archivio niente pagella reale.
    Sul cloud l'archivio viene ricommittato nel repo a ogni run."""
    if os.environ.get("NO_ARCHIVIO"):
        return                          # la run demo non archivia (sarebbe un doppione)
    run_ts = datetime.datetime.now(ROMA).replace(tzinfo=None, second=0, microsecond=0)
    rows = df.copy()
    rows["run_time"] = run_ts
    rows["lead_h"] = ((rows["date"] - run_ts).dt.total_seconds() / 3600).round(1)
    rows = rows[rows["lead_h"] >= 0]
    out = rows[["run_time", "date", "lead_h", "hs_alisee", "tp_alisee",
                "wave_height", "vento_kn", "vento_icon", "vento_dir",
                "wave_direction"]].rename(columns={"date": "target_time",
                                                   "wave_height": "hs_nwp"})
    p = os.path.join(BASE, "archivio_previsioni.csv")
    out.round(2).to_csv(p, mode="a", header=not os.path.isfile(p), index=False)
    log_n = len(out)
    print(f"[ARCHIVIO] +{log_n} previsioni archiviate (lead 0-{out.lead_h.max():.0f}h)")


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




def _dati_json(df):
    """Serializza le ore per il grafico interattivo (colori gia' risolti qui,
    cosi' il JS non deve conoscere le regole dello spot)."""
    out = []
    # Tendenza a 6 ore: per un surfista "sta montando o scemando?" conta quanto
    # l'altezza stessa. +1 in aumento, -1 in calo, 0 stabile (soglia 10 cm).
    hs = df.hs_alisee.tolist()
    trend = []
    for i in range(len(hs)):
        j = min(i + 6, len(hs) - 1)
        d = hs[j] - hs[i]
        trend.append(1 if d > 0.10 else (-1 if d < -0.10 else 0))
    for k, (_, r) in enumerate(df.iterrows()):
        lab, col = STATI[r.stato]
        _, wcol = _vento_label(float(r.vento_kn), float(r.vento_dir))
        sst = float(r.get("sea_surface_temperature", float("nan")))
        out.append({
            "gg": gg(r.date), "dm": f"{r.date:%d/%m}", "hh": f"{r.date:%H:%M}",
            "h": round(float(r.hs_alisee), 2), "a": round(float(r.hs_p10), 2),
            "b": round(float(r.hs_p90), 2), "p": round(float(r.tp_alisee), 1),
            "nw": round(float(r.wave_height), 2),
            "wd": round(float(r.wave_direction or 0)),
            "wdc": onda_cardinale(r.wave_direction),
            "k": round(float(r.vento_kn), 1),
            "kdc": onda_cardinale(r.vento_dir),
            "kl": _vento_label(float(r.vento_kn), float(r.vento_dir))[0],
            "s": lab, "sc": col, "wc": wcol, "l": 1 if r.luce else 0,
            "w": (round(sst) if sst == sst else None),
            "sp": round(float(r.swell_pct)),
            "tb": TINT[r.stato][0], "tf": TINT[r.stato][1], "td": TINT[r.stato][2],
            "tr": trend[k],
        })
    return out


def scrivi_snapshot(df, wins, best):
    """snapshot.json nel formato che il connettore MCP ALISEE sa gia' leggere.
    Pubblicato su Pages, diventa la sorgente dati per Claude: da qui nascono
    alert e contenuti senza lavoro manuale. Campi 'w_ai/wave/dir' = contratto
    del server esistente; i campi hs_*/tp/stato sono l'estensione surf."""
    ultima, prossima = orari_run()
    fc = []
    for _, r in df.iterrows():
        fc.append({
            "date": r.date.strftime("%Y-%m-%d %H:%M"),
            "w_ai": round(float(r.vento_kn), 1),          # vento (contratto MCP)
            "dir": round(float(r.vento_dir or 0)),
            "cardinal": onda_cardinale(r.vento_dir),
            "wave": round(float(r.hs_alisee), 2),         # onda (contratto MCP)
            "hs_p10": round(float(r.hs_p10), 2),
            "hs_p90": round(float(r.hs_p90), 2),
            "tp_s": round(float(r.tp_alisee), 1),
            "wave_dir": round(float(r.wave_direction or 0)),
            "wave_cardinal": onda_cardinale(r.wave_direction),
            "stato": STATI[r.stato][0],
            "luce": bool(r.luce),
        })
    snap = {
        "spot": {"id": "santa-marinella", "name": f"{SPOT} (boa di Civitavecchia)",
                 "lat": LAT_SPOT, "lon": LON_SPOT, "tipo": "surf"},
        "generato": ultima.strftime("%Y-%m-%d %H:%M"),
        "prossimo_aggiornamento": prossima.strftime("%Y-%m-%d %H:%M") if prossima else None,
        "forecast": fc,
        "finestre": [{"da": a.strftime("%Y-%m-%d %H:%M"), "a": b.strftime("%H:%M"),
                      "hs_max": round(float(hs), 2), "tp_s": round(float(tp), 1),
                      "dir": d, "probabile": [round(float(p10), 2), round(float(p90), 2)]}
                     for a, b, hs, tp, d, p10, p90 in wins],
        "momento_migliore": (None if best is None else {
            "quando": best.date.strftime("%Y-%m-%d %H:%M"),
            "hs": round(float(best.hs_alisee), 2), "tp_s": round(float(best.tp_alisee), 1),
            "dir": onda_cardinale(best.wave_direction),
            "vento_kn": round(float(best.vento_kn), 1), "stato": STATI[best.stato][0]}),
        "precisione": {c: {"alisee": a, "standard": s, "unita": u} for c, a, s, u, _, _ in SKILL},
        "regole_spot": {"swell_settore": list(SWELL_SETTORE), "hs_grosso_m": HS_GROSSO,
                        "soglia_finestra_m": 0.8},
    }
    with open(os.path.join(BASE, "snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, separators=(",", ":"))




# Motore del grafico interattivo: curve morbide, fascia probabile, notte, vento,
# crosshair che aggiorna il pannello "adesso" durante lo scrub, pillole giorno,
# countdown del prossimo aggiornamento. Vanilla JS, zero dipendenze.


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
                    "vdir": onda_cardinale(r.vento_dir),
                    "serie": [round(float(v), 2) for v in g.hs_alisee.tolist()]})
    return out[:3]


def _accuratezza():
    """Blocco VERIFICA in linguaggio semplice: quanto si sbaglia in media (unita'
    reali), e quanto errore in meno rispetto al modello standard. Niente ±, niente
    gergo: 'sbaglia di 12 cm' lo capisce chiunque."""
    def it(v, dec):
        """Numero con la virgola decimale italiana."""
        return f"{v:.{dec}f}".replace(".", ",")

    TRACK = 110                        # px: lunghezza della barra "standard" (=100%)
    righe = ""
    for cosa, ali, std, u, dec, migl in SKILL:
        # Larghezze in PX ASSOLUTI: in percentuale, il max-width del CSS le
        # schiacciava entrambe al tetto e le barre uscivano LUNGHE UGUALI
        # (bug visivo: "piu' corta = meglio" senza nessuna barra piu' corta).
        w_ali = TRACK * ali / std
        righe += (
            f'<div class="ac">'
            f'<div class="acn">{cosa} <span class="tag">{migl}% di errore in meno</span></div>'
            f'  <div class="acr"><span class="acl">ALISEE</span>'
            f'    <i style="width:{w_ali:.0f}px;background:#3fb950"></i>'
            f'    <b>{it(ali, dec)} {u}</b></div>'
            f'  <div class="acr"><span class="acl">standard</span>'
            f'    <i style="width:{TRACK}px;background:#484f58"></i>'
            f'    <b style="color:#8b949e">{it(std, dec)} {u}</b></div>'
            f'</div>')
    return righe




# In modalita' embed la pagina vive dentro un iframe sul sito del cliente:




def _vento_label(w_kn, w_dir):
    """Etichetta + colore del vento in ottica surf (non la sua forza in se')."""
    off = comp_offshore(w_dir)
    if w_kn < 5:                 return "calmo", "#3fb950"
    if off > 0.3:                return "offshore", "#3fb950"    # pulisce l'onda
    if off < -0.3 and w_kn >= 12: return "onshore teso", "#d29922"
    if off < -0.3:               return "onshore", "#8b949e"
    return "laterale", "#8b949e"


def build_dashboard(df, wins, embed=False, gate=False):
    """Genera la pagina. embed=False -> dashboard.html (pagina completa).
    embed=True -> widget.html: stessa pagina, trasparente, per l'iframe.
    gate=True  -> widget-free.html: versione FREEMIUM per gli utenti gratuiti.
        Mostra solo le prime 24h; dice CHE esiste una finestra (e quanto e'
        grande il picco) ma NON quando: giorno e orario stanno dietro il
        Premium della piattaforma. Il forecast diventa merce loro."""
    # 72h piene per grafico/card; i giorni 4-5 diventano la "tendenza"
    t0 = df.date.iloc[0]
    df72  = df[df.date <  t0 + pd.Timedelta("72h")]
    oltre = df[df.date >= t0 + pd.Timedelta("72h")]
    tnd = [(pd.Timestamp(d), float(g.hs_alisee.max()))
           for d, g in oltre.groupby(oltre.date.dt.date)][:2]

    now = df72.iloc[0]
    pk = df72.loc[df72.hs_alisee.idxmax()]
    lbl, col = STATI[now.stato]
    sst = float(now.get("sea_surface_temperature", float("nan")))
    sst_txt = f"{sst:.0f}°" if sst == sst else "—"
    swp = float(now.swell_pct)
    v_lbl, v_col = _vento_label(float(now.vento_kn), float(now.vento_dir))
    # Nel free si vede QUANTO fara' (il picco), ma non QUANDO: quello e' premium.
    pk_l = ("picco 5 giorni · quando? con Premium" if gate else
            f"picco · {gg(pk.date)} {pk.date:%H}h · prob. {pk.hs_p10:.1f}–{pk.hs_p90:.1f}")

    best = momento_migliore(df72)
    best_k = "momento migliore · 72h"
    if gate:
        # L'esca: c'e' una finestra ma il QUANDO e' premium.
        best_k = "in arrivo"
        if wins:
            best_html = (f'<div class="best-day">{wins[0][2]:.1f} m in arrivo</div>'
                         f'<div class="best-sub">c\'è una finestra surfabile nei prossimi giorni'
                         f' — giorno e orario riservati agli abbonati</div>')
        else:
            best_html = ('<div class="best-none">Mare piatto ora</div>'
                         '<div class="best-sub">controlla di nuovo: gli abbonati vedono 5 giorni</div>')
    elif best is not None:
        bg, fg, bd = TINT[best.stato]
        et = STATI[best.stato][0]
        best_html = (
            f'<div class="best-day">{gg(best.date)} {best.date:%d/%m}</div>'
            f'<div class="best-sub">ore {best.date:%H}:00 · {best.hs_alisee:.1f} m · '
            f'{best.tp_alisee:.0f}s da {onda_cardinale(best.wave_direction)} · '
            f'vento {best.vento_kn:.0f} kn</div>'
            f'<div class="best-badge" style="background:{bg};color:{fg};border-color:{bd}">'
            f'<span class="vb-t">{et}</span></div>')
    else:
        # Niente di buono nelle 72h: mai un vicolo cieco, guarda la tendenza.
        risalita = next((f"possibile risalita {gg(d)} (~{h:.1f} m)"
                         for d, h in tnd if h >= 0.8), "ancora piatto nei prossimi giorni")
        best_html = ('<div class="best-none">Niente di surfabile<br>nelle prossime 72 ore</div>'
                     f'<div class="best-sub">{risalita} · torna a controllare</div>')

    def _day_card(g):
        bg, fg, bd = TINT[g["stato"]]
        et = STATI[g["stato"]][0]
        win = (f'finestra {g["win"]}' if g["win"] != "—" else 'niente onda')
        return (f'<div class="card day" data-dm="{g["data"]:%d/%m}" '
                f'style="border-left-color:{fg}">'
                f'<div class="d">{gg(g["data"])} {g["data"]:%d/%m}</div>'
                f'<div class="h">{g["hs"]:.1f}<small style="font-size:14px;color:#8b949e"> m</small>'
                f'<span class="db" style="background:{bg};color:{fg};border-color:{bd}">{et}</span></div>'
                f'<div class="sparkwrap" style="color:{fg}">{_sparkline(g["serie"])}</div>'
                f'<div class="w">{win} · {g["dir"]} · vento {g["vento"]:.0f} kn {g["vdir"]}</div></div>')
    giorni = "".join(_day_card(g) for g in _giorni(df72))

    # Tendenza 4-5 giorni: il motivo per tornare domani (e' il dato che evolve)
    tnd_html = ""
    if tnd:
        tnd_txt = " · ".join(f"{gg(d)} {d:%d/%m} ~{h:.1f} m" for d, h in tnd)
        tnd_html = (f'<div class="trend"><b>Tendenza:</b> {tnd_txt} '
                    f'<span>· a 4-5 giorni l\'affidabilità cala, ricontrolla domani</span></div>')

    # Gancio al prodotto premium della piattaforma: il momento buono -> la cam.
    # Sta SOTTO il grafico: cosi' mentre scrubbi vedi il vento nella card in alto,
    # e la CTA arriva dopo che hai visto le condizioni (momento giusto per cliccare).
    cta_cam = (f'<div class="cta-wrap"><a class="cta" data-ev="cam" href="{_utm(CAM_URL)}">'
               f'Guarda la cam live →</a></div>' if CAM_URL else "")
    spons = (f' <span class="spons">· previsione offerta da <b>{SPONSOR}</b></span>'
             if SPONSOR else "")

    # Upsell tier superiore: previsioni a lungo termine + alert personalizzati.
    # E' la card "a pagamento" oltre il forecast base — nuova linea di ricavo.
    upsell_html = ""
    if UPSELL_URL and not gate:
        upsell_html = (
            '<div class="upsell"><div class="up-l">'
            '<div class="up-t">Vai oltre le 72 ore</div>'
            '<div class="up-s">Previsioni fino a 14 giorni · <b>alert sul telefono</b> '
            'quando le condizioni del tuo spot sono perfette · storico delle mareggiate</div>'
            f'</div><a class="up-b" data-ev="upsell" href="{_utm(UPSELL_URL)}">Attiva Premium+ →</a></div>')

    # Freemium: le 24h si vedono, il resto e' il prodotto che la piattaforma vende
    if gate:
        dfc = df72[df72.date < t0 + pd.Timedelta("24h")]
        titolo_chart = "onda e vento · prossime 24 ore"
        sblocca = (f'<a class="cta" data-ev="premium" href="{_utm(PREMIUM_URL)}">Sblocca con Premium →</a>'
                   if PREMIUM_URL else "")
        centro = (f'<div class="lock"><div class="lock-t">Previsione completa a 5 giorni</div>'
                  f'<div class="lock-s">ora per ora · finestre surfabili con giorno e orario · '
                  f'fascia probabile misurata alla boa</div>{sblocca}</div>')
        cta_cam = ""                    # nel free l'unico bottone e' lo sblocco
        dayps_html = ""
    else:
        dfc = df72
        titolo_chart = "onda e vento · prossime 72 ore"
        centro = f'<div class="days">{giorni}</div>\n{tnd_html}'
        dayps_html = '<div class="dayps" id="dayps"></div>'

    js = (JS_CHART.replace("__DATA__", json.dumps(_dati_json(dfc), ensure_ascii=False))
                  .replace("__CRON__", json.dumps(CRON_UTC)))
    banda_meta = ("" if gate else
                  f'<div class="meta2">probabile <span id="ro-band">'
                  f'{f"{now.hs_p10:.1f}".replace(".", ",")}–'
                  f'{f"{now.hs_p90:.1f}".replace(".", ",")} m</span></div>')
    ultima, prossima = orari_run()
    pross_txt = (f" · prossima {gg(prossima)} {prossima:%H:%M}" if prossima else "")

    # Badge verdetto "adesso": e' l'elemento che deve saltare all'occhio (si
    # surfa o no?), non piu' la pillina sbiadita.
    nbg, nfg, nbd = TINT[now.stato]
    vbadge = (f'<div class="vbadge" id="ro-badge" style="background:{nbg};color:{nfg};'
              f'border-color:{nbd}"><div class="vb-t">{lbl}</div>'
              f'<div class="vb-s">qualità surf</div></div>')

    # Gerarchia del titolo: lo SPOT e' cio' che il lettore cerca (dove?), il
    # marchio e' la firma sotto. Prima erano sulla stessa riga separati da
    # simboli diversi: affollato e con l'ordine sbagliato.
    logo = ('<svg class="logo" viewBox="0 0 32 32" width="30" height="30">'
            '<rect width="32" height="32" rx="9" fill="#0d2137"/>'
            '<path d="M4 20c4-6 7 4 11-2s6 3 12-3" stroke="#58a6ff" stroke-width="2.6"'
            ' fill="none" stroke-linecap="round"/></svg>')
    firma_brand = (f'<b>ALISEE</b> <span class="x">×</span> {PARTNER}' if PARTNER
                   else '<b>ALISEE</b> weather intelligence')
    titolo = (f'<div class="brandrow">{logo}<div>'
              f'<h1>{SPOT}<span class="dot" title="aggiornata"></span></h1>'
              f'<div class="tsub">previsione onda e vento · {firma_brand}</div>'
              f'</div></div>')
    sotto_embed = (f'previsione onda e vento · 72h · aggiornata {gg(ultima)} {ultima:%H:%M}')
    firma = (f'<div class="upd">{sotto_embed}</div>' if embed else
             f'<div class="upd">ultima run <b style="color:#8b949e">{gg(ultima)} '
             f'{ultima:%d/%m %H:%M}</b> (ora italiana){pross_txt} · boa di Civitavecchia'
             f' · previsione 72h</div>')

    # Precisione: blocco completo sulla dashboard; nel widget del cliente una riga
    # sola (il surfista vuole l'onda, la prova estesa serve alla trattativa).
    if embed:
        vals = " · ".join(
            f"{c} {f'{a:.{d}f}'.replace('.', ',')} {u} "
            f"(standard: {f'{s:.{d}f}'.replace('.', ',')})"
            for c, a, s, u, d, m in SKILL)
        acc_html = (f'<div class="acc"><div class="accs" style="margin:0">'
                    f'Verificata sul mare vero — errore medio: {vals}.</div></div>')
    else:
        scala_html = "".join(
            f'<span class="acs"><i>{c}</i>'
            f'<b class="no">{f"{std:.2f}".replace(".", ",")}</b> standard'
            f'<b class="si">{f"{ali:.2f}".replace(".", ",")}</b> ALISEE</span>'
            for c, std, ali in SCALA)
        acc_html = f"""<div class="acc">
  <div class="acch">Verificata sul mare vero</div>
  <div class="accs">La barra mostra l'errore medio misurato: più è corta, meno si sbaglia.</div>
  <div class="acgrid">{_accuratezza()}</div>
  <div class="acex">Un esempio concreto: quando la previsione dice <b>1,5 m</b>, 8 volte su 10
    il mare è poi tra <b>1,1 e 1,8 m</b> — è la "fascia probabile" che vedi nel grafico.</div>
  <div class="acscale">
    <div class="acsh">Perché quando diciamo 2 m sono 2 m</div>
    Sulle stesse ore abbiamo misurato anche la <b>scala</b>: si confronta ogni previsione
    con la misura reale e si guarda se le proporzioni tengono (1,00 = scala giusta).
    I modelli pubblici <b>comprimono</b>: più il mare cresce, più lo sottostimano.
    <div class="acsg">{scala_html}</div>
    Per questo altrove le mareggiate arrivano sempre più grandi del previsto. Qui no.</div>
</div>"""

    # Anteprima quando il link viene condiviso (WhatsApp, IG, Telegram): senza
    # questi tag esce un URL nudo. Il riassunto e' il dato del momento.
    og_desc = (f"Onda {f'{now.hs_alisee:.1f}'.replace('.', ',')} m · "
               f"{now.tp_alisee:.0f}s da {onda_cardinale(now.wave_direction)} · "
               f"vento {now.vento_kn:.0f} kn — previsione 72h calibrata su strumenti reali")
    # Favicon inline (onda ALISEE): nessun file da servire, nessuna richiesta in piu'
    favicon = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'"
               "%3E%3Crect width='32' height='32' rx='7' fill='%230d1117'/%3E%3Cpath d='M3 20c4-6 7 4 11-2s7 3 11-3'"
               " stroke='%2358a6ff' stroke-width='3' fill='none' stroke-linecap='round'/%3E%3C/svg%3E")

    doc = f"""<!doctype html><html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="1800">
<title>ALISEE Onda · {SPOT}</title>
<link rel="icon" href="{favicon}">
<meta property="og:title" content="Previsione onda e vento · {SPOT}">
<meta property="og:description" content="{og_desc}">
<meta property="og:type" content="website">
<meta name="theme-color" content="#0d1117">
<style>{CSS}{CSS_EMBED if embed else ""}</style></head><body>
{"" if embed else SFONDO}
<div class="wrap">
<div class="top">
  {titolo}
  {firma}
</div>

<div class="hero">
  <div class="card">
    <div class="k">adesso <span class="when2" id="ro-when"></span></div>
    <div class="now">
      {_bussola(now.wave_direction, 66, "ro-arrow")}
      <div>
        <div class="hs"><span id="ro-hs">{f"{now.hs_alisee:.1f}".replace(".", ",")}</span> <small>m</small>
          <span class="trend" id="ro-trend"></span></div>
        <div class="meta"><span id="ro-tp">{now.tp_alisee:.0f}s</span> · da <span id="ro-wdc">{onda_cardinale(now.wave_direction)}</span></div>
        {banda_meta}
      </div>
      {vbadge}
    </div>
    <div class="mini">
      <div><div class="v" id="ro-k" style="color:{v_col}">{now.vento_kn:.0f} kn</div>
        <div class="l" id="ro-kl">{onda_cardinale(now.vento_dir)} · {v_lbl}</div></div>
      <div><div class="v" id="ro-w">{sst_txt}</div><div class="l">acqua</div></div>
      <div><div class="v" id="ro-sp">{swp:.0f}%</div><div class="l">mare lungo</div>
        <div class="bar"><i id="ro-spb" style="width:{swp:.0f}%"></i></div></div>
      <div><div class="v">{pk.hs_alisee:.1f} m</div><div class="l">{pk_l}</div></div>
    </div>
  </div>
</div>

<div class="chart">
  <div class="ct"><span>{titolo_chart}</span>
    <span class="sw"><button class="swb" data-metrica="vento">vento</button><button
      class="swb" data-metrica="periodo">periodo</button> · <span id="cnt"></span></span></div>
  {dayps_html}
  <svg id="ch" viewBox="0 0 940 306" width="100%" style="touch-action:none;cursor:crosshair;display:block"></svg>
  <div class="legend">
    <span class="lg"><svg width="22" height="8"><line x1="0" y1="4" x2="22" y2="4" stroke="#58a6ff" stroke-width="2"/></svg>ALISEE</span>
    <span class="lg"><svg width="22" height="8"><line x1="0" y1="4" x2="22" y2="4" stroke="#8b949e" stroke-width="1.5" stroke-dasharray="4 3"/></svg>modello standard</span>
    <span class="lg"><i style="background:#58a6ff;opacity:.3"></i>fascia probabile</span>
    <span class="lg"><i style="background:#010409;border:1px solid #30363d"></i>notte</span>
  </div>
  <div class="legend legend2"><span class="lgt">la fascia "qualità surf" nel grafico:</span>
    {_legenda()}
    <span class="hint">tocca o trascina sul grafico: i numeri in alto seguono l'ora</span></div>
  {cta_cam}
</div>

<div class="card best">
  <div class="k">{best_k}</div>
  {best_html}
  <div class="best-foot">Come leggiamo il mare: <b>flat</b> sotto 0,5 m · <b>sotto misura</b>
    fino a 0,8 m · <b>surfabile</b> oltre 0,8 m · <b>choppy</b> quando vento onshore teso o
    periodo corto scompongono l'onda · <b>clean</b> con onda formata (1,2–2,8 m, periodo ≥6s),
    swell da W/SW e vento a favore. Finestre calcolate solo nelle ore di luce.
    "Probabile" = quanto può variare: 8 volte su 10 il mare sta in quell'intervallo.</div>
</div>

{centro}

{acc_html}

{upsell_html}

<div class="foot">
  <b>Come funziona.</b> Partiamo dai modelli meteo-marini pubblici e li correggiamo con un
  motore di intelligenza artificiale addestrato su anni di misure reali della <b>boa</b> e
  della <b>stazione di Civitavecchia</b>, a pochi km dallo spot: impara gli errori
  sistematici del modello su questo tratto di costa e li compensa. Le percentuali di
  precisione qui sopra sono verificate su ~{SKILL_ORE} ore di dati mai usati in addestramento.
</div>
<div class="brand"><b>ALISEE</b> · weather intelligence — previsioni calibrate su strumenti reali{spons}</div>
</div><script>{js}</script>{_analytics()}</body></html>"""
    nome = ("widget-free.html" if gate else
            "widget.html" if embed else "dashboard.html")
    with open(os.path.join(BASE, nome), "w", encoding="utf-8") as f:
        f.write(doc)


if __name__ == "__main__":
    df = scarica_e_prevedi()          # 5 giorni: 72h piene + tendenza
    df72 = df[df.date < df.date.iloc[0] + pd.Timedelta("72h")]
    wins = finestre(df72)             # le finestre si promettono solo sulle 72h

    print("=" * 60)
    print(f"  ALISEE ONDA — {SPOT} (boa RON Civitavecchia)  |  72h")
    print("=" * 60)
    print(f"{'quando':16s} {'Hs':>7s} {'Tp':>6s} {'dir':>5s}  stato")
    for _, r in df72.iloc[::2].iterrows():
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
    archivia_previsioni(df)
    scrivi_snapshot(df, wins, momento_migliore(df72)) # snapshot.json — sorgente MCP
    build_dashboard(df, wins)                        # dashboard.html — completa
    build_dashboard(df, wins, embed=True)            # widget.html — per abbonati
    build_dashboard(df, wins, embed=True, gate=True) # widget-free.html — esca freemium
    print("\nCSV + snapshot.json + dashboard + widget + widget-free aggiornati.")
