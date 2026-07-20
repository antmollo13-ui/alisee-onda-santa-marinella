"""
ALISEE — previsione onda + vento 72h per Santa Marinella. Il prodotto.
Carica modello_onda.pkl e modello_vento.pkl, scarica i forecast (boa RON per l'onda,
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


def _utm(url):
    """Aggiunge il tracciamento: ogni click dal widget e' attribuibile ad ALISEE.
    E' la prova, nei LORO analytics, di quanto il widget converte."""
    if not url:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}utm_source=alisee&utm_medium=widget"

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


def _bussola(direzione, size=64, arrow_id=""):
    """Rosa con freccia nel verso di propagazione (l'onda VIENE da `direzione`).
    Con arrow_id la freccia diventa pilotabile dal JS (ruota durante lo scrub)."""
    d = float(direzione or 0)
    aid = f' id="{arrow_id}"' if arrow_id else ""
    return f"""<svg viewBox="0 0 64 64" width="{size}" height="{size}">
  <circle cx="32" cy="32" r="27" fill="none" stroke="#30363d" stroke-width="1"/>
  <text x="32" y="10" text-anchor="middle" fill="#6e7681" font-size="8">N</text>
  <text x="32" y="60" text-anchor="middle" fill="#6e7681" font-size="8">S</text>
  <text x="58" y="35" text-anchor="middle" fill="#6e7681" font-size="8">E</text>
  <text x="6"  y="35" text-anchor="middle" fill="#6e7681" font-size="8">O</text>
  <g{aid} transform="rotate({d + 180:.0f} 32 32)">
    <path d="M32 14 L38 42 L32 36 L26 42 Z" fill="#58a6ff"/>
  </g></svg>"""


def _dati_json(df):
    """Serializza le ore per il grafico interattivo (colori gia' risolti qui,
    cosi' il JS non deve conoscere le regole dello spot)."""
    out = []
    for _, r in df.iterrows():
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
        })
    return out


# Motore del grafico interattivo: curve morbide, fascia probabile, notte, vento,
# crosshair che aggiorna il pannello "adesso" durante lo scrub, pillole giorno,
# countdown del prossimo aggiornamento. Vanilla JS, zero dipendenze.
JS_CHART = r"""
(function(){
const D=__DATA__, CRON=__CRON__;
const svg=document.getElementById('ch'); if(!svg||!D.length) return;
const W=940, PL=46, PR=14, PT=24, HW=170, GAP=42, HK=44, PB=26;
const H=PT+HW+GAP+HK+PB; svg.setAttribute('viewBox','0 0 '+W+' '+H);
const n=D.length, bw=(W-PL-PR)/n;
const hsMax=Math.max(1, Math.max.apply(null,D.map(d=>d.b))*1.08);
const kMax=Math.max(12, Math.max.apply(null,D.map(d=>d.k))*1.15);
const X=i=>PL+(i+0.5)*bw, YH=v=>PT+HW*(1-v/hsMax);
const yk0=PT+HW+GAP+HK, YK=v=>yk0-HK*Math.min(v,kMax)/kMax;
const E=(t,a)=>{const e=document.createElementNS('http://www.w3.org/2000/svg',t);
  for(const k in a)e.setAttribute(k,a[k]);svg.appendChild(e);return e;};
let i0=0;while(i0<n){if(!D[i0].l){let j=i0;while(j<n&&!D[j].l)j++;
  E('rect',{x:PL+i0*bw,y:PT,width:(j-i0)*bw,height:yk0-PT,fill:'#010409',opacity:.5});i0=j;}else i0++;}
const step=hsMax<=2.5?0.5:1;
for(let v=0;v<=hsMax;v+=step){E('line',{x1:PL,y1:YH(v),x2:W-PR,y2:YH(v),stroke:'#ffffff14'});
  const t=E('text',{x:PL-6,y:YH(v)+3,'text-anchor':'end',fill:'#6e7681','font-size':11});t.textContent=v.toFixed(1);}
if(kMax>10){E('line',{x1:PL,y1:YK(10),x2:W-PR,y2:YK(10),stroke:'#ffffff10'});
  const t=E('text',{x:PL-6,y:YK(10)+3,'text-anchor':'end',fill:'#6e7681','font-size':10});t.textContent='10';}
const days=[];D.forEach((d,i)=>{if(!days.length||days[days.length-1].dm!==d.dm)days.push({dm:d.dm,gg:d.gg,i:i});});
days.forEach((d,k)=>{if(k>0)E('line',{x1:PL+d.i*bw,y1:PT,x2:PL+d.i*bw,y2:yk0,stroke:'#30363d','stroke-dasharray':'3 3'});
  const t=E('text',{x:PL+d.i*bw+4,y:yk0+16,fill:'#8b949e','font-size':11});t.textContent=d.gg+' '+d.dm;});
const defs=document.createElementNS(svg.namespaceURI,'defs');
defs.innerHTML='<linearGradient id="ga" x1="0" y1="0" x2="0" y2="1">'
 +'<stop offset="0" stop-color="#58a6ff" stop-opacity="0.35"/>'
 +'<stop offset="1" stop-color="#58a6ff" stop-opacity="0.02"/></linearGradient>';
svg.appendChild(defs);
function spline(p){if(p.length<2)return'';let s='M'+p[0][0].toFixed(1)+' '+p[0][1].toFixed(1);
 for(let i=0;i<p.length-1;i++){const a=p[Math.max(0,i-1)],b=p[i],c=p[i+1],d=p[Math.min(p.length-1,i+2)];
  s+='C'+(b[0]+(c[0]-a[0])/6).toFixed(1)+' '+(b[1]+(c[1]-a[1])/6).toFixed(1)+' '
   +(c[0]-(d[0]-b[0])/6).toFixed(1)+' '+(c[1]-(d[1]-b[1])/6).toFixed(1)+' '
   +c[0].toFixed(1)+' '+c[1].toFixed(1);} return s;}
const pH=D.map((d,i)=>[X(i),YH(d.h)]), pA=D.map((d,i)=>[X(i),YH(d.a)]);
const pB=D.map((d,i)=>[X(i),YH(d.b)]), pN=D.map((d,i)=>[X(i),YH(Math.min(d.nw,hsMax))]);
let bd=spline(pB);
pA.slice().reverse().forEach(p=>{bd+='L'+p[0].toFixed(1)+' '+p[1].toFixed(1);});
E('path',{d:bd+'Z',fill:'#58a6ff',opacity:.13});
E('path',{d:spline(pH)+'L'+X(n-1).toFixed(1)+' '+YH(0).toFixed(1)+'L'+X(0).toFixed(1)+' '+YH(0).toFixed(1)+'Z',fill:'url(#ga)'});
E('path',{d:spline(pN),fill:'none',stroke:'#8b949e','stroke-width':1.2,'stroke-dasharray':'4 3',opacity:.6});
E('path',{d:spline(pH),fill:'none',stroke:'#58a6ff','stroke-width':2});
D.forEach((d,i)=>{E('rect',{x:PL+i*bw,y:PT+HW+10,width:bw+0.5,height:6,fill:d.sc,opacity:d.l?1:.45});});
D.forEach((d,i)=>{E('rect',{x:PL+i*bw+bw*0.15,y:YK(d.k),width:bw*0.7,height:yk0-YK(d.k),rx:1,
  fill:d.wc,'fill-opacity':d.l?1:.5});});
let c1=E('text',{x:PL,y:PT-9,fill:'#8b949e','font-size':11});c1.textContent='onda (m)';
let c2=E('text',{x:PL,y:yk0-HK-8,fill:'#8b949e','font-size':11});c2.textContent='vento (kn)';
const cl=E('line',{y1:PT,y2:yk0,stroke:'#e6edf3','stroke-width':1,opacity:0,'stroke-dasharray':'2 3'});
const cd=E('circle',{r:4.5,fill:'#58a6ff',stroke:'#0d1117','stroke-width':2,opacity:0});
const fmt=v=>v.toFixed(1).replace('.',',');
const $=id=>document.getElementById(id);
function setRO(i,active){const d=D[i];
 if($('ro-when'))$('ro-when').textContent=active?(d.gg+' '+d.dm+' · '+d.hh):'adesso';
 if($('ro-hs'))$('ro-hs').textContent=fmt(d.h);
 if($('ro-band'))$('ro-band').textContent=fmt(d.a)+'–'+fmt(d.b)+' m';
 if($('ro-tp'))$('ro-tp').textContent=Math.round(d.p)+'s';
 if($('ro-wdc'))$('ro-wdc').textContent=d.wdc;
 const p=$('ro-pill');if(p){p.textContent=d.s;p.style.background=d.sc;}
 const kv=$('ro-k');if(kv){kv.textContent=Math.round(d.k)+' kn';kv.style.color=d.wc;}
 if($('ro-kl'))$('ro-kl').textContent=d.kdc+' · '+d.kl;
 if($('ro-w'))$('ro-w').textContent=(d.w==null?'—':d.w+'°');
 if($('ro-sp')){$('ro-sp').textContent=d.sp+'%';const b=$('ro-spb');if(b)b.style.width=d.sp+'%';}
 const ar=$('ro-arrow');if(ar)ar.setAttribute('transform','rotate('+(d.wd+180)+' 32 32)');
 cl.setAttribute('x1',X(i));cl.setAttribute('x2',X(i));cl.setAttribute('opacity',active?0.5:0);
 cd.setAttribute('cx',X(i));cd.setAttribute('cy',YH(d.h));cd.setAttribute('opacity',active?1:0);}
function idx(e){const r=svg.getBoundingClientRect();
 const px=(e.clientX-r.left)*W/r.width;
 return Math.max(0,Math.min(n-1,Math.round((px-PL)/bw-0.5)));}
svg.addEventListener('pointermove',e=>setRO(idx(e),true));
svg.addEventListener('pointerdown',e=>setRO(idx(e),true));
svg.addEventListener('pointerleave',()=>setRO(0,false));
const dp=$('dayps');
if(dp)days.forEach(d=>{const b=document.createElement('button');b.className='dayp';
 b.textContent=d.gg+' '+d.dm;
 b.onclick=()=>setRO(Math.min(n-1,d.i+12),true);dp.appendChild(b);});
function cnt(){const now=new Date();let best=null;
 for(let g=0;g<2;g++)CRON.forEach(h=>{const t=new Date(Date.UTC(now.getUTCFullYear(),
  now.getUTCMonth(),now.getUTCDate()+g,h,0,0));if(t>now&&(!best||t<best))best=t;});
 if(!best)return;const m=Math.round((best-now)/60000);const el=$('cnt');
 if(el)el.textContent='si aggiorna tra '+(m>=60?Math.floor(m/60)+'h '+(m%60)+'m':m+' min');}
cnt();setInterval(cnt,30000);
setRO(0,false);
})();
"""


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
.trend{background:#161b22;border:1px solid #21262d;border-radius:12px;padding:10px 18px;
       margin-bottom:14px;font-size:13px}
.trend b{color:#e6edf3;font-weight:600}
.trend span{font-size:11px;color:#6e7681}
.cta{display:inline-block;margin-top:12px;background:#238636;color:#fff;font-size:12px;
     font-weight:600;padding:7px 14px;border-radius:8px;text-decoration:none}
.cta:hover{background:#2ea043}
.spons{color:#6e7681}
.spons b{color:#e6edf3}
.lock{background:#161b22;border:1px dashed #30363d;border-radius:12px;padding:22px 18px;
      text-align:center;margin-bottom:14px}
.lock-t{font-size:15px;font-weight:600}
.lock-s{font-size:12px;color:#8b949e;margin-top:5px;margin-bottom:4px}
.meta2{font-size:12px;color:#6e7681;margin-top:3px}
.meta2 span{color:#8b949e;font-weight:600}
.dayps{display:flex;gap:6px;margin:0 2px 8px;flex-wrap:wrap}
.dayp{background:#0d1117;border:1px solid #21262d;color:#8b949e;font-size:11px;
      padding:3px 11px;border-radius:14px;cursor:pointer;font-family:inherit}
.dayp:hover{color:#e6edf3;border-color:#8b949e}
.hint{color:#6e7681;font-style:italic;margin-left:auto}
.acc{background:#161b22;border:1px solid #21262d;border-radius:12px;padding:14px 18px;margin-bottom:14px}
.acch{font-size:13px;font-weight:600;margin-bottom:3px}
.accs{font-size:11px;color:#6e7681;margin-bottom:12px}
.acgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}
.acn{font-size:11px;color:#8b949e;margin-bottom:6px;text-transform:uppercase;letter-spacing:.04em}
.tag{display:inline-block;background:#1b2c20;color:#3fb950;font-size:10px;font-weight:600;
     padding:1px 7px;border-radius:10px;margin-left:6px;text-transform:none;letter-spacing:0}
.acr{display:flex;align-items:center;gap:7px;margin-bottom:4px;font-size:11px}
.acl{color:#6e7681;width:48px;flex:none}
.acr i{display:block;height:7px;border-radius:3px;flex:none}
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

    if gate and wins:
        # L'esca: l'utente gratuito sa che la finestra ESISTE e quanto e' grande,
        # ma giorno e orario sono il prodotto premium della piattaforma.
        hs = wins[0][2]
        win_html = (f'<div class="big">{hs:.1f} m in arrivo</div>'
                    f'<div class="sub">c\'è una finestra surfabile nei prossimi 3 giorni'
                    f' — giorno e orario riservati agli abbonati</div>')
    elif wins:
        a, b, hs, tp, dr, w10, w90 = wins[0]
        win_html = (f'<div class="big">{hs:.1f} m</div>'
                    f'<div class="sub">{gg(a)} {a:%d} · {a:%H:%M}–{b:%H:%M} · {tp:.0f}s {dr}'
                    f'<br>probabile tra {w10:.1f} e {w90:.1f} m</div>')
    else:
        # Giorno piatto: MAI un vicolo cieco — dai sempre un motivo per tornare.
        risalita = next((f"possibile onda {gg(d)} (~{h:.1f} m): torna a controllare"
                         for d, h in tnd if h >= 0.8),
                        "piatto anche nella tendenza — ricontrolla domani")
        win_html = ('<div class="big" style="color:#6e7681">—</div>'
                    f'<div class="sub">niente onda nelle 72h · {risalita}</div>')

    giorni = "".join(
        f'<div class="card day"><div class="d">{gg(g["data"])} {g["data"]:%d/%m}</div>'
        f'<div class="h">{g["hs"]:.1f} m <span class="pill" style="background:'
        f'{STATI[g["stato"]][1]};font-size:10px">{STATI[g["stato"]][0]}</span></div>'
        f'<div class="w">picco {g["tp"]:.0f}s {g["dir"]} · vento {g["vento"]:.0f}kn '
        f'{g["vdir"]} · finestra {g["win"]}</div></div>'
        for g in _giorni(df72))

    # Tendenza 4-5 giorni: il motivo per tornare domani (e' il dato che evolve)
    tnd_html = ""
    if tnd:
        tnd_txt = " · ".join(f"{gg(d)} {d:%d/%m} ~{h:.1f} m" for d, h in tnd)
        tnd_html = (f'<div class="trend"><b>Tendenza:</b> {tnd_txt} '
                    f'<span>· a 4-5 giorni l\'affidabilità cala, ricontrolla domani</span></div>')

    # Gancio al prodotto premium della piattaforma: il momento buono -> la cam
    cta = (f'<a class="cta" href="{_utm(CAM_URL)}">Guarda la cam live →</a>' if CAM_URL else "")
    spons = (f' <span class="spons">· previsione offerta da <b>{SPONSOR}</b></span>'
             if SPONSOR else "")

    # Freemium: le 24h si vedono, il resto e' il prodotto che la piattaforma vende
    if gate:
        dfc = df72[df72.date < t0 + pd.Timedelta("24h")]
        titolo_chart = "onda e vento · prossime 24 ore"
        sblocca = (f'<a class="cta" href="{_utm(PREMIUM_URL)}">Sblocca con Premium →</a>'
                   if PREMIUM_URL else "")
        centro = (f'<div class="lock"><div class="lock-t">Previsione completa a 5 giorni</div>'
                  f'<div class="lock-s">ora per ora · finestre surfabili con giorno e orario · '
                  f'fascia probabile misurata alla boa</div>{sblocca}</div>')
        cta = ""                        # nel free l'unico bottone e' lo sblocco
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
                    f'Verificata sul mare vero — errore medio: {vals}.</div></div>')
    else:
        acc_html = f"""<div class="acc">
  <div class="acch">Verificata sul mare vero</div>
  <div class="accs">La barra mostra l'errore medio misurato: più è corta, meno si sbaglia.</div>
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
    <div class="k" id="ro-when">adesso</div>
    <div class="now">
      {_bussola(now.wave_direction, 72, "ro-arrow")}
      <div>
        <div class="hs"><span id="ro-hs">{f"{now.hs_alisee:.1f}".replace(".", ",")}</span> <span style="font-size:18px;color:#8b949e">m</span></div>
        <div class="meta"><span id="ro-tp">{now.tp_alisee:.0f}s</span> · da <span id="ro-wdc">{onda_cardinale(now.wave_direction)}</span>
          &nbsp;<span class="pill" id="ro-pill" style="background:{col}">{lbl}</span></div>
        {banda_meta}
      </div>
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
  <div class="card">
    <div class="k">prossima finestra surfabile</div>
    {win_html}
    {cta}
    <div class="mini"><div><div class="l" style="line-height:1.6">"Buono" = onda formata,
      swell da W/SW e vento amico. Finestre solo nelle ore di luce. "Probabile" = misurato
      alla boa: 8 volte su 10 il mare sta lì.</div></div></div>
  </div>
</div>

<div class="chart">
  <div class="ct"><span>{titolo_chart}</span>
    <span>— — modello standard · <span id="cnt"></span></span></div>
  {dayps_html}
  <svg id="ch" viewBox="0 0 940 306" width="100%" style="touch-action:none;cursor:crosshair;display:block"></svg>
  <div class="legend">{_legenda()}
    <span class="lg"><i style="background:#58a6ff;opacity:.3"></i>fascia probabile</span>
    <span class="lg"><i style="background:#010409;border:1px solid #30363d"></i>notte</span>
    <span class="hint">tocca o passa il mouse sul grafico: i numeri qui sopra seguono l'ora</span></div>
</div>

{centro}

{acc_html}

<div class="foot">
  Misurato su ~{SKILL_ORE} ore di confronto con la <b>boa ondametrica RON</b> (onda) e la
  <b>stazione RMN di Civitavecchia</b> (vento), entrambe a ~8 km dallo spot, su periodi che il
  modello non aveva mai visto in addestramento. "Standard" = il modello meteo pubblico da cui
  partono i siti di previsione. I valori sono l'errore sull'analisi: su una previsione a 72 ore
  di anticipo entrambi sbagliano di più.
</div>
<div class="brand"><b>ALISEE</b> · weather intelligence — previsioni calibrate su strumenti reali{spons}</div>
</div><script>{js}</script></body></html>"""
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
    build_dashboard(df, wins)                        # dashboard.html — completa
    build_dashboard(df, wins, embed=True)            # widget.html — per abbonati
    build_dashboard(df, wins, embed=True, gate=True) # widget-free.html — esca freemium
    print("\nprevisione_onda_72h.csv + dashboard + widget + widget-free aggiornati.")
