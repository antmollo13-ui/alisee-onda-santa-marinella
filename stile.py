"""
ALISEE — DESIGN della dashboard (palette, CSS, sfondo, grafico SVG).

Qui vive TUTTO l'aspetto: chi ridisegna tocca solo questo file, il motore di
previsione (previsione_onda.py) resta intatto.

VINCOLI da rispettare in un redesign:
 1. La pagina vive anche DENTRO un iframe sul sito di un cliente: in modalita'
    embed lo sfondo resta trasparente (CSS_EMBED) e niente position:fixed.
 2. Un solo codice genera TRE pagine: dashboard, widget abbonati, widget gratuito.
 3. Zero dipendenze esterne: niente CDN, font o librerie (velocita' + privacy
    dentro la pagina altrui).
 4. I colori degli stati hanno SIGNIFICATO (verde=clean ... grigio=niente da
    surfare): si cambiano le tinte, non l'associazione.
 5. Il grafico non e' CSS: e' SVG costruito in JS_CHART.
 6. Dark obbligatorio, mobile-first (il pubblico guarda dal telefono all'alba).
 7. backdrop-filter sulle card e' bello ma pesante: usarlo con misura.
"""
import numpy as np


STATI = {  # stato -> (etichetta, colore)
    # Gergo tecnico del surf: termini INVARIABILI (niente concordanza maschile/
    # femminile con "onda"/"qualita'"/"mare") e riconoscibili da chi surfa.
    # flat e sotto misura condividono il grigio: per chi guarda sono la stessa
    # notizia (non si va), due grigi diversi erano solo rumore.
    "piatto":      ("flat",          "#484f58"),
    "piccolo":     ("sotto misura",  "#484f58"),
    "mosso/corto": ("choppy",        "#d29922"),
    "surfabile":   ("surfabile",     "#58a6ff"),
    "BUONO":       ("clean",         "#3fb950"),
}
# Legenda del grafico: le 4 classi che si VEDONO (il grigio copre due stati)
_LEGENDA = [("clean", "#3fb950"), ("surfabile", "#58a6ff"),
            ("choppy", "#d29922"), ("flat / sotto misura", "#484f58")]

# Tint per i BADGE di stato (sfondo tenue + testo colorato + bordo): rende il
# verdetto l'elemento piu' visibile, invece della pillina sbiadita.
TINT = {
    "piatto":      ("#1c2128", "#8b949e", "#30363d"),
    "piccolo":     ("#1c2128", "#8b949e", "#30363d"),
    "mosso/corto": ("#2a2109", "#e0a92e", "#5c4a12"),
    "surfabile":   ("#0d2137", "#58a6ff", "#1c436e"),
    "BUONO":       ("#10241a", "#3fb950", "#1f5133"),
}


CSS = """
*{box-sizing:border-box;margin:0;padding:0}
/* CONTENIMENTO: basta un elemento piu' largo del viewport (una riga che non va
   a capo, una fila di pillole) e il telefono rimpicciolisce TUTTA la pagina.
   Qui si blocca la propagazione alla radice. */
html,body{max-width:100%;overflow-x:hidden}
body{background:#0d1117;color:#e6edf3;font-family:-apple-system,Segoe UI,Roboto,Helvetica,sans-serif;
     padding:24px;line-height:1.5;-webkit-font-smoothing:antialiased;
     overflow-wrap:break-word}
/* Sfondo: profondita' marina + onde in SVG (nessuna immagine esterna, peso zero).
   Volutamente TENUE: e' uno strumento di dati, lo sfondo non deve competere. */
.bg{position:fixed;inset:0;z-index:-1;overflow:hidden;pointer-events:none;
    background:radial-gradient(900px 500px at 15% -12%, #1b3d5c 0%, transparent 62%),
               radial-gradient(700px 420px at 95% 5%, #143452 0%, transparent 60%),
               linear-gradient(180deg,#0d1117 0%,#0b1a29 55%,#07131d 100%)}
.bg svg{position:absolute;left:0;bottom:0;width:200%;height:min(52vh,420px)}
.bg .w1{animation:drift 46s linear infinite}
.bg .w2{animation:drift 78s linear infinite reverse}
@keyframes drift{from{transform:translateX(0)}to{transform:translateX(-50%)}}
.wrap{width:100%;max-width:1000px;margin:0 auto;position:relative;min-width:0}
.top{display:flex;justify-content:space-between;align-items:center;margin-bottom:18px;flex-wrap:wrap;gap:8px}
.brandrow{display:flex;align-items:center;gap:12px}
.logo{flex:none;border-radius:9px}
h1{font-size:23px;font-weight:600;letter-spacing:-.02em;line-height:1.15}
.tsub{font-size:12px;color:#6e7681;margin-top:2px;letter-spacing:.01em}
.tsub b{color:#58a6ff;font-weight:600;letter-spacing:.04em}
.tsub .x{color:#484f58;margin:0 3px}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:#3fb950;
     margin-left:8px;vertical-align:middle}
.upd{font-size:12px;color:#6e7681}
/* "adesso" a tutta larghezza: il momento migliore ora sta sotto il grafico,
   dove ha senso leggerlo (dopo aver visto l'andamento). */
.hero{margin-bottom:14px}
/* Card leggermente traslucide: lo sfondo si intravede (effetto vetro) ma
   l'opacita' resta alta perche' i dati devono restare nitidi. */
.card{background:rgba(22,27,34,.86);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
      border:1px solid #21262d;border-radius:14px;padding:16px 18px;
      position:relative;overflow:hidden}
.k{font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.06em;margin-bottom:12px;
   display:flex;justify-content:space-between;align-items:center}
.k .when2{color:#6e7681;text-transform:none;letter-spacing:0;font-weight:400}
/* riga principale: onda + bussola + BADGE verdetto grande */
.now{display:flex;align-items:center;gap:16px}
.hs{font-size:44px;font-weight:600;line-height:.95;letter-spacing:-.02em}
.hs small{font-size:16px;color:#8b949e;font-weight:400}
.meta{font-size:13px;color:#8b949e;margin-top:5px}
.vbadge{margin-left:auto;text-align:center;padding:10px 16px;border-radius:12px;border:1px solid}
.vbadge .vb-t{font-size:18px;font-weight:600;letter-spacing:.02em;line-height:1;white-space:nowrap}
.vbadge .vb-s{font-size:10px;opacity:.8;margin-top:3px;text-transform:uppercase;letter-spacing:.04em}
.pill{display:inline-block;font-size:11px;font-weight:600;padding:2px 10px;border-radius:20px;color:#0d1117}
.mini{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:16px;padding-top:14px;
      border-top:1px solid #21262d}
.mini .v{font-size:16px;font-weight:600}
.mini .l{font-size:10px;color:#6e7681;text-transform:uppercase;letter-spacing:.03em;margin-top:2px}
.bar{height:4px;border-radius:3px;background:#30363d;overflow:hidden;margin-top:6px}
.bar i{display:block;height:100%;background:#58a6ff}
/* card "momento migliore": sotto il grafico, in orizzontale */
.best{margin-bottom:14px;display:grid;grid-template-columns:auto auto 1fr;
      align-items:center;gap:8px 20px}
.best .k{grid-column:1/-1;grid-row:1;margin-bottom:2px}
.best-day{grid-column:1;grid-row:2;font-size:26px;font-weight:600;line-height:1.05;letter-spacing:-.01em}
.best-none{grid-column:1;grid-row:2;font-size:19px;font-weight:600;color:#8b949e;line-height:1.25}
.best-badge{grid-column:2;grid-row:2;padding:8px 16px;border-radius:12px;border:1px solid;justify-self:start}
.best-badge .vb-t{font-size:17px;font-weight:600}
.best-sub{grid-column:3;grid-row:2;font-size:13px;color:#8b949e}
.best-foot{grid-column:1/-1;grid-row:3;margin-top:6px;padding-top:10px;
           border-top:1px solid #21262d;font-size:11px;color:#6e7681;line-height:1.5}
@media(max-width:640px){.best{grid-template-columns:1fr;gap:6px}
  .best-day,.best-none,.best-badge,.best-sub{grid-column:1;grid-row:auto}}
.big{font-size:26px;font-weight:600;line-height:1.15}
.sub{font-size:13px;color:#8b949e;margin-top:3px}
.chart{background:rgba(22,27,34,.86);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);border:1px solid #21262d;border-radius:14px;padding:14px 14px 8px;margin-bottom:14px}
.ct{display:flex;justify-content:space-between;font-size:12px;color:#8b949e;margin:0 2px 10px}
.legend{display:flex;gap:14px;flex-wrap:wrap;padding:8px 2px 0;font-size:11px;color:#8b949e}
.lg i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px}
.days{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:14px}
.day{border-left:3px solid #30363d;border-radius:0 12px 12px 0}
.day .d{font-size:12px;color:#8b949e;text-transform:capitalize;margin-bottom:8px;font-weight:500}
.day .h{font-size:24px;font-weight:600;display:flex;align-items:baseline;gap:8px}
.day .h .db{font-size:11px;font-weight:600;padding:2px 9px;border-radius:20px;border:1px solid}
.day .w{font-size:12px;color:#8b949e;margin-top:8px;line-height:1.5}
.sparkwrap{margin-top:8px;line-height:0}
.spark{display:block;width:100%;height:26px}
/* freccia tendenza: sta montando o scemando? */
.trend{font-size:12px;font-weight:600;margin-left:8px;vertical-align:middle;white-space:nowrap}
/* le card reagiscono: la pagina sembra viva, non stampata */
.card,.upsell{transition:border-color .18s,transform .18s}
.card:hover,.upsell:hover{border-color:#3d4754;transform:translateY(-1px)}
.day:hover{transform:translateY(-2px)}
.foot{font-size:11px;color:#6e7681;border-top:1px solid #21262d;padding-top:12px;line-height:1.6}
.brand{margin-top:10px;font-size:12px;color:#6e7681}
.brand b{color:#58a6ff;letter-spacing:.03em}
.trend{background:rgba(22,27,34,.86);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);border:1px solid #21262d;border-radius:12px;padding:10px 18px;
       margin-bottom:14px;font-size:13px;max-width:100%;min-width:0;
       white-space:normal;overflow-wrap:break-word}
.trend b{color:#e6edf3;font-weight:600}
.trend span{font-size:11px;color:#6e7681}
.cta{display:inline-block;margin-top:12px;background:#238636;color:#fff;font-size:13px;
     font-weight:600;padding:9px 18px;border-radius:8px;text-decoration:none;
     transition:transform .12s,background .2s}
.cta:hover{background:#2ea043;transform:translateY(-1px)}
.cta-wrap{text-align:center;margin-top:12px}
.legend2{margin-top:6px}
.lgt{font-size:11px;color:#6e7681;margin-right:2px}
.legend svg{vertical-align:middle;margin-right:5px}
.upsell{display:flex;align-items:center;gap:14px;flex-wrap:wrap;background:rgba(22,27,34,.86);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
        border:1px solid #26364a;border-radius:12px;padding:16px 18px;margin-bottom:14px;
        position:relative;overflow:hidden}
.upsell:before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:#58a6ff}
.up-l{flex:1;min-width:200px}
.up-t{font-size:15px;font-weight:600}
.up-s{font-size:12px;color:#8b949e;margin-top:4px}
.up-s b{color:#e6edf3}
.up-b{background:#1f6feb;color:#fff;font-size:13px;font-weight:600;padding:9px 18px;
      border-radius:8px;text-decoration:none;white-space:nowrap;transition:transform .12s,background .2s}
.up-b:hover{background:#388bfd;transform:translateY(-1px)}
@keyframes fadeup{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.hero,.chart,.days,.acc,.upsell,.trend{animation:fadeup .5s ease both}
.chart{animation-delay:.05s}.days{animation-delay:.1s}.acc{animation-delay:.15s}
.upsell{animation-delay:.2s}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.dot{animation:pulse 2s ease-in-out infinite}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
.spons{color:#6e7681}
.spons b{color:#e6edf3}
.lock{background:rgba(22,27,34,.86);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);border:1px dashed #30363d;border-radius:12px;padding:22px 18px;
      text-align:center;margin-bottom:14px}
.lock-t{font-size:15px;font-weight:600}
.lock-s{font-size:12px;color:#8b949e;margin-top:5px;margin-bottom:4px}
.meta2{font-size:12px;color:#6e7681;margin-top:3px}
.meta2 span{color:#8b949e;font-weight:600}
.dayps{display:flex;gap:6px;margin:0 2px 8px;flex-wrap:wrap}
.dayp{background:#0d1117;border:1px solid #21262d;color:#8b949e;font-size:11px;
      padding:3px 11px;border-radius:14px;cursor:pointer;font-family:inherit}
.dayp:hover{color:#e6edf3;border-color:#8b949e}
.dayp-now{border-color:#1c436e;color:#58a6ff}
/* selettore traccia inferiore */
.sw{display:inline-flex;align-items:center;gap:4px}
.swb{background:transparent;border:1px solid #21262d;color:#6e7681;font-size:11px;
     padding:2px 9px;border-radius:12px;cursor:pointer;font-family:inherit;transition:.15s}
.swb:hover{color:#e6edf3;border-color:#8b949e}
.swb.on{background:#0d2137;border-color:#1c436e;color:#58a6ff;font-weight:600}
.hint{color:#6e7681;font-style:italic;margin-left:auto}
.acc{background:rgba(22,27,34,.86);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);border:1px solid #21262d;border-radius:12px;padding:14px 18px;margin-bottom:14px}
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
@media(max-width:640px){
 /* Su mobile le onde in fondo non si vedono (il contenuto le copre) e costano
    batteria: le porto IN CIMA, capovolte, dietro l'intestazione — lì si vedono
    all'apertura, che e' il momento in cui contano. */
 .bg svg{bottom:auto;top:0;height:150px;transform:scaleY(-1);opacity:.85}
 .bg{background:radial-gradient(600px 300px at 50% 0%, #16304a 0%, transparent 65%),
     linear-gradient(180deg,#0d1117 0%,#0a1420 45%,#081019 100%)}
 body{padding:12px}
 h1{font-size:20px}
 .brandrow{gap:10px}
 .logo{width:26px;height:26px}
 .hs{font-size:34px}
 .card{padding:13px 14px}
 .now{gap:12px}
 .mini{grid-template-columns:repeat(2,1fr);gap:12px 10px}
 .vbadge{margin-left:auto;padding:8px 12px}
 .vbadge .vb-t{font-size:16px}
 .hero{gap:10px;margin-bottom:10px}
 .chart{padding:10px 8px 6px;margin-bottom:10px}
 .top{margin-bottom:12px}
 .upd{font-size:11px}
 .hint{display:none}
 .acc{padding:12px 14px}
 /* TOCCO: 23px non si centrano col dito. Apple raccomanda ~44px: alzo i
    bersagli con il padding e li rendo scorrevoli invece di mandarli a capo. */
 .dayps{flex-wrap:nowrap;overflow-x:auto;gap:8px;padding-bottom:4px;
        scrollbar-width:none;-webkit-overflow-scrolling:touch;
        max-width:100%;min-width:0}
 .dayps::-webkit-scrollbar{display:none}
 .dayp{padding:12px 15px;font-size:12px;border-radius:20px;flex:none}
 .swb{padding:11px 15px;font-size:12px;border-radius:18px}
 .sw{gap:6px}
 /* LEGGIBILITA': niente sotto 11px su schermo piccolo */
 .mini .l,.vbadge .vb-s,.tag,.legend,.acr,.acn,.day .w{font-size:11px}
 .lgt,.foot,.best-foot{font-size:11px}
 /* SCROLL: le card giorno da blocchi impilati a righe compatte (una per giorno):
    stessa informazione, un terzo dell'altezza. */
 .days{display:block;margin-bottom:10px}
 .day{display:grid;grid-template-columns:auto auto 1fr;align-items:center;
      gap:10px;padding:11px 14px;margin-bottom:8px;border-radius:0 10px 10px 0}
 .day .d{margin:0;font-size:12px;min-width:62px}
 .day .h{font-size:19px;gap:6px}
 .day .h .db{font-size:11px;padding:2px 8px}
 .day .w{margin:0;font-size:11px;text-align:right;line-height:1.35}
 .sparkwrap{display:none}
 .best{padding:13px 14px}
}
"""


# sfondo trasparente (si adatta alla loro pagina) e niente padding esterno.
# Lo sfondo a onde resta FUORI dall'embed: dentro la pagina del cliente
# competerebbe col loro design.
CSS_EMBED = """
body{background:transparent;padding:0}
.wrap{max-width:none}
.bg{display:none}
"""


# Onde di sfondo: due creste sfasate, path che si ripete ogni 1440px cosi'
# la traslazione del 50% e' continua (nessuno scatto al riavvolgimento).
SFONDO = """<div class="bg" aria-hidden="true"><svg viewBox="0 0 2880 320" preserveAspectRatio="none">
<path class="w1" fill="#1f6feb" fill-opacity=".22" d="M0,170 C240,120 480,215 720,168 C960,122 1200,212 1440,170
 C1680,120 1920,215 2160,168 C2400,122 2640,212 2880,170 L2880,320 L0,320 Z"/>
<path class="w2" fill="#58a6ff" fill-opacity=".14" d="M0,205 C300,165 540,245 720,208 C900,172 1140,248 1440,205
 C1740,165 1980,245 2160,208 C2340,172 2580,248 2880,205 L2880,320 L0,320 Z"/>
<path class="w1" fill="none" stroke="#58a6ff" stroke-opacity=".35" stroke-width="2"
 d="M0,170 C240,120 480,215 720,168 C960,122 1200,212 1440,170
 C1680,120 1920,215 2160,168 C2400,122 2640,212 2880,170"/>
</svg></div>"""


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


def _sparkline(vals, w=104, h=26):
    """Micro-grafico dell'andamento del giorno dentro la card: l'occhio capisce
    'sale, culmina, cala' senza leggere numeri."""
    if not vals or len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    rng = max(hi - lo, 0.12)                     # evita la riga piatta schiacciata
    pts = [(i * w / (len(vals) - 1), h - 3 - (v - lo) / rng * (h - 7))
           for i, v in enumerate(vals)]
    d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    area = d + f" L{w} {h} L0 {h} Z"
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" width="{w}" height="{h}">'
            f'<path d="{area}" fill="currentColor" opacity=".13"/>'
            f'<path d="{d}" fill="none" stroke="currentColor" stroke-width="1.6" '
            f'stroke-linecap="round" stroke-linejoin="round"/></svg>')


def _legenda():
    return "".join(f'<span class="lg"><i style="background:{c}"></i>{lab}</span>'
                   for lab, c in _LEGENDA)


JS_CHART = r"""
(function(){
const D=__DATA__, CRON=__CRON__;
const svg=document.getElementById('ch'); if(!svg||!D.length) return;
const $=id=>document.getElementById(id);
const fmt=v=>v.toFixed(1).replace('.',',');
let lastW=0;
// Traccia inferiore commutabile: il periodo e' un dato chiave e senza questo
// non sarebbe graficato da nessuna parte.
let METRICA='vento';
const METRICHE={
  vento:  {lab:'vento (kn)',  get:d=>d.k, col:d=>d.wc, min:12, dec:0, u:'kn'},
  periodo:{lab:'periodo (s)', get:d=>d.p, col:()=>'#7f77dd', min:8,  dec:0, u:'s'}
};
function render(){
const cw=Math.max(320,Math.round(svg.getBoundingClientRect().width)
  ||((svg.parentElement&&svg.parentElement.clientWidth)||940));
// Guard anti-loop: il render cambia l'altezza SVG -> il parent cambia ->
// il ResizeObserver riscatterebbe all'infinito. Ridisegna solo se la LARGHEZZA
// e' davvero cambiata (>4px), non a ogni micro-variazione di altezza.
if(Math.abs(cw-lastW)<5) return; lastW=cw;
// Safari: innerHTML su elementi SVG e' inaffidabile -> svuoto via DOM
while(svg.firstChild)svg.removeChild(svg.firstChild);
const mob=cw<560;
// PT ampio: sopra il grafico ci va l'etichetta data/ora che segue il puntatore
// GAP piu' ampio: fra onda e traccia inferiore ci sta la FASCIA QUALITA' SURF
// con la sua etichetta (prima era una strisciolina anonima di 6px).
const W=cw, PL=mob?32:46, PR=mob?6:14, PT=mob?42:46, HW=mob?140:170,
      GAP=mob?46:52, HK=mob?34:44, PB=mob?22:26;
const H=PT+HW+GAP+HK+PB;
svg.setAttribute('viewBox','0 0 '+W+' '+H);
svg.style.height=H+'px';
const n=D.length, bw=(W-PL-PR)/n;
const M=METRICHE[METRICA];
const hsMax=Math.max(1, Math.max.apply(null,D.map(d=>d.b))*1.08);
const kMax=Math.max(M.min, Math.max.apply(null,D.map(M.get))*1.15);
const X=i=>PL+(i+0.5)*bw, YH=v=>PT+HW*(1-v/hsMax);
const yk0=PT+HW+GAP+HK, YK=v=>yk0-HK*Math.min(v,kMax)/kMax;
const E=(t,a)=>{const e=document.createElementNS('http://www.w3.org/2000/svg',t);
  for(const k in a)e.setAttribute(k,a[k]);svg.appendChild(e);return e;};
let i0=0;while(i0<n){if(!D[i0].l){let j=i0;while(j<n&&!D[j].l)j++;
  E('rect',{x:PL+i0*bw,y:PT,width:(j-i0)*bw,height:yk0-PT,fill:'#010409',opacity:.5});i0=j;}else i0++;}
const fs=mob?10:11;
const step=hsMax<=2.5?0.5:1;
for(let v=0;v<=hsMax;v+=step){E('line',{x1:PL,y1:YH(v),x2:W-PR,y2:YH(v),stroke:'#ffffff14'});
  const t=E('text',{x:PL-5,y:YH(v)+3,'text-anchor':'end',fill:'#6e7681','font-size':fs});t.textContent=v.toFixed(1);}
{const rif=METRICA==='vento'?10:8;                 // riga di riferimento della traccia
 if(kMax>rif){E('line',{x1:PL,y1:YK(rif),x2:W-PR,y2:YK(rif),stroke:'#ffffff10'});
  const t=E('text',{x:PL-5,y:YK(rif)+3,'text-anchor':'end',fill:'#6e7681','font-size':fs-1});
  t.textContent=String(rif);}}
const days=[];D.forEach((d,i)=>{if(!days.length||days[days.length-1].dm!==d.dm)days.push({dm:d.dm,gg:d.gg,i:i});});
days.forEach((d,k)=>{if(k>0)E('line',{x1:PL+d.i*bw,y1:PT,x2:PL+d.i*bw,y2:yk0,stroke:'#30363d','stroke-dasharray':'3 3'});
  const t=E('text',{x:PL+d.i*bw+3,y:yk0+15,fill:'#8b949e','font-size':fs});
  t.textContent=mob?(d.gg+' '+d.dm.slice(0,2)):(d.gg+' '+d.dm);});
// Gradiente dell'area onda creato via DOM: con innerHTML il parser HTML perde
// il namespace SVG e su Safari il gradiente non nasce (area invisibile).
const NS='http://www.w3.org/2000/svg';
const defs=document.createElementNS(NS,'defs');
const grad=document.createElementNS(NS,'linearGradient');
grad.setAttribute('id','ga');grad.setAttribute('x1','0');grad.setAttribute('y1','0');
grad.setAttribute('x2','0');grad.setAttribute('y2','1');
[['0','0.35'],['1','0.02']].forEach(function(v){
 const st=document.createElementNS(NS,'stop');
 st.setAttribute('offset',v[0]);st.setAttribute('stop-color','#58a6ff');
 st.setAttribute('stop-opacity',v[1]);grad.appendChild(st);});
defs.appendChild(grad);svg.appendChild(defs);
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
const line=E('path',{d:spline(pH),fill:'none',stroke:'#58a6ff','stroke-width':2,
  'stroke-linecap':'round'});
if(!window.matchMedia||!matchMedia('(prefers-reduced-motion:reduce)').matches){
  try{const L=line.getTotalLength();line.style.strokeDasharray=L;line.style.strokeDashoffset=L;
   line.style.transition='stroke-dashoffset 1s ease';
   requestAnimationFrame(()=>{requestAnimationFrame(()=>{line.style.strokeDashoffset=0;});});}catch(e){}}
// FASCIA QUALITA' SURF: e' il verdetto ora per ora. Etichettata e spessa,
// altrimenti la legenda parla di colori che nessuno sa dove trovare.
{const qy=PT+HW+20, qh=mob?9:11;
 const ql=E('text',{x:PL,y:qy-4,fill:'#8b949e','font-size':fs});
 ql.textContent='qualità surf';
 E('rect',{x:PL,y:qy,width:W-PL-PR,height:qh,rx:2,fill:'#0d1117'});
 D.forEach((d,i)=>{E('rect',{x:PL+i*bw,y:qy,width:bw+0.5,height:qh,
   fill:d.sc,opacity:d.l?1:.45});});}
D.forEach((d,i)=>{const v=M.get(d);
 E('rect',{x:PL+i*bw+bw*0.15,y:YK(v),width:bw*0.7,height:yk0-YK(v),rx:1,
  fill:M.col(d),'fill-opacity':d.l?1:.5});});
let c1=E('text',{x:PL,y:PT-8,fill:'#8b949e','font-size':fs});c1.textContent='onda (m)';
let c2=E('text',{x:PL,y:yk0-HK-7,fill:'#8b949e','font-size':fs});c2.textContent=M.lab;
// Marker del picco: il momento clou si vede senza cercarlo
let iPk=0;D.forEach((d,i)=>{if(d.h>D[iPk].h)iPk=i;});
if(D[iPk].h>=0.3){const px=X(iPk),py=YH(D[iPk].h);
 E('circle',{cx:px,cy:py,r:3.5,fill:'#0d1117',stroke:'#58a6ff','stroke-width':2});
 const lx=Math.min(Math.max(px,PL+26),W-PR-26);
 const tp=E('text',{x:lx,y:Math.max(PT+11,py-11),'text-anchor':'middle',fill:'#c9d7e6',
   'font-size':mob?10:11,'font-weight':600});
 tp.textContent='picco '+fmt(D[iPk].h)+' m';}
const cl=E('line',{y1:PT,y2:yk0,stroke:'#e6edf3','stroke-width':1,opacity:0,'stroke-dasharray':'2 3'});
const cd=E('circle',{r:4.5,fill:'#58a6ff',stroke:'#0d1117','stroke-width':2,opacity:0});
cl.style.transition='opacity .15s, transform .08s linear';
cd.style.transition='opacity .15s, transform .08s linear';
// Etichetta data/ora ANCORATA AL PUNTATORE: mentre scorri sai sempre quando sei,
// senza dover alzare gli occhi in cima alla pagina.
const tw=mob?116:150, th=22;
const tg=E('g',{opacity:0});tg.style.transition='opacity .15s';
const tr_=document.createElementNS(svg.namespaceURI,'rect');
tr_.setAttribute('width',tw);tr_.setAttribute('height',th);tr_.setAttribute('rx',6);
tr_.setAttribute('fill','#161b22');tr_.setAttribute('stroke','#3d4754');
const tt=document.createElementNS(svg.namespaceURI,'text');
tt.setAttribute('text-anchor','middle');tt.setAttribute('fill','#e6edf3');
tt.setAttribute('font-size',mob?10:11);tt.setAttribute('font-weight','600');
tg.appendChild(tr_);tg.appendChild(tt);svg.appendChild(tg);
function setRO(i,active){const d=D[i];
 if($('ro-when'))$('ro-when').textContent=active?('· '+d.gg+' '+d.dm+' '+d.hh):'';
 if($('ro-hs'))$('ro-hs').textContent=fmt(d.h);
 const tr=$('ro-trend');
 if(tr){const T=d.tr>0?['↑','in aumento','#3fb950']:(d.tr<0?['↓','in calo','#e0a92e']
   :['→','stabile','#8b949e']);
  tr.textContent=T[0]+' '+T[1];tr.style.color=T[2];}
 if($('ro-band'))$('ro-band').textContent=fmt(d.a)+'–'+fmt(d.b)+' m';
 if($('ro-tp'))$('ro-tp').textContent=Math.round(d.p)+'s';
 if($('ro-wdc'))$('ro-wdc').textContent=d.wdc;
 const bg=$('ro-badge');if(bg){bg.querySelector('.vb-t').textContent=d.s;
  bg.style.background=d.tb;bg.style.color=d.tf;bg.style.borderColor=d.td;}
 const kv=$('ro-k');if(kv){kv.textContent=Math.round(d.k)+' kn';kv.style.color=d.wc;}
 if($('ro-kl'))$('ro-kl').textContent=d.kdc+' · '+d.kl;
 if($('ro-w'))$('ro-w').textContent=(d.w==null?'—':d.w+'°');
 if($('ro-sp')){$('ro-sp').textContent=d.sp+'%';const b=$('ro-spb');if(b)b.style.width=d.sp+'%';}
 const ar=$('ro-arrow');if(ar)ar.setAttribute('transform','rotate('+(d.wd+180)+' 32 32)');
 cl.setAttribute('x1',X(i));cl.setAttribute('x2',X(i));cl.setAttribute('opacity',active?0.5:0);
 cd.setAttribute('cx',X(i));cd.setAttribute('cy',YH(d.h));cd.setAttribute('opacity',active?1:0);
 // etichetta che segue il puntatore, tenuta dentro i bordi del grafico
 const tx=Math.min(Math.max(X(i),PL+tw/2),W-PR-tw/2);
 tr_.setAttribute('x',tx-tw/2);tr_.setAttribute('y',PT-th-8);
 tt.setAttribute('x',tx);tt.setAttribute('y',PT-th+7);
 tt.textContent=d.gg+' '+d.dm+' · '+d.hh+'  ·  '+fmt(d.h)+' m';
 tg.setAttribute('opacity',active?1:0);}
function idx(e){const r=svg.getBoundingClientRect();
 const px=(e.clientX-r.left)*W/r.width;
 return Math.max(0,Math.min(n-1,Math.round((px-PL)/bw-0.5)));}
svg.onpointermove=e=>setRO(idx(e),true);
svg.onpointerdown=e=>setRO(idx(e),true);
svg.onpointerleave=()=>setRO(0,false);
// Indice del PICCO di un giorno: e' il momento che interessa ("quando e' meglio
// quel giorno"). Un +12h fisso saltava al giorno dopo se la giornata era gia'
// iniziata a meta'.
function idxPicco(dm){let k=-1;
 D.forEach((d,i)=>{if(d.dm===dm&&(k<0||d.h>D[k].h))k=i;});return k;}
const dp=$('dayps');
if(dp){dp.innerHTML='';
 // "adesso": riporta al presente dopo aver esplorato il grafico
 const nb=document.createElement('button');nb.className='dayp dayp-now';
 nb.textContent='adesso';nb.onclick=()=>setRO(0,false);dp.appendChild(nb);
 days.forEach(d=>{const b=document.createElement('button');b.className='dayp';
  b.textContent=d.gg+' '+d.dm;
  b.onclick=()=>{const k=idxPicco(d.dm);if(k>=0)setRO(k,true);};dp.appendChild(b);});}
// Le card giorno portano il grafico sul picco di quel giorno
document.querySelectorAll('.day[data-dm]').forEach(c=>{
 c.style.cursor='pointer';
 c.onclick=()=>{const k=idxPicco(c.dataset.dm);
  if(k>=0){setRO(k,true);svg.scrollIntoView({behavior:'smooth',block:'center'});}};});
// Selettore della traccia inferiore (vento / periodo)
document.querySelectorAll('[data-metrica]').forEach(b=>{
 b.classList.toggle('on',b.dataset.metrica===METRICA);
 b.onclick=()=>{METRICA=b.dataset.metrica;lastW=0;render();};});
setRO(0,false);
// Count-up del numero principale all'ingresso: piccolo tocco, alza molto la
// percezione di "prodotto vivo". Una volta sola, non a ogni re-render.
if(!window.__alisee_up){window.__alisee_up=1;
 const el=$('ro-hs');
 if(el&&(!window.matchMedia||!matchMedia('(prefers-reduced-motion:reduce)').matches)){
  const tgt=D[0].h,t0=performance.now(),dur=800;
  const step=t=>{const p=Math.min(1,(t-t0)/dur);
   el.textContent=fmt(tgt*(1-Math.pow(1-p,3)));
   if(p<1)requestAnimationFrame(step);else el.textContent=fmt(tgt);};
  requestAnimationFrame(step);}}
}
render();
let rT;window.addEventListener('resize',()=>{clearTimeout(rT);rT=setTimeout(render,150);});
if(window.ResizeObserver)new ResizeObserver(()=>{clearTimeout(rT);rT=setTimeout(render,150);})
  .observe(svg.parentElement);
function cnt(){const now=new Date();let best=null;
 for(let g=0;g<2;g++)CRON.forEach(h=>{const t=new Date(Date.UTC(now.getUTCFullYear(),
  now.getUTCMonth(),now.getUTCDate()+g,h,0,0));if(t>now&&(!best||t<best))best=t;});
 if(!best)return;const m=Math.round((best-now)/60000);const el=$('cnt');
 if(el)el.textContent='si aggiorna tra '+(m>=60?Math.floor(m/60)+'h '+(m%60)+'m':m+' min');}
cnt();setInterval(cnt,30000);
})();
"""
