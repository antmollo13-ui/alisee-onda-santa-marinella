# ALISEE Onda — Santa Marinella

Previsione onda e vento a 72 ore per Santa Marinella, calibrata su strumenti reali:
la **boa ondametrica RON** e la **stazione mareografica RMN** di Civitavecchia (~8 km).

## Cosa pubblica

Ogni run rigenera e pubblica su GitHub Pages:

| Pagina | A cosa serve |
|---|---|
| `index.html` | dashboard completa |
| `widget.html` | widget compatto, incorporabile |
| `previsione_onda_72h.csv` | i dati grezzi della previsione |

### Incorporare il widget

```html
<iframe src="https://USERNAME.github.io/REPO/widget.html"
        width="100%" height="380" style="border:0" loading="lazy"></iframe>
```

## Precisione (verificata out-of-sample)

Scarto medio rispetto agli strumenti, su periodi mai visti in addestramento
(onda: 2026, 4.666 ore — vento: 2025, 8.571 ore):

| | ALISEE | modello standard |
|---|---|---|
| altezza onda | ± 12 cm | ± 17 cm |
| periodo | ± 0,54 s | ± 0,86 s |
| vento | ± 1,16 kn | ± 1,71 kn |

Lo scarto cresce con l'altezza: ± 7 cm sotto i 40 cm, ± 32 cm sopra i 2 m.
I valori sono l'errore sull'analisi: su una previsione a 72 ore entrambi sbagliano di più.

## Come gira

`.github/workflows/dashboard.yml` esegue `previsione_onda.py` alle **05, 11, 17, 23 UTC**,
subito dopo l'uscita delle run del modello d'onda, e pubblica su Pages.
Nessun dato in tempo reale richiesto: i modelli sono pre-addestrati.

## Contenuto

- `previsione_onda.py` — scarica il forecast, applica i modelli, genera dashboard e widget
- `onda_features.py` / `vento_features.py` — feature engineering
- `modello_onda.pkl` / `modello_vento.pkl` — modelli addestrati (XGBoost)

Gli script di addestramento, validazione e benchmark non fanno parte di questo repository.
