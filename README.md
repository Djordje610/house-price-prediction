# House Price Prediction

Predikcija cena nekretnina pomoću mašinskog učenja na **Ames Housing** datasetu. Projekat upoređuje tri modela i nudi web interfejs za unos karakteristika kuće i procenu cene.

## Dataset

- **Izvor:** Ames Housing (Kaggle format)
- **Fajlovi:** `data/train.csv` (1460 kuća), `data/test.csv` (1459 kuća)
- **Ciljna promenljiva:** `SalePrice`
- **Atributi:** ~80 (numeričkiểm i kategorijske kolone)

Ako podaci nisu u repou, preuzmi ih:

```bash
python shared/download_data.py
```

## Modeli

| Folder | Model | Opis |
|--------|-------|------|
| `01_linearna_regresija/` | Ridge | Linearna regresija sa L2 regularizacijom |
| `02_random_forest/` | Random Forest | Ensemble od 300 stabala |
| `03_xgboost/` | XGBoost | Gradient boosting |

### Rezultati (test skup, 20%)

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Linearna regresija (Ridge) | 30 642 | 19 038 | 0.878 |
| Random Forest | 28 748 | 17 473 | 0.892 |
| XGBoost | 24 292 | 14 975 | 0.923 |

## Instalacija

Potreban je **Python 3.10+**.

```bash
git clone https://github.com/TVOJ_USERNAME/ML_house_price_prediction.git
cd ML_house_price_prediction
python -m pip install -r requirements.txt
```

## Pokretanje

### Web aplikacija (Streamlit)

```bash
python run.py
```

Otvori u browseru: **http://localhost:8501**

Prvi put se automatski treniraju i čuvaju modeli u `models/` (potrebno par minuta).

**Windows (alternativa):**

- `pokreni.bat` — pokretanje sa terminalom
- `POKRENI_APLIKACIJU.vbs` — pokretanje u pozadini
- `ugasi.bat` / `UGASI_APLIKACIJU.vbs` — gašenje aplikacije

### Treniranje svih modela

```bash
python run_all.py
```

Pokreće EDA, trenira sva tri modela i pravi uporednu tabelu u `results/`.

### Pojedinačno treniranje

```bash
python 01_linearna_regresija/train.py
python 02_random_forest/train.py
python 03_xgboost/train.py
```

### Čuvanje modela za frontend

```bash
python shared/save_models.py
```

## Struktura projekta

```
ML_house_price_prediction/
├── 01_linearna_regresija/   # Ridge model + train.py
├── 02_random_forest/        # Random Forest + train.py
├── 03_xgboost/              # XGBoost + train.py
├── shared/                  # Zajednička priprema podataka i evaluacija
├── frontend/                # Streamlit web UI
├── data/                    # train.csv, test.csv
├── models/                  # Sačuvani modeli (.joblib) i konfiguracija
├── results/                 # EDA grafici i uporedna tabela
├── run.py                   # Pokretanje frontenda
├── run_all.py               # EDA + sva tri modela
└── requirements.txt
```

## Rezultati

Nakon treniranja:

- `results/eda/` — EDA grafici (distribucija cene, korelacije, nedostajuće vrednosti)
- `results/uporedna_tabela.csv` — uporedba metrika
- `results/uporedba_modela.png` — vizuelna uporedba
- `0X_*/results/` — metrike, predikcije i feature importance po modelu

## Tehnologije

- Python, pandas, NumPy
- scikit-learn, XGBoost
- Streamlit (frontend)
- matplotlib, seaborn (vizualizacija)

## Napomena o reproduktivnosti

Treniranje uvek daje iste rezultate jer su fiksirani `random_state=42` i isti hiperparametri — to je namerno radi ponovljivosti eksperimenata.

## Licenca

Projekat je namenjen edukativnoj upotrebi. Dataset Ames Housing ima sopstvenu licencu (Kaggle / originalni autori).
