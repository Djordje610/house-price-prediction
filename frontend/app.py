# -*- coding: utf-8 -*-
"""
House Price Prediction - Frontend
Pokretanje: streamlit run frontend/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"

MODEL_OPTIONS = {
    "Linearna Regresija": "ridge_model.joblib",
    "Random Forest": "rf_model.joblib",
    "XGBoost": "xgb_model.joblib",
}

MODEL_SHORT = {
    "Linearna Regresija": "Linearna Regresija",
    "Random Forest": "Random Forest",
    "XGBoost": "XGBoost",
}

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_json(name: str) -> dict | list:
    with open(MODELS_DIR / name, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_model(filename: str):
    return joblib.load(MODELS_DIR / filename)


def build_input(user: dict, defaults: dict, columns: list) -> pd.DataFrame:
    row = {**defaults, **user}
    return pd.DataFrame([{c: row.get(c, defaults.get(c)) for c in columns}])


def fmt_price(value: float) -> str:
    return f"${value:,.0f}"


def fmt_price_suffix(value: float) -> str:
    """Broj sa $ na kraju, npr. 180,921 $"""
    return f"{value:,.0f} $"


# Skracenice iz dataseta -> puni nazivi na srpskom
QUALITY_LABELS = {
    "Ex": "Odlično",
    "Gd": "Dobro",
    "TA": "Prosečno",
    "Fa": "Slabo",
    "Po": "Loše",
    "None": "Nema / nije primenjivo",
}

YES_NO_LABELS = {
    "Y": "Da",
    "N": "Ne",
}

FOUNDATION_LABELS = {
    "PConc": "Beton (armirana ploča)",
    "CBlock": "Betonski blokovi",
    "BrkTil": "Cigla i pločice",
    "Wood": "Drvo",
    "Slab": "Ploča na tlu",
    "Stone": "Kamen",
    "Cemnt": "Cement",
}

HOUSE_STYLE_LABELS = {
    "1Story": "Jednospratnica",
    "2Story": "Dvospratnica",
    "1.5Fin": "Polusprat — završeno potkrovlje",
    "1.5Unf": "Polusprat — nezavršeno potkrovlje",
    "2.5Fin": "Dva i po sprata — završeno",
    "2.5Unf": "Dva i po sprata — nezavršeno",
    "SFoyer": "Jednospratnica sa foajéom",
    "SLvl": "Jednospratnica na više nivoa",
}

COLUMN_LABELS: dict[str, dict[str, str]] = {
    "KitchenQual": QUALITY_LABELS,
    "ExterQual": QUALITY_LABELS,
    "ExterCond": QUALITY_LABELS,
    "BsmtQual": QUALITY_LABELS,
    "HeatingQC": QUALITY_LABELS,
    "CentralAir": YES_NO_LABELS,
    "Foundation": FOUNDATION_LABELS,
    "HouseStyle": HOUSE_STYLE_LABELS,
}


def select_field(df: pd.DataFrame, col: str, label: str, defaults: dict) -> str:
    raw_options = sorted(df[col].dropna().astype(str).unique())
    label_map = COLUMN_LABELS.get(col, {})

    if not label_map:
        idx = raw_options.index(str(defaults.get(col, raw_options[0]))) if str(defaults.get(col)) in raw_options else 0
        return st.selectbox(label, raw_options, index=idx)

    display_to_raw: dict[str, str] = {}
    display_options: list[str] = []
    for raw in raw_options:
        display = label_map.get(raw, raw)
        display_to_raw[display] = raw
        display_options.append(display)

    default_raw = str(defaults.get(col, raw_options[0]))
    default_display = label_map.get(default_raw, default_raw)
    idx = display_options.index(default_display) if default_display in display_options else 0
    chosen_display = st.selectbox(label, display_options, index=idx)
    return display_to_raw[chosen_display]


def main() -> None:
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(90deg, #1e3a5f, #2d6a9f);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .sub-title { color: #5a6a7a; font-size: 1rem; margin-bottom: 1.5rem; }
        .price-box {
            background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
            color: white;
            padding: 1.8rem 2rem;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 8px 24px rgba(30,58,95,0.25);
            margin: 1rem 0;
        }
        .price-label { font-size: 0.95rem; opacity: 0.9; margin-bottom: 0.3rem; }
        .price-value { font-size: 2.8rem; font-weight: 800; letter-spacing: -1px; }
        .metric-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1rem 0.5rem;
            text-align: center;
            min-height: 108px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            box-sizing: border-box;
        }
        .metric-card .model-name {
            font-weight: 700;
            font-size: 0.9rem;
            line-height: 1.35;
            min-height: 2.7em;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .metric-card .model-price {
            font-size: 1.25rem;
            font-weight: 700;
            color: #1e3a5f;
            margin-top: 0.35rem;
        }
        .summary-box {
            background: #e8f4fc;
            border: 1px solid #b8d9f0;
            border-radius: 12px;
            padding: 1rem 1.25rem;
            margin-top: 1rem;
            color: #1e3a5f;
            line-height: 1.8;
        }
        .summary-box .summary-line {
            margin: 0.35rem 0;
        }
        .summary-box .summary-label {
            font-weight: 600;
        }
        .summary-box .summary-value {
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if not (MODELS_DIR / "ridge_model.joblib").exists():
        st.error("Modeli nisu sacuvani. Pokreni: `python shared/save_models.py`")
        st.stop()

    defaults = load_json("defaults.json")
    columns = load_json("columns.json")
    df = pd.read_csv(ROOT / "data" / "train.csv")

    st.markdown('<p class="main-title">🏠 House Price Prediction</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Ames Housing · tri ML modela · predikcija SalePrice</p>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("⚙️ Model")
        model_label = st.selectbox("Izaberi algoritam", list(MODEL_OPTIONS.keys()))
        st.divider()
        st.caption("Unesi karakteristike kuće. Ostala polja se automatski popunjavaju prosečnim vrednostima iz dataseta.")

    col1, col2 = st.columns([1.1, 0.9], gap="large")

    user_input: dict = {}

    with col1:
        st.subheader("📋 Karakteristike nekretnine")

        t1, t2 = st.tabs(["Osnovno", "Detalji"])

        with t1:
            st.caption("Najbitniji atributi za procenu cene")
            c1, c2 = st.columns(2)
            with c1:
                user_input["OverallQual"] = st.slider(
                    "Ukupan kvalitet (1–10)",
                    1, 10, int(defaults.get("OverallQual", 6)),
                )
                user_input["GrLivArea"] = st.number_input(
                    "Stambena površina (ft²)", 300, 6000, int(defaults.get("GrLivArea", 1500)),
                )
                user_input["LotArea"] = st.number_input(
                    "Površina parcele (ft²)", 1000, 100000, int(defaults.get("LotArea", 9000)),
                )
                user_input["YearBuilt"] = st.number_input(
                    "Godina gradnje", 1872, 2025, int(defaults.get("YearBuilt", 2000)),
                )
                user_input["Neighborhood"] = select_field(df, "Neighborhood", "Kvart (lokacija)", defaults)
            with c2:
                user_input["BedroomAbvGr"] = st.slider(
                    "Spavaće sobe", 0, 8, int(defaults.get("BedroomAbvGr", 3)),
                )
                user_input["FullBath"] = st.slider(
                    "Kupatilo sa kompletnom opremom",
                    0, 4, int(defaults.get("FullBath", 2)),
                    help="Umivaonik, toalet i kada ili tuš",
                )
                user_input["GarageCars"] = st.slider(
                    "Mesta u garaži", 0, 4, int(defaults.get("GarageCars", 2)),
                )
                user_input["TotRmsAbvGrd"] = st.slider(
                    "Ukupan broj soba", 2, 15, int(defaults.get("TotRmsAbvGrd", 6)),
                )
                user_input["CentralAir"] = select_field(df, "CentralAir", "Centralna klima", defaults)

        with t2:
            st.caption("Dodatne karakteristike — ostalo se popunjava automatski")
            c3, c4 = st.columns(2)
            with c3:
                user_input["TotalBsmtSF"] = st.number_input(
                    "Površina podruma (ft²)", 0, 3000, int(defaults.get("TotalBsmtSF", 900)),
                )
                user_input["1stFlrSF"] = st.number_input(
                    "Površina prvog sprata (ft²)", 0, 4000, int(defaults.get("1stFlrSF", 1100)),
                )
                user_input["2ndFlrSF"] = st.number_input(
                    "Površina drugog sprata (ft²)", 0, 2000, int(defaults.get("2ndFlrSF", 0)),
                )
                user_input["GarageArea"] = st.number_input(
                    "Površina garaže (ft²)", 0, 1500, int(defaults.get("GarageArea", 400)),
                )
                user_input["LotFrontage"] = st.number_input(
                    "Širina parcele na ulici (ft)", 0, 200,
                    int(defaults.get("LotFrontage", 70) or 70),
                )
                user_input["YearRemodAdd"] = st.number_input(
                    "Godina renoviranja", 1950, 2025, int(defaults.get("YearRemodAdd", 2000)),
                )
                user_input["HalfBath"] = st.slider(
                    "Polukupatilo (WC + lavabo)", 0, 3, int(defaults.get("HalfBath", 0)),
                )
                user_input["Fireplaces"] = st.slider(
                    "Kamini", 0, 3, int(defaults.get("Fireplaces", 0)),
                )
            with c4:
                user_input["HouseStyle"] = select_field(df, "HouseStyle", "Stil kuće", defaults)
                user_input["KitchenQual"] = select_field(df, "KitchenQual", "Kvalitet kuhinje", defaults)
                user_input["ExterQual"] = select_field(df, "ExterQual", "Kvalitet eksterijera", defaults)
                user_input["ExterCond"] = select_field(df, "ExterCond", "Stanje eksterijera", defaults)
                user_input["BsmtQual"] = select_field(df, "BsmtQual", "Kvalitet podruma", defaults)
                user_input["HeatingQC"] = select_field(df, "HeatingQC", "Kvalitet grejanja", defaults)
                user_input["OverallCond"] = st.slider(
                    "Opšte stanje (1–10)", 1, 10, int(defaults.get("OverallCond", 5)),
                )
                user_input["Foundation"] = select_field(df, "Foundation", "Tip temelja", defaults)

        predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)

    with col2:
        st.subheader("📊 Rezultat")

        if predict_btn:
            X = build_input(user_input, defaults, columns)
            model = load_model(MODEL_OPTIONS[model_label])
            price = float(model.predict(X)[0])

            st.markdown(
                f"""
                <div class="price-box">
                    <div class="price-label">Predviđena cena ({model_label})</div>
                    <div class="price-value">{fmt_price(price)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("**Poređenje sva tri modela:**")
            cmp_cols = st.columns(3)
            for i, (label, fname) in enumerate(MODEL_OPTIONS.items()):
                m = load_model(fname)
                p = float(m.predict(X)[0])
                with cmp_cols[i]:
                    short = MODEL_SHORT.get(label, label)
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="model-name">{short}</div>'
                        f'<div class="model-price">{fmt_price(p)}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            avg_train = df["SalePrice"].mean()
            st.markdown(
                f"""
                <div class="summary-box">
                  <div class="summary-line">
                    <span class="summary-label">Prosečna cena u datasetu:</span>
                    <span class="summary-value">{fmt_price_suffix(avg_train)}</span>
                  </div>
                  <div class="summary-line" style="margin-top:0.75rem;">
                    <span class="summary-label">Vaša predikcija:</span>
                    <span class="summary-value">{fmt_price_suffix(price)}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="background:#f1f5f9;border:2px dashed #cbd5e1;border-radius:16px;
                padding:3rem 2rem;text-align:center;color:#64748b;">
                <div style="font-size:3rem;margin-bottom:0.5rem">🏡</div>
                <b>Unesite podatke i kliknite Predict</b><br>
                <small>Rezultat će se prikazati ovde</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()
    with st.expander("ℹ️ O projektu"):
        st.markdown(
            """
            - **Dataset:** Ames Housing (1460 kuća, Iowa)
            - **Modeli:** Linearna Regresija, Random Forest, XGBoost
            - **Metrike (test):** XGBoost RMSE ≈ $24 292, R² ≈ 0.92
            - **Osnovno:** 10 najbitnijih atributa · **Detalji:** još 16 · ostatak (~55) automatski iz dataseta.
            """
        )


if __name__ == "__main__":
    main()
