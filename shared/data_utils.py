# -*- coding: utf-8 -*-
"""Zajednicko ucitavanje, priprema i evaluacija za sva tri modela."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "train.csv"
TARGET = "SalePrice"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_raw() -> pd.DataFrame:
    if not DATA_PATH.exists():
        from shared.download_data import main as download

        download()
    return pd.read_csv(DATA_PATH)


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    y = df[TARGET].copy()
    X = df.drop(columns=[TARGET, "Id"], errors="ignore")
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )


def get_train_test() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    df = load_raw()
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    preprocessor = build_preprocessor(X_train)
    return X_train, X_test, y_train, y_test, preprocessor


def evaluate(y_true, y_pred) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def save_metrics(metrics: Dict[str, float], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def plot_predictions(y_true, y_pred, title: str, out_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolors="k", linewidths=0.3)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "r--", lw=2, label="Idealno")
    plt.xlabel("Stvarna cena (SalePrice)")
    plt.ylabel("Predvidjena cena")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_feature_importance(names, values, title: str, out_path: Path, top_n: int = 15) -> None:
    idx = np.argsort(values)[::-1][:top_n]
    top_names = [names[i] for i in idx]
    top_vals = [values[i] for i in idx]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=top_vals, y=top_names, orient="h")
    plt.xlabel("Vaznost")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_eda(out_dir: Path) -> None:
    """Jednostavna EDA za izvestaj (jednom, u shared)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_raw()

    plt.figure(figsize=(8, 5))
    sns.histplot(df[TARGET], kde=True, bins=40)
    plt.title("Distribucija SalePrice")
    plt.xlabel("SalePrice ($)")
    plt.tight_layout()
    plt.savefig(out_dir / "01_distribucija_cene.png", dpi=150)
    plt.close()

    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr(numeric_only=True)[TARGET].sort_values(ascending=False).head(12)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=corr.values, y=corr.index, orient="h")
    plt.title("Top korelacije sa SalePrice")
    plt.tight_layout()
    plt.savefig(out_dir / "02_korelacije.png", dpi=150)
    plt.close()

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False).head(15)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=missing.values, y=missing.index, orient="h")
    plt.title("Kolone sa najvise nedostajucih vrednosti")
    plt.tight_layout()
    plt.savefig(out_dir / "03_nedostajuce_vrednosti.png", dpi=150)
    plt.close()
