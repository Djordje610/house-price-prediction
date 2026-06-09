# -*- coding: utf-8 -*-
"""Trenira i cuva sva tri modela za frontend."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.data_utils import TARGET, get_train_test, load_raw, split_xy

MODELS_DIR = ROOT / "models"


def default_row(df: pd.DataFrame) -> dict:
    row = {}
    for col in df.columns:
        if col in (TARGET, "Id"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            row[col] = float(df[col].median())
        else:
            row[col] = str(df[col].mode(dropna=True).iloc[0])
    return row


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    X, _ = split_xy(df)
    defaults = default_row(df)

    with open(MODELS_DIR / "defaults.json", "w", encoding="utf-8") as f:
        json.dump(defaults, f, indent=2, ensure_ascii=False)

    with open(MODELS_DIR / "columns.json", "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, indent=2)

    X_train, X_test, y_train, y_test, preprocessor = get_train_test()

    specs = {
        "ridge": ("ridge_model.joblib", Ridge(alpha=10.0, random_state=42)),
        "random_forest": (
            "rf_model.joblib",
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        "xgboost": (
            "xgb_model.joblib",
            XGBRegressor(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    }

    for key, (fname, reg) in specs.items():
        pipe = Pipeline([("prep", preprocessor), ("reg", reg)])
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, MODELS_DIR / fname)
        print(f"Sacuvano: {MODELS_DIR / fname}")

    print("Modeli spremni za frontend.")


if __name__ == "__main__":
    main()
