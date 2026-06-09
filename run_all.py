# -*- coding: utf-8 -*-
"""Pokrece EDA i sva tri modela, pravi uporednu tabelu."""
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from shared.data_utils import load_raw, run_eda

MODELS = [
    ("01_linearna_regresija", "Linearna regresija (Ridge)"),
    ("02_random_forest", "Random Forest"),
    ("03_xgboost", "XGBoost"),
]


def run_script(rel_path: str) -> None:
    script = ROOT / rel_path
    print(f"\n>>> {script}")
    subprocess.run([sys.executable, str(script)], check=True)


def build_comparison() -> pd.DataFrame:
    rows = []
    for folder, name in MODELS:
        metrics_path = ROOT / folder / "results" / "metrics.json"
        with open(metrics_path, encoding="utf-8") as f:
            m = json.load(f)
        rows.append({"Model": name, **m})
    df = pd.DataFrame(rows)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "uporedna_tabela.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(df))
    width = 0.25
    ax.bar([i - width for i in x], df["RMSE"], width, label="RMSE")
    ax.bar(x, df["MAE"], width, label="MAE")
    ax.bar([i + width for i in x], df["R2"] * 10000, width, label="R2 x 10000")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["Model"], rotation=10)
    ax.set_title("Uporedba modela")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / "uporedba_modela.png", dpi=150)
    plt.close()
    return df


def main() -> None:
    print("Dataset:", load_raw().shape)
    run_eda(ROOT / "results" / "eda")

    for folder, _ in MODELS:
        run_script(f"{folder}/train.py")

    df = build_comparison()
    print("\n=== UPOREDNA TABELA ===")
    print(df.to_string(index=False))
    print(f"\nSacuvano: {ROOT / 'results'}")


if __name__ == "__main__":
    main()
