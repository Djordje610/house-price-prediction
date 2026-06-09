# -*- coding: utf-8 -*-
"""
Nacin 1: Linearna regresija (Ridge)
House Price Prediction - Ames Housing Dataset
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from shared.data_utils import (
    evaluate,
    get_train_test,
    plot_predictions,
    save_metrics,
)

OUT = Path(__file__).resolve().parent / "results"


def main() -> None:
    X_train, X_test, y_train, y_test, preprocessor = get_train_test()

    model = Pipeline(
        [
            ("prep", preprocessor),
            ("reg", Ridge(alpha=10.0, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = evaluate(y_test, y_pred)
    save_metrics(metrics, OUT)
    plot_predictions(
        y_test,
        y_pred,
        "Linearna regresija (Ridge) - predvidjeno vs stvarno",
        OUT / "predictions.png",
    )

    print("=== Linearna regresija (Ridge) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f}" if k != "R2" else f"  {k}: {v:.4f}")
    print(f"Grafik: {OUT / 'predictions.png'}")


if __name__ == "__main__":
    main()
