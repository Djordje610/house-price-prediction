# -*- coding: utf-8 -*-
"""
Nacin 3: XGBoost (Gradient Boosting)
House Price Prediction - Ames Housing Dataset
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from shared.data_utils import (
    evaluate,
    get_train_test,
    plot_feature_importance,
    plot_predictions,
    save_metrics,
)

OUT = Path(__file__).resolve().parent / "results"


def main() -> None:
    X_train, X_test, y_train, y_test, preprocessor = get_train_test()

    model = Pipeline(
        [
            ("prep", preprocessor),
            (
                "reg",
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
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = evaluate(y_test, y_pred)
    save_metrics(metrics, OUT)
    plot_predictions(
        y_test,
        y_pred,
        "XGBoost - predvidjeno vs stvarno",
        OUT / "predictions.png",
    )

    prep = model.named_steps["prep"]
    reg = model.named_steps["reg"]
    feature_names = prep.get_feature_names_out()
    plot_feature_importance(
        feature_names,
        reg.feature_importances_,
        "XGBoost - top 15 feature-a",
        OUT / "feature_importance.png",
    )

    print("=== XGBoost ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f}" if k != "R2" else f"  {k}: {v:.4f}")
    print(f"Grafici: {OUT}")


if __name__ == "__main__":
    main()
