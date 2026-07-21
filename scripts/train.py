"""Train a regression model for a `model_runs` row."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from scripts.common import load_dataset, run_with_status


def build_model(model_type: str):
    mt = (model_type or "").lower()
    if mt in ("linear", "ridge", "lasso"):
        return Ridge()
    if mt == "rf":
        return RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)
    if mt in ("gbm", "gbr"):
        return GradientBoostingRegressor(random_state=0)
    if mt == "xgboost":
        try:
            from xgboost import XGBRegressor  # type: ignore
            return XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=0)
        except Exception:
            return GradientBoostingRegressor(random_state=0)
    if mt == "svr":
        return SVR(kernel="rbf")
    if mt == "dnn":
        return MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=0)
    return GradientBoostingRegressor(random_state=0)


def train(sb, payload, run_id) -> dict[str, Any]:
    run = sb.table("model_runs").select("*").eq("id", run_id).single().execute().data
    hp = run.get("hyperparameters") or {}
    target = hp.get("target")
    test_size = float(hp.get("test_size", 0.2))
    cv_folds = int(hp.get("cv_folds", 5))
    dsv_id = run.get("dataset_version_id")
    if not (target and dsv_id):
        raise RuntimeError("Run is missing target or dataset_version_id")

    df = load_dataset(sb, dsv_id).dropna()
    if target not in df.columns:
        raise RuntimeError(f"Target column '{target}' not in dataset columns: {list(df.columns)}")

    y = df[target].astype(float)
    X = df.drop(columns=[target]).select_dtypes(include=[np.number]).astype(float)
    if X.shape[1] == 0:
        raise RuntimeError("No numeric feature columns found")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)
    model = build_model(run.get("model_type", "gbm"))
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    metrics = {
        "r2": float(r2_score(y_te, y_pred)),
        "mae": float(mean_absolute_error(y_te, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_te, y_pred))),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "n_features": int(X.shape[1]),
        "feature_names": list(X.columns),
    }
    try:
        cv = cross_val_score(build_model(run.get("model_type", "gbm")), X, y,
                             cv=KFold(cv_folds, shuffle=True, random_state=0), scoring="r2")
        metrics["cv_r2_mean"] = float(cv.mean())
        metrics["cv_r2_std"] = float(cv.std())
    except Exception:
        pass

    return {"metrics": metrics}


if __name__ == "__main__":
    run_with_status("model_runs", train)
