"""Compute SHAP values for a completed model on its training dataset."""
from __future__ import annotations

from typing import Any

import numpy as np

from scripts.common import load_dataset, run_with_status
from scripts.train import build_model


def run_shap(sb, payload, run_id) -> dict[str, Any]:
    shap_run = sb.table("shap_runs").select("*").eq("id", run_id).single().execute().data
    model_run_id = shap_run.get("model_run_id")
    cfg = shap_run.get("config") or {}
    n_samples = int(cfg.get("n_samples", 200))

    model_run = sb.table("model_runs").select("*").eq("id", model_run_id).single().execute().data
    hp = model_run.get("hyperparameters") or {}
    target = hp.get("target")
    dsv_id = model_run.get("dataset_version_id")
    df = load_dataset(sb, dsv_id).dropna()
    y = df[target].astype(float)
    X = df.drop(columns=[target]).select_dtypes(include=[np.number]).astype(float)

    model = build_model(model_run.get("model_type", "gbm"))
    model.fit(X, y)

    import shap  # type: ignore
    sample = X.sample(min(n_samples, len(X)), random_state=0)
    try:
        explainer = shap.Explainer(model, sample)
        values = explainer(sample)
        vals = np.abs(values.values).mean(axis=0)
    except Exception:
        explainer = shap.KernelExplainer(model.predict, sample.iloc[: min(50, len(sample))])
        raw = explainer.shap_values(sample.iloc[: min(50, len(sample))])
        vals = np.abs(np.array(raw)).mean(axis=0)

    ranking = sorted(
        [{"feature": f, "mean_abs_shap": float(v)} for f, v in zip(X.columns, vals)],
        key=lambda r: r["mean_abs_shap"],
        reverse=True,
    )
    return {"results": {"feature_importance": ranking, "n_samples": int(len(sample))}}


if __name__ == "__main__":
    run_with_status("shap_runs", run_shap)
