"""Multi-objective optimization (NSGA-II) over a trained strength model."""
from __future__ import annotations

from typing import Any

import numpy as np

from scripts.common import load_dataset, run_with_status
from scripts.train import build_model


def run_optimize(sb, payload, run_id) -> dict[str, Any]:
    opt = sb.table("optimization_runs").select("*").eq("id", run_id).single().execute().data
    objectives = opt.get("objectives") or {}
    constraints = opt.get("constraints") or {}
    model_run_id = objectives.get("model_run_id")
    if not model_run_id:
        raise RuntimeError("optimization_runs.objectives.model_run_id is required")

    model_run = sb.table("model_runs").select("*").eq("id", model_run_id).single().execute().data
    hp = model_run.get("hyperparameters") or {}
    target = hp.get("target")
    dsv_id = model_run.get("dataset_version_id")
    df = load_dataset(sb, dsv_id).dropna()
    y = df[target].astype(float)
    X = df.drop(columns=[target]).select_dtypes(include=[np.number]).astype(float)

    model = build_model(model_run.get("model_type", "gbm"))
    model.fit(X, y)

    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling

    lows = X.min().to_numpy()
    highs = X.max().to_numpy()
    max_cement = float(constraints.get("max_cement", highs.max() + 1))
    min_strength = float(constraints.get("min_strength", 0.0))
    features = list(X.columns)
    cement_idx = next((i for i, c in enumerate(features) if "cement" in c.lower()), None)

    class Mix(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=X.shape[1], n_obj=2, n_constr=2, xl=lows, xu=highs)

        def _evaluate(self, x, out, *args, **kwargs):
            import pandas as pd
            pred = float(model.predict(pd.DataFrame([x], columns=features))[0])
            cement = float(x[cement_idx]) if cement_idx is not None else 0.0
            out["F"] = [-pred, cement]
            out["G"] = [min_strength - pred, cement - max_cement]

    nsga = (constraints or {}).get("nsga") or {}
    algo = NSGA2(
        pop_size=int(nsga.get("pop_size", 60)),
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )
    res = minimize(Mix(), algo, ("n_gen", int(nsga.get("generations", 40))), verbose=False)

    front: list[dict[str, Any]] = []
    if res.F is not None:
        for xv, fv in zip(res.X, res.F):
            front.append({
                "features": {f: float(v) for f, v in zip(features, xv)},
                "predicted_strength": float(-fv[0]),
                "cement": float(fv[1]),
            })
    return {"pareto_front": front[:200]}


if __name__ == "__main__":
    run_with_status("optimization_runs", run_optimize)
