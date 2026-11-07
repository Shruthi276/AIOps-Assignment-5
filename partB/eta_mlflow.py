import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import eta_data

import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient

MODEL_NAME = "eta-model"
EXPERIMENT = "swiftparcel-eta"
RUSH_HOURS = [7, 8, 9, 16, 17, 18, 19]
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")


class EtaPyfunc(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        b = joblib.load(context.artifacts["bundle"])
        self.model, self.features, self.version = b["model"], list(b["feature_columns"]), b["version"]

    def predict(self, context, model_input, params=None):
        X = pd.DataFrame(model_input)[self.features]
        eta = np.round(self.model.predict(X).astype(float), 1)
        return pd.DataFrame({"eta_minutes": eta, "model_version": [self.version] * len(X)})


def _pinned_requirements():
    import sklearn
    return [f"mlflow=={mlflow.__version__}", f"scikit-learn=={sklearn.__version__}",
            f"pandas=={pd.__version__}", f"numpy=={np.__version__}", f"joblib=={joblib.__version__}"]


def _setup():
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(ROOT / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT)


def registry_table():
    client = MlflowClient(tracking_uri=TRACKING_URI)
    rows = []
    for mv in sorted(client.search_model_versions(f"name='{MODEL_NAME}'"), key=lambda m: int(m.version)):
        full = client.get_model_version(MODEL_NAME, mv.version)
        rows.append({"version": int(mv.version), "aliases": ",".join(full.aliases) or "-",
                     "bundle": full.tags.get("bundle_version", "?"), "run_id": mv.run_id})
    return pd.DataFrame(rows)


def register():
    _setup()
    from mlflow.models import infer_signature
    client = MlflowClient(tracking_uri=TRACKING_URI)
    fresh = eta_data.make_data(2000, seed=99)
    example = fresh[eta_data.FEATURES].head(5)
    versions = {}
    for v in ("v1", "v2"):
        path = ROOT / "data" / f"eta_{v}.joblib"
        bundle = joblib.load(path)
        wrapper = EtaPyfunc()
        wrapper.model, wrapper.features, wrapper.version = bundle["model"], eta_data.FEATURES, v
        signature = infer_signature(example, wrapper.predict(None, example))
        with mlflow.start_run(run_name=f"eta-{v}"):
            mlflow.log_param("estimator", type(bundle["model"]).__name__)
            mlflow.log_metric("fresh_mae", float(np.mean(np.abs(
                bundle["model"].predict(fresh[eta_data.FEATURES]) - fresh.eta_minutes))))
            info = mlflow.pyfunc.log_model(
                name="model", python_model=EtaPyfunc(), artifacts={"bundle": str(path)},
                signature=signature, input_example=example,
                pip_requirements=_pinned_requirements(), registered_model_name=MODEL_NAME)
        mv = info.registered_model_version
        client.set_model_version_tag(MODEL_NAME, mv, "bundle_version", v)
        versions[v] = mv
    client.set_registered_model_alias(MODEL_NAME, "prod", versions["v1"])
    client.set_registered_model_alias(MODEL_NAME, "candidate", versions["v2"])
    table = registry_table()
    print(table.to_string(index=False))
    return table


def decide(prod, cand, min_rel_gain=0.03, max_p95_ms=10.0, max_slice_regression=0.05):
    eps = 1e-9
    rel_gain = (prod["mae"] - cand["mae"]) / prod["mae"]
    rush_ratio = cand["mae_rush"] / prod["mae_rush"] - 1
    rain_ratio = cand["mae_rain"] / prod["mae_rain"] - 1
    criteria = {
        "mae_gain": rel_gain >= min_rel_gain - eps,
        "latency_p95": cand["p95_ms"] <= max_p95_ms + eps,
        "no_rush_regression": rush_ratio <= max_slice_regression + eps,
        "no_rain_regression": rain_ratio <= max_slice_regression + eps,
    }
    details = {
        "rel_mae_gain": round(rel_gain, 4), "min_rel_gain": min_rel_gain,
        "cand_p95_ms": round(cand["p95_ms"], 3), "max_p95_ms": max_p95_ms,
        "rush_rel_change": round(rush_ratio, 4), "rain_rel_change": round(rain_ratio, 4),
        "max_slice_regression": max_slice_regression,
    }
    return {"criteria": criteria, "promote": all(criteria.values()), "details": details}


def _evaluate(model, df, n_latency=200):
    y = df.eta_minutes.to_numpy()
    pred = model.predict(df[eta_data.FEATURES])["eta_minutes"].to_numpy()
    err = np.abs(pred - y)
    rush = df.hour.isin(RUSH_HOURS).to_numpy()
    rain = (df.rain == 1).to_numpy()
    lat = []
    for i in range(n_latency):
        one = df[eta_data.FEATURES].iloc[[i]]
        t0 = time.perf_counter()
        model.predict(one)
        lat.append((time.perf_counter() - t0) * 1000)
    return {"mae": float(err.mean()), "mae_rush": float(err[rush].mean()),
            "mae_rain": float(err[rain].mean()), "p95_ms": float(np.percentile(lat, 95))}


def promotion_gate(**thresholds):
    mlflow.set_tracking_uri(TRACKING_URI)
    fresh = eta_data.make_data(2000, seed=99)
    prod = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@prod")
    cand = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@candidate")
    m_prod, m_cand = _evaluate(prod, fresh), _evaluate(cand, fresh)
    result = decide(m_prod, m_cand, **thresholds)
    result["metrics"] = {"prod": {k: round(v, 3) for k, v in m_prod.items()},
                         "candidate": {k: round(v, 3) for k, v in m_cand.items()}}
    result["thresholds"] = thresholds or "defaults"
    return result


def shadow(prod_url="http://127.0.0.1:5101/invocations", cand_url="http://127.0.0.1:5102/invocations", n=200):
    import requests
    rows = pd.read_csv(ROOT / "data" / "eta_fresh.csv")[eta_data.FEATURES].head(n).to_dict(orient="records")
    s = requests.Session()

    def call(url, r):
        t0 = time.perf_counter()
        resp = s.post(url, json={"dataframe_records": [r]}, timeout=30)
        ms = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        p = resp.json()["predictions"][0]
        return p["eta_minutes"], p["model_version"], ms

    out = {"prod": [], "cand": []}
    for r in rows:
        out["prod"].append(call(prod_url, r))
        out["cand"].append(call(cand_url, r))
    p = np.array([x[0] for x in out["prod"]]); c = np.array([x[0] for x in out["cand"]])
    diff = np.abs(p - c)
    res = {
        "requests": n,
        "prod_version": out["prod"][0][1], "cand_version": out["cand"][0][1],
        "mean_abs_diff_min": round(float(diff.mean()), 2),
        "pct_diff_gt_10min": round(float((diff > 10).mean() * 100), 1),
        "max_abs_diff_min": round(float(diff.max()), 2),
        "prod_p95_ms": round(float(np.percentile([x[2] for x in out["prod"]], 95)), 2),
        "cand_p95_ms": round(float(np.percentile([x[2] for x in out["cand"]], 95)), 2),
        "prod_p50_ms": round(float(np.percentile([x[2] for x in out["prod"]], 50)), 2),
        "cand_p50_ms": round(float(np.percentile([x[2] for x in out["cand"]], 50)), 2),
    }
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["register", "gate", "shadow", "table"])
    ap.add_argument("--min-rel-gain", type=float)
    ap.add_argument("--max-p95-ms", type=float)
    ap.add_argument("--max-slice-regression", type=float)
    a = ap.parse_args()
    if a.cmd == "register":
        register()
    elif a.cmd == "table":
        print(registry_table().to_string(index=False))
    elif a.cmd == "gate":
        th = {k: v for k, v in {"min_rel_gain": a.min_rel_gain, "max_p95_ms": a.max_p95_ms,
                                "max_slice_regression": a.max_slice_regression}.items() if v is not None}
        print(json.dumps(promotion_gate(**th), indent=2))
    elif a.cmd == "shadow":
        print(json.dumps(shadow(), indent=2))
