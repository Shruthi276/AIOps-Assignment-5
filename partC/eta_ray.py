import hashlib
import time
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, Header
from pydantic import BaseModel, Field, model_validator
from ray import serve

ROOT = Path(__file__).resolve().parent.parent
V1_PATH = str(ROOT / "data" / "eta_v1.joblib")
V2_PATH = str(ROOT / "data" / "eta_v2.joblib")
REPLICA_CPUS = 0.25


def bucket(customer_id: str) -> float:
    h = hashlib.sha256(customer_id.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def choose_arm(customer_id: str, weight: float, beta: bool = False) -> str:
    return "v2" if beta or bucket(customer_id) < weight else "v1"


class Instance(BaseModel):
    distance_km: float = Field(ge=0.1, le=500)
    n_stops: int = Field(ge=1, le=60)
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    traffic_index: float = Field(ge=0, le=1)
    rain: int = Field(ge=0, le=1)
    weight_kg: float = Field(gt=0, le=1000)

    @model_validator(mode="after")
    def feasible(self):
        if self.distance_km > 150 and self.n_stops > 20:
            raise ValueError("infeasible delivery: distance_km > 150 together with n_stops > 20")
        return self


class RoutedRequest(BaseModel):
    customer_id: str = Field(min_length=1)
    instances: list[Instance] = Field(min_length=1, max_length=500)


@serve.deployment(ray_actor_options={"num_cpus": REPLICA_CPUS})
class EtaModel:
    def __init__(self, bundle_path: str, extra_work_ms: float = 0.0):
        b = joblib.load(bundle_path)
        self.model, self.features, self.version = b["model"], list(b["feature_columns"]), b["version"]
        self.extra_work_ms = extra_work_ms
        try:
            self.replica = serve.get_replica_context().replica_id.unique_id
        except Exception:
            self.replica = "unknown"

    def predict(self, rows: list[dict]) -> dict:
        X = pd.DataFrame(rows)[self.features]
        preds = [round(float(p), 1) for p in self.model.predict(X)]
        if self.extra_work_ms > 0:
            end = time.perf_counter() + self.extra_work_ms / 1000
            while time.perf_counter() < end:
                pass
        return {"predictions": preds, "model_version": self.version, "served_by": self.replica}


api = FastAPI(title="SwiftParcel ETA canary router")


@serve.deployment(ray_actor_options={"num_cpus": REPLICA_CPUS})
@serve.ingress(api)
class Router:
    def __init__(self, prod, cand, weight: float = 0.1):
        self.arms = {"v1": prod, "v2": cand}
        self.weight = weight

    @api.post("/v1/models/eta:predict")
    async def predict(self, req: RoutedRequest, x_beta: Optional[str] = Header(default=None)):
        arm = choose_arm(req.customer_id, self.weight, beta=(x_beta == "1"))
        rows = [i.model_dump() for i in req.instances]
        out = await self.arms[arm].predict.remote(rows)
        return {**out, "arm": arm, "customer_id": req.customer_id}

    @api.get("/healthz")
    async def healthz(self):
        return {"status": "ok", "weight": self.weight}


def build(weight: float = 0.1, extra_work_ms: float = 0.0):
    prod = EtaModel.options(name="EtaModelProd", num_replicas=2).bind(V1_PATH, extra_work_ms)
    cand = EtaModel.options(
        name="EtaModelCandidate",
        autoscaling_config={
            "min_replicas": 1,
            "max_replicas": 3,
            "initial_replicas": 1,
            "target_ongoing_requests": 2,
            "upscale_delay_s": 5,
            "downscale_delay_s": 60,
        },
    ).bind(V2_PATH, extra_work_ms)
    return Router.bind(prod, cand, weight)
