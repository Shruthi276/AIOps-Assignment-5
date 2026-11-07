import os
import threading
import time
import uuid
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parent.parent


def _resolve(path: str) -> str:
    p = Path(path)
    if p.exists() or p.is_absolute():
        return str(p)
    return str(ROOT / p)


MODEL_PATH = _resolve(os.environ.get("ETA_MODEL", "data/eta_v1.joblib"))
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = list(bundle["feature_columns"])
MODEL_VERSION = bundle["version"]


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


class PredictRequest(BaseModel):
    instances: list[Instance] = Field(min_length=1, max_length=500)


BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
_lock = threading.Lock()
_requests = {}
_bucket_counts = [0] * len(BUCKETS)
_lat_sum = 0.0
_lat_count = 0


def _observe(path: str, status: int, seconds: float) -> None:
    global _lat_sum, _lat_count
    with _lock:
        _requests[(path, status)] = _requests.get((path, status), 0) + 1
        for i, b in enumerate(BUCKETS):
            if seconds <= b:
                _bucket_counts[i] += 1
        _lat_sum += seconds
        _lat_count += 1


def _render_metrics() -> str:
    with _lock:
        lines = ["# HELP eta_requests_total Total HTTP requests by path and status.",
                 "# TYPE eta_requests_total counter"]
        for (path, status), n in sorted(_requests.items()):
            lines.append(f'eta_requests_total{{path="{path}",status="{status}"}} {n}')
        lines += ["# HELP eta_request_latency_seconds Request latency in seconds.",
                  "# TYPE eta_request_latency_seconds histogram"]
        for b, n in zip(BUCKETS, _bucket_counts):
            lines.append(f'eta_request_latency_seconds_bucket{{le="{b}"}} {n}')
        lines.append(f'eta_request_latency_seconds_bucket{{le="+Inf"}} {_lat_count}')
        lines.append(f"eta_request_latency_seconds_sum {_lat_sum:.6f}")
        lines.append(f"eta_request_latency_seconds_count {_lat_count}")
    return "\n".join(lines) + "\n"


app = FastAPI(title="SwiftParcel ETA service")


@app.middleware("http")
async def request_id_and_metrics(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = rid
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        response = JSONResponse({"error": "internal_error", "request_id": rid}, status_code=500)
    response.headers["X-Request-ID"] = rid
    if request.url.path != "/metrics":
        _observe(request.url.path, response.status_code, time.perf_counter() - t0)
    return response


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    details = []
    for err in exc.errors():
        loc = [str(x) for x in err.get("loc", ()) if x != "body"]
        details.append({"field": ".".join(loc) or "body", "msg": err.get("msg", "invalid")})
    return JSONResponse({"error": "validation_failed", "details": details,
                         "request_id": request.state.request_id}, status_code=422)


@app.post("/v1/models/eta:predict")
def predict(req: PredictRequest, request: Request):
    rid = request.state.request_id
    try:
        X = pd.DataFrame([inst.model_dump() for inst in req.instances])[FEATURES]
        preds = [round(float(p), 1) for p in model.predict(X)]
    except Exception:
        return JSONResponse({"error": "internal_error", "request_id": rid}, status_code=500)
    return {"predictions": preds, "model_version": MODEL_VERSION, "request_id": rid}


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.get("/metrics")
def metrics():
    return PlainTextResponse(_render_metrics(), media_type="text/plain; version=0.0.4")
