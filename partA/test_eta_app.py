import re
import uuid

import pytest
from fastapi.testclient import TestClient

import eta_app

URL = "/v1/models/eta:predict"


def row(**overrides):
    r = {"distance_km": 12.5, "n_stops": 4, "hour": 14, "weekday": 2,
         "traffic_index": 0.4, "rain": 0, "weight_kg": 6.0}
    r.update(overrides)
    return r


@pytest.fixture
def client():
    return TestClient(eta_app.app)


def test_valid_single_request(client):
    r = client.post(URL, json={"instances": [row()]})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"predictions", "model_version", "request_id"}
    assert len(body["predictions"]) == 1
    p = body["predictions"][0]
    assert isinstance(p, float) and p > 0
    assert round(p, 1) == p
    assert body["model_version"] == eta_app.MODEL_VERSION


def test_batch_order_preserved(client):
    rows = [row(distance_km=d) for d in (2, 150, 30, 80, 5)]
    batch = client.post(URL, json={"instances": rows}).json()["predictions"]
    singles = [client.post(URL, json={"instances": [r]}).json()["predictions"][0] for r in rows]
    assert batch == singles
    assert batch[1] > batch[0]


def test_out_of_range_value(client):
    r = client.post(URL, json={"instances": [row(hour=24)]})
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "validation_failed"
    assert any("hour" in d["field"] for d in body["details"])
    assert "request_id" in body


def test_missing_field(client):
    bad = row()
    del bad["weight_kg"]
    r = client.post(URL, json={"instances": [bad]})
    assert r.status_code == 422
    assert any("weight_kg" in d["field"] for d in r.json()["details"])


def test_wrong_type(client):
    r = client.post(URL, json={"instances": [row(n_stops="many")]})
    assert r.status_code == 422
    assert any("n_stops" in d["field"] for d in r.json()["details"])


@pytest.mark.parametrize("field,value", [("rain", 2), ("weight_kg", 0), ("traffic_index", 1.5),
                                         ("distance_km", 0.05), ("weekday", 7)])
def test_other_range_limits(client, field, value):
    assert client.post(URL, json={"instances": [row(**{field: value})]}).status_code == 422


def test_infeasible_cross_field(client):
    r = client.post(URL, json={"instances": [row(distance_km=151, n_stops=21)]})
    assert r.status_code == 422
    assert "infeasible" in r.text


def test_cross_field_boundary_is_feasible(client):
    assert client.post(URL, json={"instances": [row(distance_km=150, n_stops=20)]}).status_code == 200


def test_empty_batch_rejected(client):
    assert client.post(URL, json={"instances": []}).status_code == 422


def test_batch_of_500_ok_and_501_rejected(client):
    ok = client.post(URL, json={"instances": [row()] * 500})
    assert ok.status_code == 200 and len(ok.json()["predictions"]) == 500
    assert client.post(URL, json={"instances": [row()] * 501}).status_code == 422


def test_request_id_echoed(client):
    r = client.post(URL, json={"instances": [row()]}, headers={"X-Request-ID": "abc-123"})
    assert r.headers["X-Request-ID"] == "abc-123"
    assert r.json()["request_id"] == "abc-123"


def test_request_id_generated_on_success_and_error(client):
    ok = client.post(URL, json={"instances": [row()]})
    uuid.UUID(ok.headers["X-Request-ID"])
    assert ok.json()["request_id"] == ok.headers["X-Request-ID"]

    err = client.post(URL, json={"instances": [row(hour=99)]})
    assert err.status_code == 422
    uuid.UUID(err.headers["X-Request-ID"])
    assert err.json()["request_id"] == err.headers["X-Request-ID"]


def test_request_id_echoed_on_error(client):
    r = client.post(URL, json={"instances": []}, headers={"X-Request-ID": "err-42"})
    assert r.status_code == 422
    assert r.headers["X-Request-ID"] == "err-42" and r.json()["request_id"] == "err-42"


def _counter(text, path, status):
    m = re.search(rf'eta_requests_total\{{path="{re.escape(path)}",status="{status}"\}} (\d+)', text)
    return int(m.group(1)) if m else 0


def test_metrics_output(client):
    before = client.get("/metrics").text
    client.post(URL, json={"instances": [row()]})
    client.post(URL, json={"instances": [row(hour=99)]})
    text = client.get("/metrics").text
    client.get("/metrics")
    after = client.get("/metrics").text

    assert _counter(text, URL, 200) == _counter(before, URL, 200) + 1
    assert _counter(text, URL, 422) == _counter(before, URL, 422) + 1
    assert 'path="/metrics"' not in after
    assert len(re.findall(r'eta_request_latency_seconds_bucket\{le="[^"]+"\}', text)) >= 3
    assert 'eta_request_latency_seconds_bucket{le="+Inf"}' in text
    assert "eta_request_latency_seconds_sum" in text
    assert "eta_request_latency_seconds_count" in text


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": eta_app.MODEL_VERSION}


def test_generic_500_when_model_raises(client, monkeypatch):
    class Broken:
        def predict(self, X):
            raise RuntimeError("secret internal detail: /opt/models/weights.bin")

    monkeypatch.setattr(eta_app, "model", Broken())
    r = client.post(URL, json={"instances": [row()]}, headers={"X-Request-ID": "boom-1"})
    assert r.status_code == 500
    assert "secret" not in r.text and "RuntimeError" not in r.text and "Traceback" not in r.text
    assert r.headers["X-Request-ID"] == "boom-1"
    assert r.json()["request_id"] == "boom-1"
