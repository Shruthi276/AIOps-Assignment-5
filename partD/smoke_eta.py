import argparse
import sys
import time

import requests

PREDICT = "/v1/models/eta:predict"
BASE_ROW = {"distance_km": 20.0, "n_stops": 5, "hour": 13, "weekday": 2,
            "traffic_index": 0.3, "rain": 0, "weight_kg": 5.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--expect-version", required=True)
    ap.add_argument("--max-p95-ms", type=float, default=80.0)
    ap.add_argument("--beta", action="store_true", help="send X-Beta: 1 (Ray router: force candidate)")
    ap.add_argument("--n-latency", type=int, default=50)
    a = ap.parse_args()

    base = a.url.split(PREDICT)[0].rstrip("/")
    url = base + PREDICT
    s = requests.Session()
    if a.beta:
        s.headers["X-Beta"] = "1"
    results = []

    def post(rows):
        return s.post(url, json={"customer_id": "smoke-test", "instances": rows}, timeout=10)

    def eta(**kw):
        r = post([{**BASE_ROW, **kw}])
        r.raise_for_status()
        return r.json()["predictions"][0]

    def check(name, fn):
        try:
            ok, info = fn()
        except Exception as e:
            ok, info = False, f"{type(e).__name__}: {e}"
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {info}")

    def c_version():
        r = post([BASE_ROW])
        j = r.json()
        p = j["predictions"][0]
        ok = r.status_code == 200 and j.get("model_version") == a.expect_version and 0 < p < 1000
        return ok, f"status={r.status_code} model_version={j.get('model_version')} eta={p}"
    check("expected version + plausible ETA", c_version)

    def c_batch():
        rows = [{**BASE_ROW, "distance_km": d} for d in (5, 60, 15)]
        batch = post(rows).json()["predictions"]
        singles = [eta(distance_km=d) for d in (5, 60, 15)]
        return batch == singles, f"batch={batch} singles={singles}"
    check("batch size and order", c_batch)

    def c_distance():
        e = [eta(distance_km=d) for d in (5, 25, 60, 120)]
        return all(x <= y for x, y in zip(e, e[1:])), f"distance 5/25/60/120 km -> {e}"
    check("sanity: ETA non-decreasing in distance", c_distance)

    def c_traffic():
        e = [eta(traffic_index=t) for t in (0.0, 0.5, 1.0)]
        return all(x <= y for x, y in zip(e, e[1:])), f"traffic 0.0/0.5/1.0 -> {e}"
    check("sanity: ETA non-decreasing in traffic", c_traffic)

    def c_stops():
        e = [eta(n_stops=n) for n in (1, 8, 20)]
        return all(x <= y for x, y in zip(e, e[1:])), f"stops 1/8/20 -> {e}"
    check("sanity: ETA non-decreasing in stops", c_stops)

    def c_invalid():
        codes = [post([{**BASE_ROW, "hour": 25}]).status_code,
                 post([{**BASE_ROW, "distance_km": 200, "n_stops": 30}]).status_code,
                 post([]).status_code]
        return codes == [422, 422, 422], f"hour=25 / infeasible / empty batch -> {codes}"
    check("invalid input returns 422", c_invalid)

    def c_latency():
        lat = []
        for _ in range(a.n_latency):
            t0 = time.perf_counter()
            post([BASE_ROW]).raise_for_status()
            lat.append((time.perf_counter() - t0) * 1000)
        lat.sort()
        p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
        return p95 <= a.max_p95_ms, f"p95={p95:.1f} ms over {a.n_latency} requests (limit {a.max_p95_ms} ms)"
    check("latency p95 within limit", c_latency)

    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed -> {'RELEASE OK' if all(results) else 'BLOCK RELEASE'}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
