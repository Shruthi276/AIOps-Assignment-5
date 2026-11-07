# SwiftParcel ETA service — Module 4 assignment

This repo serves a delivery-time (ETA) regression model in four ways:

| Part | What it does | Main files |
|---|---|---|
| **A** | FastAPI REST service with validation, request ids and Prometheus metrics, plus load testing | `partA/eta_app.py`, `partA/test_eta_app.py` |
| **B** | MLflow registry (custom pyfunc, aliases `prod`/`candidate`), multi-criteria promotion gate, shadow comparison | `partB/eta_mlflow.py`, `partB/test_gate.py` |
| **C** | Ray Serve app with a **sticky canary** (customer-level hash routing, `X-Beta` override, autoscaling candidate) | `partC/eta_ray.py`, `partC/verify_ray.py` |
| **D** | Release-gate smoke test (version, model-sanity, 422, latency checks) | `partD/smoke_eta.py` |



The models: **v1** = Ridge regression, **v2** = gradient boosting. Both are trained by `eta_data.py` on synthetic data and saved as joblib bundles `{model, feature_columns, version}`. The seven input features are `distance_km, n_stops, hour, weekday, traffic_index, rain, weight_kg`.

## Repository layout

```
eta_data.py, load_test.py, requirements.txt   given starter files (unchanged)
REPORT.md                                     report: tables, answers, machine, not-tested list
partA/  eta_app.py  test_eta_app.py  loadtest_results.json
partB/  eta_mlflow.py  test_gate.py  gate_output.txt  shadow_output.txt
partC/  eta_ray.py  verify_ray.py  verify_output.txt
partD/  smoke_eta.py
```

`data/`, `mlflow.db` and `mlruns/` are generated (not submitted). `eta_data.py` recreates the data and model files.

---

## 1. Requirements

* Linux (tested on Ubuntu in a VirtualBox VM with 4 vCPUs), **2 or more CPU cores**, **at least 4 GB RAM** (Ray needs memory), CPU only
* Python 3.10 or newer, `git`
* Free ports **8000**, **5101** and **5102** (nothing else may be listening on them; see Troubleshooting)

On Ubuntu:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git
```


## 2. Setup

Do these steps once, from a clean checkout.

**1. Clone the repo**

```bash
git clone <your-repo-url> module4
cd module4
```

**2. Create and activate a virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Run `source .venv/bin/activate` again in every new terminal.

**3. Install the dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**4. Generate the data and train the models**

```bash
python eta_data.py
```

This creates `data/eta_train.csv`, `data/eta_fresh.csv`, `data/eta_v1.joblib` and `data/eta_v2.joblib`.

**5. Create the sample file used by the load test**

```bash
python -c "import pandas as pd, eta_data; pd.read_csv('data/eta_fresh.csv')[eta_data.FEATURES].head(200).to_csv('data/eta_sample.csv', index=False)"
```

---

## 3. Part A — REST service

**Tests** (20 tests):

```bash
python -m pytest partA -q
```

**Run the service** (in terminal 1; this serves v1):

```bash
PYTHONPATH=partA uvicorn eta_app:app --port 8000 --log-level warning --no-access-log
# to serve v2 instead:  ETA_MODEL=data/eta_v2.joblib PYTHONPATH=partA uvicorn eta_app:app --port 8000
```

**Try it** (terminal 2):

```bash
python - <<'EOF'
import requests
row = {"distance_km": 12.5, "n_stops": 4, "hour": 8, "weekday": 2, "traffic_index": 0.6, "rain": 1, "weight_kg": 6}
r = requests.post("http://127.0.0.1:8000/v1/models/eta:predict", json={"instances": [row]})
print(r.status_code, r.headers["X-Request-ID"], r.json())
print(requests.get("http://127.0.0.1:8000/healthz").json())
print(requests.get("http://127.0.0.1:8000/metrics").text[:600])
EOF
```

| Endpoint | Behaviour |
|---|---|
| `POST /v1/models/eta:predict` | `{"instances":[...]}` (1–500 rows), returns `predictions` (1 decimal, input order), `model_version`, `request_id` |
| `GET /healthz` | `{"status":"ok","model_version":"v1"}` |
| `GET /metrics` | `eta_requests_total{path,status}` counter and `eta_request_latency_seconds` histogram. Requests to `/metrics` itself are not counted |
| Errors | 422 `{"error":"validation_failed","details":[{"field","msg"}],"request_id"}`. Infeasible deliveries (> 150 km **and** > 20 stops) return 422 with the word "infeasible". Model failures return a generic 500 |

Every response carries `X-Request-ID`: the client's id is echoed back, otherwise a new UUID is generated.

**Load tests** (A3). Start the server with **one worker** (as above), wait until `/healthz` answers, then in a second terminal:

```bash
URL=http://127.0.0.1:8000/v1/models/eta:predict
LT="python load_test.py --url $URL --data data/eta_sample.csv --duration 4"

$LT --concurrency 1 2 4 8 16 --out partA/loadtest_results.json     # A3.1 / A3.2: the table to submit
$LT --concurrency 1 --batch 1   --out partA/loadtest_batch1.json   # A3.3: cost per row, batch 1
$LT --concurrency 1 --batch 100 --out partA/loadtest_batch100.json # A3.3: cost per row, batch 100
```

**The change (A3.4): 2 worker processes.** Stop the server (Ctrl+C), start it again with two workers, and repeat the first load test:

```bash
PYTHONPATH=partA uvicorn eta_app:app --port 8000 --workers 2 --log-level warning --no-access-log
# in the second terminal:
$LT --concurrency 1 2 4 8 16 --out partA/loadtest_results_2workers.json
```

**Reading the results.** Each JSON file has one entry per concurrency level (`rps`, `p50_ms`, `p95_ms`, `p99_ms`, `errors`). This prints a table with Little's Law (`L = rps × p50`) and ms/row (`1000 / (rps × batch)`):

```bash
python - partA/loadtest_results.json <<'EOF'
import json, sys
BATCH = 1                                   # use 100 for loadtest_batch100.json
for r in json.load(open(sys.argv[1])):
    print(r["concurrency"], r["rps"], r["p50_ms"], r["p95_ms"], r["p99_ms"], r["errors"],
          "L=%.2f" % (r["rps"] * r["p50_ms"] / 1000), "ms/row=%.3f" % (1000 / (r["rps"] * BATCH)))
EOF
```

The SLO is p95 ≤ 80 ms. Capacity is the highest `rps` among the levels that meet it.

---

## 4. Part B — MLflow registry, gate and shadow

```bash
export MLFLOW_TRACKING_URI=sqlite:///$PWD/mlflow.db       # the SQLite tracking store

python -m pytest partB -q                                  # 10 tests of decide()

rm -f mlflow.db; rm -rf mlruns                             # start from a fresh registry
python partB/eta_mlflow.py register                        # eta-model v1 (alias prod) and v2 (alias candidate)
python partB/eta_mlflow.py table                           # the registry table (version, alias)

{
  echo "### promotion_gate() with default thresholds"
  python partB/eta_mlflow.py gate
  echo
  echo "### promotion_gate(min_rel_gain=0.60)"
  python partB/eta_mlflow.py gate --min-rel-gain 0.60
} | tee partB/gate_output.txt
```

**Shadow comparison (B3).** Start the two servers in two separate terminals (activate the venv and `export MLFLOW_TRACKING_URI=...` in each):

```bash
mlflow models serve -m "models:/eta-model@prod"      -p 5101 --env-manager local
mlflow models serve -m "models:/eta-model@candidate" -p 5102 --env-manager local
```

When both answer on `http://127.0.0.1:5101/ping` and `:5102/ping`, run in a third terminal:

```bash
python partB/eta_mlflow.py shadow | tee partB/shadow_output.txt
```

Then stop both servers with Ctrl+C.

---

## 5. Part C — Ray Serve sticky canary

Make sure port 8000 is free first (stop the Part A server).

```bash
python partC/verify_ray.py      # starts Ray (2 CPUs), deploys, runs checks (a)-(e), writes partC/verify_output.txt
```

To keep the app running for manual requests:

```bash
python -c "import sys; sys.path.insert(0,'partC'); import ray, eta_ray; from ray import serve; \
ray.init(num_cpus=2, runtime_env={'working_dir':'partC'}); serve.run(eta_ray.build(0.1), route_prefix='/'); input('Enter to stop')"
```

Then, from a second terminal:

```bash
python - <<'EOF'
import requests
row = {"distance_km": 12.5, "n_stops": 4, "hour": 8, "weekday": 2, "traffic_index": 0.6, "rain": 1, "weight_kg": 6}
u = "http://127.0.0.1:8000/v1/models/eta:predict"
for c in ["alice", "bob", "carol"]:
    print(c, requests.post(u, json={"customer_id": c, "instances": [row]}).json())
print("beta", requests.post(u, json={"customer_id": "alice", "instances": [row]}, headers={"X-Beta": "1"}).json())
EOF
```

---

## 6. Part D — smoke test (release gate)

The smoke test needs a server on port 8000. Run it three times (each time start the server, run the test, stop the server):

```bash
# 1. v1 server, expect v1 -> should PASS (exit 0)
PYTHONPATH=partA uvicorn eta_app:app --port 8000 --log-level warning --no-access-log &
sleep 5
python partD/smoke_eta.py --url http://127.0.0.1:8000 --expect-version v1 --max-p95-ms 80; echo "exit=$?"

# 2. same v1 server, expect v2 -> should FAIL (exit 1): the gate blocks a wrong version
python partD/smoke_eta.py --url http://127.0.0.1:8000 --expect-version v2 --max-p95-ms 80; echo "exit=$?"
kill %1; sleep 2

# 3. v2 server, expect v2 -> should PASS (exit 0)
ETA_MODEL=data/eta_v2.joblib PYTHONPATH=partA uvicorn eta_app:app --port 8000 --log-level warning --no-access-log &
sleep 5
python partD/smoke_eta.py --url http://127.0.0.1:8000 --expect-version v2 --max-p95-ms 80; echo "exit=$?"
kill %1
```

Against the Ray app, add `--beta` to smoke-test the candidate: `--expect-version v2 --beta`.

---



## AI Disclosure (Fair Use of AI)

**Tool used:** Claude

**How it was used:**

- **Debugging:** I pasted an error and asked what it meant. Examples: an MLflow signature or pip-requirements error,  Ray hangs or is killed (OOM) and a Pydantic validation message.
- **Concept explanations:** I asked why Python's `hash()` is randomly salted per process, how Little's Law relates throughput and latency, and what an MLflow alias is.
- **Boilerplate:** I used it for things like the FastAPI middleware skeleton and a `Dockerfile`-style setup, which I then edited myself.
- **Syntax help:** I asked how to write a pytest `monkeypatch` and a Pydantic `model_validator`.


**Impact:** Claude helped me understand errors and concepts, gave me starting skeletons and syntax, and saved time on the narration wording. I ran all the measurements and tests, and I checked the AI-assisted parts against my own results before using them.
