import concurrent.futures as cf
import sys
import time
from pathlib import Path

import ray
import requests
from ray import serve

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eta_ray

URL = "http://127.0.0.1:8000/v1/models/eta:predict"
APP = "eta"
N_CUSTOMERS = 1000
ROW = {"distance_km": 18.0, "n_stops": 5, "hour": 8, "weekday": 1,
       "traffic_index": 0.6, "rain": 1, "weight_kg": 4.0}

_log = open(HERE / "verify_output.txt", "w")


def say(msg=""):
    print(msg)
    _log.write(msg + "\n")
    _log.flush()


def call(cid, beta=False):
    headers = {"X-Beta": "1"} if beta else {}
    r = requests.post(URL, json={"customer_id": cid, "instances": [ROW]}, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def arms_for(ids, beta=False):
    with cf.ThreadPoolExecutor(8) as ex:
        return dict(zip(ids, (j["arm"] for j in ex.map(lambda c: call(c, beta), ids))))


def deploy(weight):
    t0 = time.time()
    serve.run(eta_ray.build(weight=weight), name=APP, route_prefix="/",
              logging_config={"enable_access_log": False, "log_level": "WARNING"})
    say(f"  deployed app '{APP}' with weight={weight} in {time.time() - t0:.1f}s")


def check(name, cond, evidence):
    say(f"[{'PASS' if cond else 'FAIL'}] {name}: {evidence}")
    assert cond, name


def main():
    ray.init(num_cpus=2, runtime_env={"working_dir": str(HERE)}, ignore_reinit_error=True)
    say(f"Ray cluster resources: {ray.cluster_resources()}")
    customers = [f"cust-{i:04d}" for i in range(N_CUSTOMERS)]

    say("\n== deploy weight=0.2 ==")
    deploy(0.2)
    try:
        for name, d in serve.status().applications[APP].deployments.items():
            say(f"  deployment {name}: {d.status} replicas={dict(d.replica_states)}")
    except Exception as e:
        say(f"  (could not read serve.status(): {e})")

    first = arms_for(customers)
    share = sum(a == "v2" for a in first.values()) / N_CUSTOMERS
    check("(a) candidate share within 0.20 +/- 0.04", abs(share - 0.2) <= 0.04,
          f"{share:.3f} of {N_CUSTOMERS} distinct customers routed to v2")

    sample = customers[:200]
    second = arms_for(sample)
    changed = [c for c in sample if second[c] != first[c]]
    check("(b) 200 customers sticky on 2nd call", not changed,
          f"{len(changed)} of 200 changed arm (v2 among them: {sum(first[c] == 'v2' for c in sample)})")

    say("\n== delete + redeploy identical app (weight=0.2) ==")
    serve.delete(APP)
    deploy(0.2)
    third = arms_for(sample)
    changed = [c for c in sample if third[c] != first[c]]
    check("(c) 200 customers sticky after redeploy", not changed, f"{len(changed)} of 200 changed arm")

    say("\n== update weight 0.2 -> 0.5 ==")
    deploy(0.5)
    after = arms_for(customers)
    was_v2 = [c for c in customers if first[c] == "v2"]
    moved_back = [c for c in was_v2 if after[c] == "v1"]
    new_share = sum(a == "v2" for a in after.values()) / N_CUSTOMERS
    moved_up = sum(first[c] == "v1" and after[c] == "v2" for c in customers)
    check("(d) no candidate customer moves back to prod", not moved_back,
          f"{len(was_v2)} were on v2, {len(moved_back)} moved back; "
          f"{moved_up} moved v1->v2; new share {new_share:.3f}")

    prod_customers = [c for c in customers if after[c] == "v1"][:50]
    forced = arms_for(prod_customers, beta=True)
    versions = {call(c, beta=True)["model_version"] for c in prod_customers[:5]}
    n_v2 = sum(a == "v2" for a in forced.values())
    check("(e) X-Beta header forces candidate", n_v2 == 50 and versions == {"v2"},
          f"{n_v2}/50 prod-bucket customers served by v2 with X-Beta: 1; model_version={versions}")
    unforced = arms_for(prod_customers[:10])
    check("(e') without header they return to prod", all(a == "v1" for a in unforced.values()),
          f"{sum(a == 'v1' for a in unforced.values())}/10 back on v1")

    say("\nALL CHECKS PASSED")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        say(f"\nVERIFICATION FAILED: {e}")
        sys.exit(1)
    finally:
        try:
            serve.shutdown()
        finally:
            ray.shutdown()
            _log.close()
