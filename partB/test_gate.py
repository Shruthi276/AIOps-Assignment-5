import copy

from eta_mlflow import decide

PROD = {"mae": 10.0, "mae_rush": 12.0, "mae_rain": 11.0, "p95_ms": 2.0}


def cand(**kw):
    c = {"mae": 8.0, "mae_rush": 10.0, "mae_rain": 9.0, "p95_ms": 3.0}
    c.update(kw)
    return c


def test_clear_improvement_promotes():
    out = decide(PROD, cand())
    assert out["promote"] is True
    assert all(out["criteria"].values())


def test_mae_gain_just_inside_threshold():
    assert decide(PROD, cand(mae=9.70))["criteria"]["mae_gain"] is True


def test_mae_gain_just_outside_threshold():
    out = decide(PROD, cand(mae=9.71))
    assert out["criteria"]["mae_gain"] is False and out["promote"] is False


def test_candidate_worse_overall_is_rejected():
    assert decide(PROD, cand(mae=10.5))["promote"] is False


def test_latency_just_inside_and_outside():
    assert decide(PROD, cand(p95_ms=10.0))["criteria"]["latency_p95"] is True
    out = decide(PROD, cand(p95_ms=10.01))
    assert out["criteria"]["latency_p95"] is False and out["promote"] is False


def test_rush_slice_regression_boundary():
    assert decide(PROD, cand(mae_rush=12.6))["criteria"]["no_rush_regression"] is True
    out = decide(PROD, cand(mae_rush=12.61))
    assert out["criteria"]["no_rush_regression"] is False and out["promote"] is False


def test_rain_slice_regression_blocks_even_with_big_overall_gain():
    out = decide(PROD, cand(mae=5.0, mae_rain=12.1))
    assert out["criteria"]["mae_gain"] is True
    assert out["criteria"]["no_rain_regression"] is False
    assert out["promote"] is False


def test_custom_thresholds_are_respected():
    c = cand(mae=8.0)
    assert decide(PROD, c, min_rel_gain=0.60)["promote"] is False
    assert decide(PROD, c, min_rel_gain=0.20)["promote"] is True
    assert decide(PROD, cand(p95_ms=15.0), max_p95_ms=20.0)["promote"] is True
    assert decide(PROD, cand(mae_rush=13.0), max_slice_regression=0.10)["promote"] is True


def test_decide_is_pure():
    p, c = copy.deepcopy(PROD), cand()
    first = decide(p, c)
    assert p == PROD and c == cand()
    assert decide(p, c) == first


def test_output_shape():
    out = decide(PROD, cand())
    assert set(out) >= {"criteria", "promote"}
    assert set(out["criteria"]) == {"mae_gain", "latency_p95", "no_rush_regression", "no_rain_regression"}
    assert isinstance(out["promote"], bool)
