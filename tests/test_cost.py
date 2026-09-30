from edge_grader.cost import Rates, estimate, fit_rates


def _req(pt, pms, gt=10, gms=1000.0, w=5.0):
    return {"pt": pt, "pms": pms, "gt": gt, "gms": gms, "dev": {"w_mean": w}}


def test_fit_recovers_linear_prefill_and_skips_long_prompts():
    reqs = [_req(pt, 100 + 2 * pt) for pt in (50, 100, 200, 400)] + [_req(5000, 1.0)]
    r = fit_rates("pi", "e2b", reqs, max_prompt=600)
    assert r.n_requests == 4
    assert r.prefill_fixed_ms == 100 and r.prefill_ms_per_token == 2
    assert r.prefill_r2 == 1
    assert r.decode_ms_per_token == 100
    assert r.watts == 5


def test_estimate_uses_each_answers_tokens():
    r = Rates("pi", "e2b", 4, 100.0, 2.0, 1.0, 100.0, 10.0)
    # 100 + 2*200 + 100*2 = 700 ms and 100 + 2*400 + 100*2 = 1100 ms
    est = estimate(r, [{"prompt_tokens": 200, "completion_tokens": 2},
                       {"prompt_tokens": 400, "completion_tokens": 2}])
    assert est["median_latency_s"] == 0.9
    assert est["mean_j_per_answer"] == 9.0
    assert est["wh_per_1000_answers"] == 2.5
