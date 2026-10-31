from importlib import resources

import pytest
from typer.testing import CliRunner

from inference_doctor import cost
from inference_doctor import cost_metrics as cm
from inference_doctor.cli import app


def _demo():
    base = resources.files("inference_doctor").joinpath("data", "cost_demo")
    return [
        cm.snapshot_from_text(base.joinpath(n).read_text())
        for n in ("vllm-t0.prom", "vllm-t1.prom")
    ]


def test_parse_vllm_window():
    a, b = _demo()
    assert a.engine == "vllm" and a.prefix_caching_enabled is False and a.cache_dtype == "auto"
    w = cm.window(a, b, None)
    assert w.seconds == 60
    assert w.prompt_tps == pytest.approx(5200) and w.gen_tps == pytest.approx(1300)
    assert w.prefix_hit_rate == 0


def test_parse_sglang():
    s = cm.snapshot_from_text(
        "sglang:prompt_tokens_total 100\nsglang:generation_tokens_total 50\n"
        "sglang:num_running_reqs 4\nsglang:num_queue_reqs 2\nsglang:token_usage 0.5\n"
        "sglang:cache_hit_rate 0.3\n"
    )
    assert (
        s.engine == "sglang" and s.waiting == 2 and s.kv_usage == 0.5 and s.prefix_hit_rate == 0.3
    )


def test_single_snapshot_needs_seconds():
    a, _ = _demo()
    with pytest.raises(ValueError):
        cm.window(a, None, None)


def test_cost_math():
    u = cost.Usage("t", prompt_tps=0, gen_tps=1000)
    r = cost.analyze(u, "H100-SXM", 1, price=3.6)
    assert r.cost_per_1m_output_tokens == pytest.approx(1.0)
    assert r.monthly_cost == pytest.approx(3.6 * 730)


def test_rightsize_and_waste():
    u = cost.Usage("t", 1000, 1000, running=10, waiting=0, kv_usage=0.2, cache_dtype="fp8")
    r = cost.analyze(u, "H100-SXM", 8, price=4.0)
    assert r.occupancy == pytest.approx(0.2)
    assert r.monthly_waste == pytest.approx(8 * 4 * 730 * 0.75)
    titles = [f.title for f in r.fixes]
    assert "Right-size: 8 -> 2 GPUs" in titles
    assert not any("FP8" in t for t in titles)


def test_queue_means_no_waste_and_batch_fix():
    u = cost.Usage("t", 1000, 1000, running=256, waiting=40, kv_usage=0.3)
    r = cost.analyze(u, "H100-SXM", 1, price=4.0)
    assert r.monthly_waste == 0
    assert any("max-num-seqs" in f.title for f in r.fixes)
    assert not any("Right-size" in f.title for f in r.fixes)


def test_fixes_ranked_and_api():
    a, b = _demo()
    r = cost.analyze(
        cost.usage_from_window(cm.window(a, b, None)),
        "H100-SXM",
        8,
        replicas=4,
        tp=2,
        api_model="gpt-oss-120b",
    )
    s = [f.monthly_savings for f in r.fixes]
    assert s == sorted(s, reverse=True)
    assert any("hosted API" in f.title for f in r.fixes)


def test_price_table_is_sourced():
    t = cost.gpu_table()
    for row in t["gpus"].values():
        for provider in row["prices"]:
            assert t["sources"][provider]["url"].startswith("https://")
    for m in cost.api_table()["models"].values():
        assert m["provider"] in cost.api_table()["sources"]


def test_gate():
    assert cost.cost_gate([1.0, 1.0, 1.0], [1.2, 1.2, 1.2], 5)[0] == "FAIL"
    assert cost.cost_gate([1.0, 1.0, 1.0], [1.01, 1.0, 1.02], 5)[0] == "PASS"


def test_cli_demo():
    res = CliRunner().invoke(app, ["cost", "--demo"])
    assert res.exit_code == 0, res.output
    assert "Ranked fixes" in res.output and "/GPU-h" in res.output
