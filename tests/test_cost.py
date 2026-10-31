import json
from importlib import resources
from pathlib import Path

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
    assert cost.cost_gate([1.0, 1.01, 0.99], [1.2, 1.21, 1.19], 5).result == "FAIL"
    assert cost.cost_gate([1.0, 1.01, 0.99], [1.01, 1.0, 1.02], 5).result == "PASS"
    assert cost.cost_gate([1.0, 1.01, 0.99], [0.8, 1.3, 1.0], 5).result == "INCONCLUSIVE"


FIX = Path(__file__).parent / "fixtures" / "cost_compare"


@pytest.mark.parametrize(
    ("candidate", "code", "verdict"),
    [
        ("candidate_same", 0, "PASS"),
        ("candidate_20pct_slower", 1, "FAIL"),
        ("candidate_noisy", 2, "INCONCLUSIVE"),
    ],
)
def test_cost_compare_cli(candidate, code, verdict):
    args = [
        "cost-compare",
        "--baseline",
        str(FIX / "baseline"),
        "--candidate",
        str(FIX / candidate),
        "--gpu",
        "H100",
        "--json",
    ]
    res = CliRunner().invoke(app, args)
    assert res.exit_code == code, res.output
    data = json.loads(res.output)
    assert data["result"] == verdict and data["runs"] == [5, 5]
    assert data["hourly_usd"] == pytest.approx(3.99)
    if verdict == "FAIL":
        assert data["change_percent"] == pytest.approx(20, abs=0.5)


def test_cost_compare_needs_repetitions(tmp_path):
    res = CliRunner().invoke(
        app, ["cost-compare", "--baseline", str(tmp_path), "--candidate", str(tmp_path)]
    )
    assert res.exit_code == 3


def test_cli_demo():
    res = CliRunner().invoke(app, ["cost", "--demo"])
    assert res.exit_code == 0, res.output
    assert "Ranked fixes" in res.output and "/GPU-h" in res.output


REAL = Path(__file__).parent / "fixtures" / "real_vllm_0_32"


def test_real_vllm_0_32_scrape():
    snaps = [cm.snapshot_from_text((REAL / f"t{i}.prom").read_text()) for i in (0, 1)]
    w = cm.window(snaps[0], snaps[1], 60)
    assert w.engine == "vllm"
    assert w.prompt_tps == pytest.approx((1199128 - 376000) / 60)
    assert w.gen_tps == pytest.approx((368976 - 114576) / 60)
    assert snaps[1].prefix_caching_enabled is True
    assert 0.8 < w.prefix_hit_rate < 0.9


def test_real_scrape_unlisted_gpu_with_price():
    args = [
        "cost",
        "--metrics",
        str(REAL / "t0.prom"),
        "--metrics",
        str(REAL / "t1.prom"),
        "--seconds",
        "60",
        "--gpu",
        "RTX A5000",
        "--price",
        "0.27",
        "--json",
    ]
    res = CliRunner().invoke(app, args)
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert not any(f["title"].startswith(("Switch GPU", "Quantize")) for f in data["fixes"])


def test_metrics_files_fall_back_to_mtime(tmp_path):
    import os

    a, b = tmp_path / "a.prom", tmp_path / "b.prom"
    a.write_text((REAL / "t0.prom").read_text())
    b.write_text((REAL / "t1.prom").read_text())
    os.utime(a, (1000, 1000))
    os.utime(b, (1060, 1060))
    res = CliRunner().invoke(
        app, ["cost", "--metrics", str(a), "--metrics", str(b), "--gpu", "L4", "--json"]
    )
    assert res.exit_code == 0, res.output
