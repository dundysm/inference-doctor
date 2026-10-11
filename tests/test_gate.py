"""The interval-based regression gate: its statistics, its decisions, and its error rates."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

from inference_doctor.quality import QualityThresholds, decide_regression
from inference_doctor.stats import t_cdf, t_quantile, welch_log_ratio_interval

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from gate_simulation import verdict_rates  # noqa: E402

DEFAULTS = QualityThresholds()


@pytest.mark.parametrize(
    ("p", "df", "expected"),
    [
        (0.975, 1, 12.7062),
        (0.975, 4, 2.7764),
        (0.95, 10, 1.8125),
        (0.90, 3, 1.6377),
        (0.975, 1e6, 1.9600),
    ],
)
def test_t_quantile_matches_published_tables(p, df, expected):
    assert t_quantile(p, df) == pytest.approx(expected, abs=1e-4)


def test_t_quantile_inverts_t_cdf_for_fractional_degrees_of_freedom():
    for df in (1.5, 2.7, 6.8, 30.2):
        for p in (0.6, 0.9, 0.95, 0.995):
            assert t_cdf(t_quantile(p, df), df) == pytest.approx(p, abs=1e-9)
    assert t_quantile(0.05, 5) == pytest.approx(-t_quantile(0.95, 5))


def test_interval_is_symmetric_on_the_log_scale():
    interval = welch_log_ratio_interval([100, 102, 98], [90, 92, 88], 0.90)
    assert interval.ratio == pytest.approx(0.9)
    assert math.log(interval.ratio / interval.low) == pytest.approx(
        math.log(interval.high / interval.ratio)
    )


def test_identical_spreadless_sides_give_a_point_interval():
    interval = welch_log_ratio_interval([100, 100, 100], [90, 90, 90], 0.90)
    assert interval.low == interval.ratio == interval.high == pytest.approx(0.9)
    assert interval.degrees_of_freedom is None


def test_clear_throughput_drop_fails():
    decision = decide_regression(
        [1000, 1010, 990, 1005, 995],
        [800, 810, 790, 805, 795],
        lower_is_better=False,
        thresholds=DEFAULTS,
    )
    assert decision.status == "FAIL"
    assert decision.regression_percent == pytest.approx(20.0)
    assert decision.regression_percent_low >= 10
    assert "entirely at or past the 10% threshold" in decision.reason


def test_small_change_with_tight_interval_passes():
    decision = decide_regression(
        [1000, 1010, 990, 1005, 995],
        [970, 980, 960, 975, 965],
        lower_is_better=False,
        thresholds=DEFAULTS,
    )
    assert decision.status == "PASS"
    assert decision.regression_percent_high < 10


def test_interval_straddling_the_threshold_is_inconclusive():
    decision = decide_regression(
        [1000, 1100, 900],
        [900, 990, 810],
        lower_is_better=False,
        thresholds=DEFAULTS,
    )
    assert decision.status == "INCONCLUSIVE"
    assert decision.regression_percent_low < 10 <= decision.regression_percent_high


def test_latency_rise_is_a_regression_and_latency_drop_is_not():
    worse = decide_regression(
        [100, 101, 99], [130, 131, 129], lower_is_better=True, thresholds=DEFAULTS
    )
    better = decide_regression(
        [100, 101, 99], [70, 71, 69], lower_is_better=True, thresholds=DEFAULTS
    )
    assert worse.status == "FAIL" and worse.regression_percent == pytest.approx(30.0)
    assert better.status == "PASS" and better.regression_percent == pytest.approx(-30.0)


def test_all_zero_sides_pass_and_a_drop_to_zero_fails():
    zeros = decide_regression([0, 0, 0], [0, 0, 0], lower_is_better=True, thresholds=DEFAULTS)
    assert zeros.status == "PASS" and zeros.regression_percent == 0
    collapsed = decide_regression(
        [1000, 1001, 999], [0, 0, 0], lower_is_better=False, thresholds=DEFAULTS
    )
    assert collapsed.status == "FAIL" and collapsed.regression_percent == pytest.approx(100.0)


def test_higher_confidence_widens_the_interval():
    args = ([1000, 1030, 970, 1010, 990], [930, 960, 900, 940, 920])
    narrow = decide_regression(*args, lower_is_better=False, thresholds=DEFAULTS)
    wide = decide_regression(
        *args, lower_is_better=False, thresholds=QualityThresholds(confidence=0.99)
    )
    assert wide.regression_percent_low < narrow.regression_percent_low
    assert wide.regression_percent_high > narrow.regression_percent_high


# Error rates. These rows are the claims in docs/gate-calibration.md, checked at a
# smaller trial count with margins for sampling error.
TRIALS = 1500


def test_no_change_rarely_fails_and_usually_passes_with_five_repetitions():
    rates = verdict_rates(5, 4.0, 0.0, trials=TRIALS)
    assert rates["FAIL"] < 1.0
    assert rates["PASS"] > 95.0


def test_large_regression_is_caught_with_five_repetitions():
    rates = verdict_rates(5, 4.0, 20.0, trials=TRIALS)
    assert rates["PASS"] < 1.0
    assert rates["FAIL"] > 97.0


def test_at_the_threshold_each_wrong_verdict_is_bounded_by_the_confidence_level():
    # A 90% two-sided interval puts about 5% in each tail.
    rates = verdict_rates(5, 4.0, 10.0, trials=TRIALS)
    assert rates["PASS"] < 7.5
    assert rates["FAIL"] < 7.5


def test_noise_below_the_threshold_never_turns_into_frequent_false_alarms():
    for repetitions, cv in ((3, 8.0), (10, 8.0)):
        rates = verdict_rates(repetitions, cv, 5.0, trials=TRIALS)
        assert rates["FAIL"] < 2.0
