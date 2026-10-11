"""Small-sample statistics for the regression gate, with no third-party dependencies.

The gate compares the mean of repeated candidate runs with the mean of repeated
baseline runs. It works on the log of the ratio of means, so the interval is for a
relative change and is symmetric between "10% slower" and "10% faster" after
transforming back. The standard error comes from the delta method and the degrees
of freedom from Welch-Satterthwaite, so the two sides may have different spreads
and different numbers of repetitions.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    # Lentz's method for the continued fraction of the incomplete beta function.
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(a * math.log(x) + b * math.log1p(-x) - _log_beta(a, b))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    """Cumulative distribution function of Student's t with `df` degrees of freedom."""
    if df <= 0:
        raise ValueError("degrees of freedom must be positive")
    tail = 0.5 * regularized_incomplete_beta(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t >= 0 else tail


def t_quantile(p: float, df: float) -> float:
    """Inverse of `t_cdf`: the t value with probability `p` below it."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be strictly between 0 and 1")
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -t_quantile(1.0 - p, df)
    low, high = 0.0, 1.0
    while t_cdf(high, df) < p:
        high *= 2.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if t_cdf(mid, df) < p:
            low = mid
        else:
            high = mid
        if high - low < 1e-12 * max(1.0, high):
            break
    return (low + high) / 2.0


@dataclass(frozen=True)
class RatioInterval:
    """Confidence interval for candidate_mean / baseline_mean."""

    ratio: float
    low: float
    high: float
    confidence: float
    degrees_of_freedom: float | None  # None when both sides have zero spread


def welch_log_ratio_interval(
    baseline: Sequence[float],
    candidate: Sequence[float],
    confidence: float,
) -> RatioInterval:
    """Two-sided interval for the ratio of means, computed on the log scale.

    Both sides need at least two values and a positive mean.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    if len(baseline) < 2 or len(candidate) < 2:
        raise ValueError("each side needs at least two values")
    baseline_mean = statistics.fmean(baseline)
    candidate_mean = statistics.fmean(candidate)
    if baseline_mean <= 0 or candidate_mean <= 0:
        raise ValueError("both means must be positive to form a ratio")

    # Delta method: Var(log mean) ~= Var(x) / (n * mean^2).
    baseline_term = statistics.variance(baseline) / (len(baseline) * baseline_mean**2)
    candidate_term = statistics.variance(candidate) / (len(candidate) * candidate_mean**2)
    standard_error = math.sqrt(baseline_term + candidate_term)
    log_ratio = math.log(candidate_mean / baseline_mean)
    ratio = math.exp(log_ratio)
    if standard_error == 0:
        return RatioInterval(ratio, ratio, ratio, confidence, None)

    degrees_of_freedom = (baseline_term + candidate_term) ** 2 / (
        baseline_term**2 / (len(baseline) - 1) + candidate_term**2 / (len(candidate) - 1)
    )
    half_width = t_quantile(0.5 + confidence / 2.0, degrees_of_freedom) * standard_error
    return RatioInterval(
        ratio=ratio,
        low=math.exp(log_ratio - half_width),
        high=math.exp(log_ratio + half_width),
        confidence=confidence,
        degrees_of_freedom=degrees_of_freedom,
    )
