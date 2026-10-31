"""(Experimental) Cost doctor: turn measured serving numbers into dollars.

Measured inputs: token counters, queue and KV-cache gauges (Prometheus) or benchmark
throughput (result files). Everything that turns those into *savings* is a heuristic
with its assumption written next to it; see docs/cost.md.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from importlib import resources
from typing import Any

HOURS_PER_MONTH = 730.0
SECONDS_PER_MONTH = HOURS_PER_MONTH * 3600.0
TARGET_OCCUPANCY = 0.8  # assumption: healthy steady-state batch/KV occupancy
FP8_SPEEDUP = 1.3  # assumption: conservative throughput gain from FP8 weights+KV
PREFIX_SAVING_OF_PROMPT_SHARE = 0.2  # assumption


def _load(name: str) -> dict[str, Any]:
    return json.loads(resources.files("inference_doctor").joinpath("data", name).read_text())


def gpu_table() -> dict[str, Any]:
    return _load("gpu_prices.json")


def api_table() -> dict[str, Any]:
    return _load("api_prices.json")


def resolve_gpu(name: str) -> tuple[str, dict[str, Any]]:
    table = gpu_table()["gpus"]
    key = name.upper().replace(" ", "-")
    aliases = {"H100": "H100-SXM", "A100": "A100-80G"}
    key = aliases.get(key, key)
    if key not in table:
        raise ValueError(f"Unknown GPU '{name}'. Known: {', '.join(table)}; or pass --price")
    return key, table[key]


def list_price(gpu: str, provider: str | None = None) -> tuple[float, str]:
    key, row = resolve_gpu(gpu)
    prices = row["prices"]
    provider = provider or min(prices, key=prices.get)
    if provider not in prices:
        raise ValueError(f"No {provider} price for {key}; have {', '.join(prices)}")
    return float(prices[provider]), gpu_table()["sources"][provider]["url"]


@dataclass
class Usage:
    source: str
    prompt_tps: float
    gen_tps: float
    running: float | None = None
    waiting: float | None = None
    kv_usage: float | None = None
    prefix_hit_rate: float | None = None
    prefix_caching_enabled: bool | None = None
    cache_dtype: str | None = None
    gpu_util: float | None = None
    measured_rates: bool = True


@dataclass
class Fix:
    title: str
    monthly_savings: float
    why: str
    how: str
    basis: str  # "measured", "heuristic" or "list price"


@dataclass
class CostReport:
    gpu: str
    gpus: int
    price_per_gpu_hour: float
    price_source: str
    monthly_cost: float
    cost_per_1m_tokens: float | None
    cost_per_1m_output_tokens: float | None
    occupancy: float | None
    monthly_waste: float | None
    usage: Usage
    fixes: list[Fix] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def usage_from_window(w: Any) -> Usage:
    return Usage(
        source=f"prometheus ({w.engine}, {w.seconds:.0f} s window)",
        prompt_tps=w.prompt_tps,
        gen_tps=w.gen_tps,
        running=w.running,
        waiting=w.waiting,
        kv_usage=w.kv_usage,
        prefix_hit_rate=w.prefix_hit_rate,
        prefix_caching_enabled=w.prefix_caching_enabled,
        cache_dtype=w.cache_dtype,
        gpu_util=w.gpu_util,
        measured_rates=w.measured_rates,
    )


def usage_from_benchmarks(results: list[Any]) -> Usage:
    out = [r.metrics.output_token_throughput for r in results if r.metrics.output_token_throughput]
    tot = [r.metrics.total_token_throughput for r in results if r.metrics.total_token_throughput]
    if not out and not tot:
        raise ValueError("Benchmark results carry no token throughput; cost needs tokens/s")
    gen = statistics.median(out) if out else 0.0
    total = statistics.median(tot) if tot else gen
    return Usage(
        source=f"benchmark ({len(results)} runs, median)",
        prompt_tps=max(total - gen, 0),
        gen_tps=gen,
    )


def _per_million(cost_per_hour: float, tps: float) -> float | None:
    return None if tps <= 0 else cost_per_hour / (tps * 3600 / 1e6)


def analyze(
    usage: Usage,
    gpu: str,
    gpus: int,
    price: float | None = None,
    provider: str | None = None,
    max_num_seqs: int = 256,
    replicas: int = 1,
    tp: int = 1,
    api_model: str | None = None,
) -> CostReport:
    key, row = resolve_gpu(gpu)
    if price is None:
        price, src = list_price(key, provider)
    else:
        src = "user supplied (--price)"
    hourly = price * gpus
    monthly = hourly * HOURS_PER_MONTH
    tps = usage.prompt_tps + usage.gen_tps
    rep = CostReport(
        gpu=key,
        gpus=gpus,
        price_per_gpu_hour=price,
        price_source=src,
        monthly_cost=monthly,
        cost_per_1m_tokens=_per_million(hourly, tps),
        cost_per_1m_output_tokens=_per_million(hourly, usage.gen_tps),
        occupancy=None,
        monthly_waste=None,
        usage=usage,
    )
    if not usage.measured_rates:
        rep.notes.append("Rates are lifetime averages from one scrape, not a measured window.")

    # occupancy: the fuller of batch slots and KV cache (measured gauges, averaged)
    parts = []
    if usage.running is not None and max_num_seqs > 0:
        parts.append(usage.running / replicas / max_num_seqs)
    if usage.kv_usage is not None:
        parts.append(usage.kv_usage)
    queued = (usage.waiting or 0) > 0.5
    if parts:
        occ = min(max(parts), 1.0)
        rep.occupancy = occ
        rep.monthly_waste = 0.0 if queued else monthly * max(0.0, 1 - occ / TARGET_OCCUPANCY)

    fixes: list[Fix] = []
    if rep.occupancy is not None and not queued and rep.occupancy < TARGET_OCCUPANCY:
        need = max(tp, math.ceil(gpus * rep.occupancy / TARGET_OCCUPANCY / tp) * tp)
        if need < gpus:
            fixes.append(
                Fix(
                    f"Right-size: {gpus} -> {need} GPUs",
                    (gpus - need) * price * HOURS_PER_MONTH,
                    f"Occupancy {rep.occupancy:.0%} with an empty queue; traffic fits in {need} GPUs "
                    f"at {TARGET_OCCUPANCY:.0%} target occupancy.",
                    "Scale replicas down (or autoscale on vllm:num_requests_waiting / KV usage). "
                    "Check p95 TTFT after; peaks are not visible in a short window.",
                    "heuristic",
                )
            )
    if queued and usage.kv_usage is not None and usage.kv_usage < 0.6:
        fixes.append(
            Fix(
                "Raise --max-num-seqs (batch limit)",
                monthly * (1 - usage.kv_usage / TARGET_OCCUPANCY) * 0.5,
                f"Requests are queueing while KV cache is only {usage.kv_usage:.0%} full: the "
                "scheduler's sequence cap, not memory, is the bottleneck.",
                "Raise --max-num-seqs (and --max-num-batched-tokens) until KV usage nears 80%.",
                "heuristic",
            )
        )
    share = usage.prompt_tps / tps if tps else 0
    if usage.prefix_caching_enabled is False and share > 0.5:
        fixes.append(
            Fix(
                "Enable prefix caching",
                monthly * share * PREFIX_SAVING_OF_PROMPT_SHARE,
                f"{share:.0%} of tokens are prompt tokens and prefix caching is off.",
                "vLLM: --enable-prefix-caching. Saving depends on shared prefixes (system "
                "prompts, multi-turn); measure the hit rate after.",
                "heuristic",
            )
        )
    elif usage.prefix_hit_rate is not None and usage.prefix_hit_rate < 0.1 and share > 0.5:
        rep.notes.append(
            f"Prefix cache hit rate is {usage.prefix_hit_rate:.0%}; "
            "prompts may not share prefixes, or routing breaks session affinity."
        )
    if row.get("fp8") and (usage.cache_dtype in (None, "auto")):
        fixes.append(
            Fix(
                "Quantize to FP8 (weights and KV cache)",
                monthly * (1 - 1 / FP8_SPEEDUP),
                f"{key} has FP8 tensor cores; the KV cache dtype is "
                f"'{usage.cache_dtype or 'unknown'}'.",
                "Serve an FP8 checkpoint (or --quantization fp8) and --kv-cache-dtype fp8; gate "
                "quality with an eval before rollout.",
                "heuristic",
            )
        )
    cur_eff = price / row["rel_throughput"]
    best = None
    for other, orow in gpu_table()["gpus"].items():
        if other == key:
            continue
        oprice = min(orow["prices"].values())
        if orow["vram_gb"] * gpus < row["vram_gb"] * gpus * 0.999:
            continue  # must hold the same model at the same count
        saving = monthly * (1 - (oprice / orow["rel_throughput"]) / cur_eff)
        if saving > 0 and (best is None or saving > best[1]):
            best = (other, saving, oprice)
    if best:
        fixes.append(
            Fix(
                f"Switch GPU: {key} -> {best[0]}",
                best[1],
                f"{best[0]} lists at ${best[2]:.2f}/GPU-h; per unit of assumed throughput it is "
                "cheaper.",
                "Benchmark the same workload on it first (inference-doctor compare-runs).",
                "list price + heuristic throughput",
            )
        )
    apis = api_table()
    if api_model:
        if api_model not in apis["models"]:
            raise ValueError(f"Unknown --api-model; known: {', '.join(apis['models'])}")
        m = apis["models"][api_model]
        api_month = (
            (usage.prompt_tps * m["input"] + usage.gen_tps * m["output"]) * SECONDS_PER_MONTH / 1e6
        )
        src_url = apis["sources"][m["provider"]]["url"]
        if api_month < monthly:
            fixes.append(
                Fix(
                    f"Use a hosted API: {m['label']}",
                    monthly - api_month,
                    f"Your measured traffic would cost ${api_month:,.0f}/month at list price "
                    f"({src_url}) vs ${monthly:,.0f} for the GPUs.",
                    "Only valid if that model meets your quality, latency and data-residency needs.",
                    "measured traffic x list price",
                )
            )
        else:
            rep.notes.append(
                f"Hosted {m['label']} would cost ${api_month:,.0f}/month for "
                "this traffic; self-hosting is cheaper."
            )
    fixes.sort(key=lambda f: f.monthly_savings, reverse=True)
    rep.fixes = fixes
    if len(fixes) > 1:
        rep.notes.append("Savings overlap; they are not additive.")
    return rep


def render(rep: CostReport) -> str:
    u = rep.usage

    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.0%}"

    def money(x: float | None, fmt: str = ",.0f") -> str:
        return "n/a" if x is None else f"${x:{fmt}}"

    lines = [
        "inference-doctor cost (experimental)",
        f"  source            {u.source}",
        f"  hardware          {rep.gpus} x {rep.gpu} @ ${rep.price_per_gpu_hour:.2f}/GPU-h ({rep.price_source})",
        f"  throughput        {u.prompt_tps:,.0f} prompt + {u.gen_tps:,.0f} output tok/s   [measured]",
        f"  cost / 1M tokens  {money(rep.cost_per_1m_tokens, '.3f')} all, {money(rep.cost_per_1m_output_tokens, '.3f')} output",
        f"  monthly cost      {money(rep.monthly_cost)}  (730 h)",
        f"  occupancy         {pct(rep.occupancy)}  (batch slots / KV cache, max of the two)",
        f"  KV cache usage    {pct(u.kv_usage)}   queue {u.waiting if u.waiting is not None else 'n/a'}   "
        f"GPU util (DCGM) {pct(u.gpu_util)}",
        f"  est. $ wasted/mo  {money(rep.monthly_waste)}  [heuristic: idle capacity below {TARGET_OCCUPANCY:.0%} occupancy]",
        "",
        "Ranked fixes (estimated monthly savings):",
    ]
    if not rep.fixes:
        lines.append("  none found")
    for i, f in enumerate(rep.fixes, 1):
        lines += [
            f"  {i}. {f.title:<44} ~{money(f.monthly_savings)}/mo  [{f.basis}]",
            f"     why: {f.why}",
            f"     how: {f.how}",
        ]
    for n in rep.notes:
        lines.append(f"note: {n}")
    return "\n".join(lines)


@dataclass
class CostGate:
    result: str  # PASS | FAIL | INCONCLUSIVE
    baseline_mean: float
    candidate_mean: float
    change_percent: float
    change_percent_low: float
    change_percent_high: float
    confidence: float
    max_increase_percent: float
    runs: tuple[int, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def cost_gate(
    baseline: list[float],
    candidate: list[float],
    max_increase_pct: float,
    confidence: float = 0.9,
) -> CostGate:
    """Gate on $/1M tokens with the same Welch log-ratio interval as compare-runs.

    FAIL when the whole interval for the increase is above the limit, PASS when the
    whole interval is at or below it, INCONCLUSIVE when it straddles the limit.
    """
    from inference_doctor.stats import welch_log_ratio_interval

    iv = welch_log_ratio_interval(baseline, candidate, confidence)
    low, high = (iv.low - 1) * 100, (iv.high - 1) * 100
    if low > max_increase_pct:
        result = "FAIL"
    elif high <= max_increase_pct:
        result = "PASS"
    else:
        result = "INCONCLUSIVE"
    return CostGate(
        result,
        statistics.fmean(baseline),
        statistics.fmean(candidate),
        (iv.ratio - 1) * 100,
        low,
        high,
        confidence,
        max_increase_pct,
        (len(baseline), len(candidate)),
    )
