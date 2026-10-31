"""(cost) Parse Prometheus text exposition from vLLM / SGLang (and optional DCGM) into one normalized snapshot."""

from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass, field

_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([^\s]+)(?:\s+[0-9]+)?$")
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')

Samples = dict[str, list[tuple[dict[str, str], float]]]


def parse_prometheus(text: str) -> Samples:
    out: Samples = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name, labels, value = m.groups()
        try:
            v = float(value)
        except ValueError:
            continue
        lab = dict(_LABEL.findall(labels or ""))
        out.setdefault(name, []).append((lab, v))
    return out


def _sum(s: Samples, *names: str) -> float | None:
    for n in names:
        if n in s:
            return sum(v for _, v in s[n])
    return None


def _mean(s: Samples, *names: str) -> float | None:
    for n in names:
        if n in s and s[n]:
            return sum(v for _, v in s[n]) / len(s[n])
    return None


@dataclass
class Snapshot:
    engine: str
    prompt_tokens_total: float | None = None
    generation_tokens_total: float | None = None
    running: float | None = None
    waiting: float | None = None
    kv_usage: float | None = None  # 0..1
    prefix_hits: float | None = None
    prefix_queries: float | None = None
    prefix_hit_rate: float | None = None  # 0..1 gauge if exposed directly
    prefix_caching_enabled: bool | None = None
    cache_dtype: str | None = None
    gpu_util: float | None = None  # 0..1 from DCGM if present
    timestamp: float | None = None
    extra: dict[str, str] = field(default_factory=dict)


def snapshot_from_text(text: str) -> Snapshot:
    s = parse_prometheus(text)
    names = set(s)
    if any(n.startswith("vllm:") for n in names):
        engine = "vllm"
    elif any(n.startswith("sglang:") for n in names):
        engine = "sglang"
    else:
        engine = "unknown"

    snap = Snapshot(engine=engine)
    if engine == "vllm":
        snap.prompt_tokens_total = _sum(s, "vllm:prompt_tokens_total")
        snap.generation_tokens_total = _sum(s, "vllm:generation_tokens_total")
        snap.running = _sum(s, "vllm:num_requests_running")
        snap.waiting = _sum(s, "vllm:num_requests_waiting")
        snap.kv_usage = _mean(s, "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")
        snap.prefix_hits = _sum(s, "vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits")
        snap.prefix_queries = _sum(
            s, "vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries"
        )
        snap.prefix_hit_rate = _mean(s, "vllm:gpu_prefix_cache_hit_rate")
        info = s.get("vllm:cache_config_info")
        if info:
            lab = info[0][0]
            snap.extra.update(lab)
            if "enable_prefix_caching" in lab:
                snap.prefix_caching_enabled = lab["enable_prefix_caching"].lower() == "true"
            snap.cache_dtype = lab.get("cache_dtype")
    elif engine == "sglang":
        snap.prompt_tokens_total = _sum(s, "sglang:prompt_tokens_total")
        snap.generation_tokens_total = _sum(s, "sglang:generation_tokens_total")
        snap.running = _sum(s, "sglang:num_running_reqs")
        snap.waiting = _sum(s, "sglang:num_queue_reqs")
        snap.kv_usage = _mean(s, "sglang:token_usage")
        snap.prefix_hit_rate = _mean(s, "sglang:cache_hit_rate")
        # SGLang's RadixAttention prefix cache is on by default
        snap.prefix_caching_enabled = True

    dcgm = _mean(s, "DCGM_FI_DEV_GPU_UTIL")
    if dcgm is not None:
        snap.gpu_util = dcgm / 100.0
    ts = _sum(s, "inference_doctor_scrape_timestamp_seconds")
    if ts is not None:
        snap.timestamp = ts
    return snap


def fetch(url: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - user-supplied URL
        return r.read().decode("utf-8", "replace")


@dataclass
class Window:
    """Rates derived from two snapshots (or one snapshot + assumed window)."""

    engine: str
    seconds: float
    prompt_tps: float
    gen_tps: float
    running: float
    waiting: float
    kv_usage: float | None
    prefix_hit_rate: float | None
    prefix_caching_enabled: bool | None
    cache_dtype: str | None
    gpu_util: float | None
    measured_rates: bool


def _avg(a: float | None, b: float | None) -> float | None:
    vals = [x for x in (a, b) if x is not None]
    return sum(vals) / len(vals) if vals else None


def window(a: Snapshot, b: Snapshot | None, seconds: float | None) -> Window:
    if b is not None:
        dt = seconds
        if dt is None and a.timestamp is not None and b.timestamp is not None:
            dt = b.timestamp - a.timestamp
        if not dt or dt <= 0:
            raise ValueError(
                "Need a positive window: pass --seconds or include inference_doctor_scrape_timestamp_seconds"
            )
        pt = ((b.prompt_tokens_total or 0) - (a.prompt_tokens_total or 0)) / dt
        gt = ((b.generation_tokens_total or 0) - (a.generation_tokens_total or 0)) / dt
        hr = None
        if a.prefix_queries is not None and b.prefix_queries is not None:
            dq = b.prefix_queries - a.prefix_queries
            if dq > 0:
                hr = ((b.prefix_hits or 0) - (a.prefix_hits or 0)) / dq
        if hr is None:
            hr = _avg(a.prefix_hit_rate, b.prefix_hit_rate)
        return Window(
            engine=b.engine,
            seconds=dt,
            prompt_tps=max(pt, 0.0),
            gen_tps=max(gt, 0.0),
            running=_avg(a.running, b.running) or 0.0,
            waiting=_avg(a.waiting, b.waiting) or 0.0,
            kv_usage=_avg(a.kv_usage, b.kv_usage),
            prefix_hit_rate=hr,
            prefix_caching_enabled=b.prefix_caching_enabled,
            cache_dtype=b.cache_dtype,
            gpu_util=_avg(a.gpu_util, b.gpu_util),
            measured_rates=True,
        )
    if not seconds:
        raise ValueError(
            "A single snapshot only has lifetime counters; pass --seconds (server uptime) or give two snapshots"
        )
    hr = a.prefix_hit_rate
    if hr is None and a.prefix_queries:
        hr = (a.prefix_hits or 0) / a.prefix_queries
    return Window(
        engine=a.engine,
        seconds=seconds,
        prompt_tps=(a.prompt_tokens_total or 0) / seconds,
        gen_tps=(a.generation_tokens_total or 0) / seconds,
        running=a.running or 0.0,
        waiting=a.waiting or 0.0,
        kv_usage=a.kv_usage,
        prefix_hit_rate=hr,
        prefix_caching_enabled=a.prefix_caching_enabled,
        cache_dtype=a.cache_dtype,
        gpu_util=a.gpu_util,
        measured_rates=False,
    )
