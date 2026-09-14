# Example reports

These abbreviated terminal reports are representative deterministic snapshots aligned with the v0.2 regression scenarios. They illustrate rule behavior; their values are not benchmarks and will differ by model, hardware, traffic, and Prometheus bucket layout.

## Healthy service

High utilization alone is not a finding. Here the cache is busy, but there are no waiting requests, preemptions, or SLO violations.

```text
INFERENCE DOCTOR v0.2                         0 critical · 0 warnings
TTFT p95             57.1 ms
TPOT p95              9.6 ms
Queue p95             285.0 ms
Queue mean            5.0 ms
Prefill p95           24.0 ms
Prompt tokens p95     100.0 tokens
Waiting requests      0
KV cache              91.0%
Preemptions/min       0.00

No diagnostic conditions were triggered.
```

## Queue / capacity pressure

TTFT crosses its SLO while output-token latency stays healthy. Waiting work and elevated KV occupancy raise confidence. Queue p95 is displayed, but is not used to score this finding.

```text
INFERENCE DOCTOR v0.2                         1 critical · 0 warnings
TTFT p95             812.0 ms
TPOT p95              42.0 ms
Queue p95             285.0 ms
Queue mean            118.0 ms
Prefill p95           88.0 ms
Prompt tokens p95     512.0 tokens
Waiting requests      18
KV cache              91.0%
Preemptions/min       0.00

P0 QUEUE / CAPACITY PRESSURE
Confidence: HIGH
Evidence: TTFT p95 812.0 ms; waiting requests 18; KV cache 91.0%;
          TPOT p95 42.0 ms
Diagnosis: TTFT exceeds the supplied SLO while decode latency remains healthy.
```

## Decode pressure

TTFT remains within its SLO while TPOT breaches its SLO, concentrating the evidence in output-token generation rather than admission or queueing.

```text
INFERENCE DOCTOR v0.2                         0 critical · 1 warnings
TTFT p95             220.0 ms
TPOT p95              91.0 ms
Queue p95             285.0 ms
Queue mean            12.0 ms
Prefill p95           46.0 ms
Prompt tokens p95     256.0 tokens
Waiting requests      0
KV cache              35.0%
Preemptions/min       0.00

P0 DECODE PRESSURE
Confidence: MEDIUM
Evidence: TPOT p95 91.0 ms; TTFT p95 220.0 ms
Diagnosis: Per-output-token latency exceeds the supplied SLO while TTFT is healthy.
```

## KV-cache pressure

KV occupancy is meaningful here because the scheduler is also waiting on work. A cache above 90% with no corroborating evidence would produce no finding.

```text
INFERENCE DOCTOR v0.2                         1 critical · 0 warnings
TTFT p95             650.0 ms
TPOT p95              37.0 ms
Queue p95             N/A
Queue mean            N/A
Prefill p95           N/A
Prompt tokens p95     N/A
Waiting requests      9
KV cache              97.0%
Preemptions/min       4.50

P0 KV CACHE PRESSURE
Confidence: HIGH
Evidence: KV cache 97.0%; waiting requests 9; preemptions/min 4.50;
          TTFT p95 650.0 ms
Diagnosis: High KV occupancy is accompanied by scheduler pressure.
```
