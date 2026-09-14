from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GPU_QUERY = (
    "uuid,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu,"
    "pstate,memory.used"
)


def _number(value: str) -> float | None:
    value = value.strip().replace("[N/A]", "")
    try:
        return float(value) if value else None
    except ValueError:
        return None


def _gpu_sample() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        f"--query-gpu={GPU_QUERY}",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise RuntimeError(completed.stderr.strip() or "nvidia-smi returned no data")
    values = [part.strip() for part in completed.stdout.splitlines()[0].split(",")]
    if len(values) < 8:
        raise RuntimeError("nvidia-smi returned incomplete GPU telemetry")
    return {
        "gpu_uuid": values[0] or None,
        "gpu_utilization_percent": _number(values[1]),
        "sm_clock_mhz": _number(values[2]),
        "memory_clock_mhz": _number(values[3]),
        "power_draw_watts": _number(values[4]),
        "temperature_celsius": _number(values[5]),
        "performance_state": values[6] or None,
        "gpu_memory_used_mib": _number(values[7]),
    }


def _cpu_sample() -> dict[str, Any]:
    sample: dict[str, Any] = {
        "cpu_utilization_percent": None,
        "load_average": None,
        "system_memory_used_percent": None,
    }
    try:
        import psutil  # type: ignore

        sample["cpu_utilization_percent"] = float(psutil.cpu_percent(interval=None))
        sample["load_average"] = list(psutil.getloadavg())
        sample["system_memory_used_percent"] = float(psutil.virtual_memory().percent)
        return sample
    except (ImportError, AttributeError, OSError):
        pass

    try:
        sample["load_average"] = list(os.getloadavg())
    except (AttributeError, OSError):
        pass
    try:
        memory = {
            line.split(":", 1)[0]: int(line.split()[1])
            for line in Path("/proc/meminfo").read_text().splitlines()
            if ":" in line and len(line.split()) >= 2
        }
        total = memory.get("MemTotal")
        available = memory.get("MemAvailable")
        if total and available is not None:
            sample["system_memory_used_percent"] = (total - available) / total * 100
    except (OSError, ValueError):
        pass
    return sample


def sample_once() -> dict[str, Any]:
    sample: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "available": True,
        "gpu_uuid": None,
        "gpu_utilization_percent": None,
        "sm_clock_mhz": None,
        "memory_clock_mhz": None,
        "power_draw_watts": None,
        "temperature_celsius": None,
        "performance_state": None,
        "gpu_memory_used_mib": None,
        **_cpu_sample(),
    }
    try:
        sample.update(_gpu_sample())
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        sample["available"] = False
        sample["error"] = str(exc)
    return sample


class TelemetrySampler:
    """Best-effort periodic sampler; telemetry errors never stop a benchmark."""

    def __init__(self, interval_seconds: float = 5.0) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample(self) -> dict[str, Any]:
        value = sample_once()
        self.samples.append(value)
        return value

    def start(self) -> None:
        if self._thread is not None:
            return
        self.sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.sample()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval_seconds + 1, 2))
            self._thread = None
        if not self.samples:
            self.sample()

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema_version": "inference-doctor-telemetry-1",
                    "interval_seconds": self.interval_seconds,
                    "samples": self.samples,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
