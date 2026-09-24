"""VRAM sampling on AMD GPUs.

The grader samples VRAM every 3 s and requires the *peak* to be within
[1 GiB, 48 GiB * 1.01]. This module lets us enforce the same check locally.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field

GIB = 1024**3
VRAM_MIN_BYTES = 1 * GIB
VRAM_MAX_BYTES = int(48 * GIB * 1.01)


def _run(cmd: list[str]) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def used_vram_bytes() -> int | None:
    """Total VRAM in use across visible AMD GPUs, or None if no tool is available."""
    out = _run(["amd-smi", "metric", "--mem", "--json"])
    if out:
        try:
            data = json.loads(out)
            items = data if isinstance(data, list) else data.get("gpu_data", [data])
            total = 0
            for item in items:
                mem = item.get("mem_usage") or item.get("memory") or {}
                used = mem.get("used_vram") or mem.get("USED_VRAM")
                if isinstance(used, dict):
                    value, unit = float(used.get("value", 0)), str(used.get("unit", "MB")).upper()
                else:
                    value, unit = float(used or 0), "MB"
                total += int(value * {"B": 1, "KB": 1024, "MB": 1024**2, "GB": GIB}.get(unit, 1024**2))
            return total
        except (ValueError, TypeError, AttributeError):
            pass
    out = _run(["rocm-smi", "--showmeminfo", "vram", "--json"])
    if out:
        try:
            data = json.loads(out)
            return sum(int(v.get("VRAM Total Used Memory (B)", 0)) for v in data.values() if isinstance(v, dict))
        except (ValueError, TypeError):
            pass
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():  # ROCm builds expose the cuda API
            return sum(
                torch.cuda.mem_get_info(i)[1] - torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())
            )
    except Exception:  # noqa: BLE001
        pass
    return None


@dataclass
class VramMonitor:
    """Background sampler: ``with VramMonitor() as m: ...; m.peak_bytes``."""

    interval: float = 1.0
    samples: list[int] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def __enter__(self) -> VramMonitor:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval * 2)

    def _loop(self) -> None:
        while not self._stop.is_set():
            value = used_vram_bytes()
            if value is not None:
                self.samples.append(value)
            self._stop.wait(self.interval)

    @property
    def peak_bytes(self) -> int | None:
        return max(self.samples) if self.samples else None

    def within_limits(self) -> bool | None:
        peak = self.peak_bytes
        return None if peak is None else VRAM_MIN_BYTES <= peak <= VRAM_MAX_BYTES


def human(n: int | None) -> str:
    if n is None:
        return "n/a"
    return f"{n / GIB:.2f} GiB"


if __name__ == "__main__":
    while True:
        print(time.strftime("%H:%M:%S"), human(used_vram_bytes()), flush=True)
        time.sleep(3)


__all__ = ["VRAM_MAX_BYTES", "VRAM_MIN_BYTES", "VramMonitor", "human", "used_vram_bytes"]
