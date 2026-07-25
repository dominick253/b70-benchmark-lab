"""
Intel GPU-specific telemetry collection.

Provides deep Intel GPU metrics via intel_gpu_top JSON mode,
sysfs interface, and DRM interface parsing.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

from benchlab.logging import get_logger

logger = get_logger("intel_gpu")


@dataclass
class IntelGPUSnapshot:
    """Complete GPU telemetry snapshot."""
    timestamp: float
    # EU (Execution Unit) metrics
    eu_total_utilization: float = 0.0
    render_3d_utilization: float = 0.0
    blt_utilization: float = 0.0
    video_utilization: float = 0.0
    video_protect_utilization: float = 0.0
    # Engine metrics
    render_engine_freq: float = 0.0
    blt_engine_freq: float = 0.0
    video_engine_freq: float = 0.0
    # Memory metrics
    memory_bandwidth: float = 0.0
    memory_used: float = 0.0
    memory_total: float = 0.0
    memory_utilization: float = 0.0
    # Power metrics
    package_power: float = 0.0
    render_power: float = 0.0
    memory_power: float = 0.0
    # Thermal metrics
    package_temperature: float = 0.0
    memory_temperature: float = 0.0
    # Frequency
    current_freq: float = 0.0
    max_freq: float = 0.0
    min_freq: float = 0.0
    # Raw data
    raw: dict[str, Any] = field(default_factory=dict)


class IntelGPUCollector:
    """Collect Intel GPU telemetry."""

    def __init__(self, gpu_index: int = 0) -> None:
        self.gpu_index = gpu_index
        self._snapshots: list[IntelGPUSnapshot] = []

    async def collect(self) -> IntelGPUSnapshot | None:
        """Collect a GPU telemetry snapshot."""
        try:
            data = await self._collect_intel_gpu_top()
            if data:
                return self._parse_intel_gpu_top(data)

            # Fallback to sysfs
            return await self._collect_sysfs()
        except Exception as e:
            logger.debug(f"GPU collection error: {e}")
            return None

    async def _collect_intel_gpu_top(self) -> dict[str, Any] | None:
        """Collect via intel_gpu_top JSON mode."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "intel_gpu_top", "-J", "-d", str(self.gpu_index),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode != 0:
                return None
            return json.loads(stdout.decode("utf-8", errors="replace"))
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, Exception):
            return None

    def _parse_intel_gpu_top(self, data: dict[str, Any]) -> IntelGPUSnapshot:
        """Parse intel_gpu_top JSON output."""
        now = time.time()
        engines = data.get("engines_metrics", {})
        mem = data.get("memory_metrics", {})
        pkg_power = data.get("package_power", {})
        pkg_thermal = data.get("package_thermal", {})

        # EU utilization
        eu_util = 0.0
        for engine_key in ("render_3d_0", "blt_0", "video_0", "video_protect_0"):
            engine = engines.get(engine_key, {})
            busy = engine.get("busy", {})
            if isinstance(busy, dict):
                eu_util = max(eu_util, float(busy.get("value", 0)))

        # Engine frequencies
        render_freq = 0.0
        blt_freq = 0.0
        video_freq = 0.0
        for engine_key in engines:
            freq = engines[engine_key].get("freq", {})
            if isinstance(freq, dict):
                val = float(freq.get("value", 0))
                if "render" in engine_key:
                    render_freq = max(render_freq, val)
                elif "blt" in engine_key:
                    blt_freq = max(blt_freq, val)
                elif "video" in engine_key:
                    video_freq = max(video_freq, val)

        # Memory
        mem_used = 0.0
        mem_total = 0.0
        if isinstance(mem.get("used"), dict):
            mem_used = float(mem["used"].get("value", 0)) / (1024 * 1024)
        if isinstance(mem.get("total"), dict):
            mem_total = float(mem["total"].get("value", 0)) / (1024 * 1024)

        # Power
        pkg_power_val = 0.0
        if isinstance(pkg_power.get("average"), dict):
            pkg_power_val = float(pkg_power["average"].get("value", 0))

        # Temperature
        pkg_temp = 0.0
        if isinstance(pkg_thermal.get("average"), dict):
            pkg_temp = float(pkg_thermal["average"].get("value", 0))

        return IntelGPUSnapshot(
            timestamp=now,
            eu_total_utilization=eu_util,
            memory_used=mem_used,
            memory_total=mem_total,
            memory_utilization=(mem_used / mem_total * 100) if mem_total > 0 else 0,
            package_power=pkg_power_val,
            package_temperature=pkg_temp,
            current_freq=render_freq,
            raw=data,
        )

    async def _collect_sysfs(self) -> IntelGPUSnapshot | None:
        """Fallback collection via sysfs."""
        try:
            card_path = f"/sys/class/drm/card{self.gpu_index}"

            freq = 0.0
            try:
                freq_file = f"{card_path}/gt_cur_freq_mhz"
                if asyncio.get_event_loop().run_in_executor(
                    None, lambda: __import__("os").path.exists(freq_file)
                ):
                    with open(freq_file) as f:
                        freq = float(f.read().strip())
            except Exception:
                pass

            return IntelGPUSnapshot(
                timestamp=time.time(),
                current_freq=freq,
                raw={"source": "sysfs"},
            )
        except Exception:
            return None

    def get_snapshots(self) -> list[IntelGPUSnapshot]:
        """Get all collected snapshots."""
        return self._snapshots

    def get_latest(self) -> IntelGPUSnapshot | None:
        """Get the latest snapshot."""
        return self._snapshots[-1] if self._snapshots else None

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        if not self._snapshots:
            return {}

        utils = [s.eu_total_utilization for s in self._snapshots]
        powers = [s.package_power for s in self._snapshots]
        temps = [s.package_temperature for s in self._snapshots]
        mem_utils = [s.memory_utilization for s in self._snapshots]

        def avg(data: list[float]) -> float:
            return sum(data) / len(data) if data else 0

        return {
            "eu_utilization": {"avg": avg(utils), "max": max(utils) if utils else 0},
            "power": {"avg": avg(powers), "max": max(powers) if powers else 0},
            "temperature": {"avg": avg(temps), "max": max(temps) if temps else 0},
            "memory_utilization": {"avg": avg(mem_utils), "max": max(mem_utils) if mem_utils else 0},
            "snapshot_count": len(self._snapshots),
        }
