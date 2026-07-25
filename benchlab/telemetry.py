"""
System and GPU telemetry collection for BenchLab.

Collects Intel GPU telemetry via intel_gpu_top JSON mode,
system metrics via psutil, and Docker container stats.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchlab.logging import get_logger
from benchlab.metrics import MetricsCollector

logger = get_logger("telemetry")


@dataclass
class GPUMetricPoint:
    """A single GPU telemetry point."""
    timestamp: float
    eu_utilization: float = 0.0
    render_engine_util: float = 0.0
    copy_engine_util: float = 0.0
    media_engine_util: float = 0.0
    memory_bandwidth: float = 0.0
    frequency_mhz: float = 0.0
    power_draw: float = 0.0
    temperature: float = 0.0
    vram_used: float = 0.0
    vram_total: float = 0.0
    vram_utilization: float = 0.0
    raw_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetricPoint:
    """A single system telemetry point."""
    timestamp: float
    cpu_percent: float = 0.0
    cpu_percent_per_core: list[float] = field(default_factory=list)
    memory_used: float = 0.0
    memory_total: float = 0.0
    memory_percent: float = 0.0
    disk_read: float = 0.0
    disk_write: float = 0.0
    disk_percent: float = 0.0
    network_sent: float = 0.0
    network_recv: float = 0.0
    swap_used: float = 0.0
    swap_total: float = 0.0
    page_faults: int = 0
    context_switches: int = 0
    boot_time: float = 0.0
    uptime: float = 0.0
    raw_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class DockerMetricPoint:
    """A single Docker container metric point."""
    timestamp: float
    container_id: str = ""
    memory_usage: float = 0.0
    memory_limit: float = 0.0
    memory_percent: float = 0.0
    cpu_percent: float = 0.0
    network_rx: float = 0.0
    network_tx: float = 0.0
    block_read: float = 0.0
    block_write: float = 0.0
    raw_data: dict[str, Any] = field(default_factory=dict)


class IntelGPUCollector:
    """Collect Intel GPU telemetry via intel_gpu_top."""

    def __init__(self, interval: float = 1.0) -> None:
        self.interval = interval
        self._running = False
        self._task: asyncio.Task | None = None
        self._metrics: list[GPUMetricPoint] = []

    async def start(self, metrics_collector: MetricsCollector) -> None:
        """Start GPU telemetry collection."""
        self._running = True
        self._task = asyncio.create_task(self._collect_loop(metrics_collector))
        logger.info("Intel GPU telemetry collection started")

    async def stop(self) -> None:
        """Stop GPU telemetry collection."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _collect_loop(self, metrics_collector: MetricsCollector) -> None:
        """Background collection loop."""
        while self._running:
            try:
                data = await self._collect_once()
                if data:
                    self._metrics.append(data)
                    metrics_collector.record("gpu_eu_utilization", data.eu_utilization)
                    metrics_collector.record("gpu_memory_bandwidth", data.memory_bandwidth)
                    metrics_collector.record("gpu_power_draw", data.power_draw)
                    metrics_collector.record("gpu_temperature", data.temperature)
                    metrics_collector.record("gpu_vram_utilization", data.vram_utilization)
                    metrics_collector.record("gpu_frequency", data.frequency_mhz)
            except Exception as e:
                logger.debug(f"GPU collection error: {e}")
            await asyncio.sleep(self.interval)

    async def _collect_once(self) -> GPUMetricPoint | None:
        """Collect a single GPU telemetry snapshot."""
        try:
            process = await asyncio.create_subprocess_exec(
                "intel_gpu_top", "-J", "-d", "0",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await process.communicate()

            if process.returncode != 0:
                return None

            data = json.loads(stdout.decode("utf-8", errors="replace"))
            return self._parse_gpu_data(data)

        except FileNotFoundError:
            logger.warning("intel_gpu_top not found, using fallback")
            return await self._collect_fallback()
        except Exception as e:
            logger.debug(f"GPU collection error: {e}")
            return None

    def _parse_gpu_data(self, data: dict[str, Any]) -> GPUMetricPoint:
        """Parse intel_gpu_top JSON output."""
        now = time.time()

        # Try to extract metrics from various possible JSON structures
        engines = data.get("engines_metrics", {})
        mem = data.get("memory_metrics", {})
        pkg = data.get("package_power", {})
        temps = data.get("package_thermal", {})

        eu_util = 0.0
        for engine_type in ("render_3d_0", "blt_0", "video_0", "video_protect_0"):
            engine = engines.get(engine_type, {})
            utilization = engine.get("busy", {})
            if isinstance(utilization, dict):
                eu_util = max(eu_util, utilization.get("value", 0))

        mem_used = mem.get("used", {}).get("value", 0) / (1024 * 1024) if isinstance(mem.get("used", {}).get("value", 0), (int, float)) else 0
        mem_total = mem.get("total", {}).get("value", 0) / (1024 * 1024) if isinstance(mem.get("total", {}).get("value", 0), (int, float)) else 0

        power = pkg.get("average", {}).get("value", 0) if isinstance(pkg.get("average", {}).get("value", 0), (int, float)) else 0
        temp = temps.get("average", {}).get("value", 0) if isinstance(temps.get("average", {}).get("value", 0), (int, float)) else 0

        freq = 0.0
        for engine_type in engines:
            freq_data = engines[engine_type].get("freq", {})
            if isinstance(freq_data, dict) and freq_data.get("value", 0) > freq:
                freq = freq_data.get("value", 0)

        return GPUMetricPoint(
            timestamp=now,
            eu_utilization=eu_util,
            memory_bandwidth=mem_used,
            power_draw=power,
            temperature=temp,
            vram_used=mem_used,
            vram_total=mem_total,
            vram_utilization=(mem_used / mem_total * 100) if mem_total > 0 else 0,
            frequency_mhz=freq,
            raw_data=data,
        )

    async def _collect_fallback(self) -> GPUMetricPoint | None:
        """Fallback GPU collection via /sys interface."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "cat", "/sys/class/drm/card0/gt_cur_freq_mhz",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            freq = float(stdout.decode().strip()) if stdout.strip() else 0

            return GPUMetricPoint(
                timestamp=time.time(),
                frequency_mhz=freq,
                raw_data={"source": "fallback"},
            )
        except Exception:
            return None

    def get_metrics(self) -> list[GPUMetricPoint]:
        return self._metrics


class SystemTelemetryCollector:
    """Collect system-level telemetry."""

    def __init__(self, interval: float = 1.0) -> None:
        self.interval = interval
        self._running = False
        self._task: asyncio.Task | None = None
        self._metrics: list[SystemMetricPoint] = []
        self._disk_io_start: dict[str, int] = {}
        self._net_io_start: dict[str, int] = {}

    async def start(self, metrics_collector: MetricsCollector) -> None:
        """Start system telemetry collection."""
        self._running = True
        self._task = asyncio.create_task(self._collect_loop(metrics_collector))
        logger.info("System telemetry collection started")

    async def stop(self) -> None:
        """Stop system telemetry collection."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _collect_loop(self, metrics_collector: MetricsCollector) -> None:
        """Background collection loop."""
        import psutil

        while self._running:
            try:
                data = await self._collect_once(psutil)
                if data:
                    self._metrics.append(data)
                    metrics_collector.record("cpu_percent", data.cpu_percent)
                    metrics_collector.record("memory_percent", data.memory_percent)
                    metrics_collector.record("disk_percent", data.disk_percent)

                    if self._metrics and len(self._metrics) >= 2:
                        prev = self._metrics[-2]
                        data_rate = (data.timestamp - prev.timestamp)
                        if data_rate > 0:
                            metrics_collector.record("disk_read_rate", data.disk_read / data_rate)
                            metrics_collector.record("disk_write_rate", data.disk_write / data_rate)
                            metrics_collector.record("network_sent_rate", data.network_sent / data_rate)
                            metrics_collector.record("network_recv_rate", data.network_recv / data_rate)
            except Exception as e:
                logger.debug(f"System collection error: {e}")
            await asyncio.sleep(self.interval)

    async def _collect_once(self, psutil: Any) -> SystemMetricPoint | None:
        """Collect a single system telemetry snapshot."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0)
            cpu_percent_per_core = psutil.cpu_percent(interval=0, percpu=True)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            disk_io = psutil.disk_io_counters()
            net_io = psutil.net_io_counters()

            now = time.time()
            uptime = now - psutil.boot_time()

            return SystemMetricPoint(
                timestamp=now,
                cpu_percent=cpu_percent,
                cpu_percent_per_core=cpu_percent_per_core,
                memory_used=memory.used,
                memory_total=memory.total,
                memory_percent=memory.percent,
                disk_percent=disk.percent,
                disk_read=disk_io.read_bytes if disk_io else 0,
                disk_write=disk_io.write_bytes if disk_io else 0,
                network_sent=net_io.bytes_sent if net_io else 0,
                network_recv=net_io.bytes_recv if net_io else 0,
                page_faults=0,
                context_switches=0,
                boot_time=psutil.boot_time(),
                uptime=uptime,
                raw_data={
                    "cpu_count": psutil.cpu_count(),
                    "swap": psutil.swap_memory()._asdict() if hasattr(psutil, "swap_memory") else {},
                },
            )
        except Exception as e:
            logger.debug(f"System collection error: {e}")
            return None

    def get_metrics(self) -> list[SystemMetricPoint]:
        return self._metrics


class TelemetryManager:
    """Manage all telemetry collection."""

    def __init__(self, collection_interval: float = 1.0) -> None:
        self.collection_interval = collection_interval
        self.gpu_collector = IntelGPUCollector(collection_interval)
        self.system_collector = SystemTelemetryCollector(collection_interval)
        self.metrics_collector = MetricsCollector(collection_interval)
        self._running = False

    async def start(self) -> None:
        """Start all telemetry collectors."""
        self._running = True
        await self.gpu_collector.start(self.metrics_collector)
        await self.system_collector.start(self.metrics_collector)
        logger.info("All telemetry collectors started")

    async def stop(self) -> None:
        """Stop all telemetry collectors."""
        self._running = False
        await self.gpu_collector.stop()
        await self.system_collector.stop()
        logger.info("All telemetry collectors stopped")

    def get_all_metrics(self) -> dict[str, Any]:
        """Get all collected metrics."""
        return {
            "gpu": [
                {
                    "timestamp": m.timestamp,
                    "eu_utilization": m.eu_utilization,
                    "power_draw": m.power_draw,
                    "temperature": m.temperature,
                    "vram_utilization": m.vram_utilization,
                }
                for m in self.gpu_collector.get_metrics()
            ],
            "system": [
                {
                    "timestamp": m.timestamp,
                    "cpu_percent": m.cpu_percent,
                    "memory_percent": m.memory_percent,
                    "disk_percent": m.disk_percent,
                }
                for m in self.system_collector.get_metrics()
            ],
            "aggregated": self.metrics_collector.export_json(),
        }
