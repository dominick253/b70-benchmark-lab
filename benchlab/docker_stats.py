"""
Docker container statistics collection for BenchLab.

Monitors Docker container resource usage including memory,
CPU, network, and disk I/O.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from benchlab.logging import get_logger

logger = get_logger("docker_stats")


@dataclass
class DockerStatsPoint:
    """A single Docker stats snapshot."""
    timestamp: float
    container_id: str
    container_name: str
    memory_usage: float = 0.0
    memory_limit: float = 0.0
    memory_percent: float = 0.0
    cpu_percent: float = 0.0
    cpu_usage_system: float = 0.0
    cpu_usage_total: float = 0.0
    network_rx: float = 0.0
    network_tx: float = 0.0
    network_rx_rate: float = 0.0
    network_tx_rate: float = 0.0
    block_read: float = 0.0
    block_write: float = 0.0
    block_read_rate: float = 0.0
    block_write_rate: float = 0.0
    pids: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


class DockerStatsCollector:
    """Collect Docker container statistics."""

    def __init__(self, interval: float = 1.0) -> None:
        self.interval = interval
        self._stats: dict[str, list[DockerStatsPoint]] = {}
        self._previous: dict[str, dict[str, float]] = {}

    async def collect(self, container_id: str, container_name: str = "") -> DockerStatsPoint | None:
        """Collect stats for a container."""
        try:
            import docker as docker_sdk
            client = docker_sdk.from_env()
            container = client.containers.get(container_id)
            stats = container.stats(one_shot=True)

            point = self._parse_stats(stats, container_id, container_name)
            point.timestamp = time.time()

            if container_id not in self._stats:
                self._stats[container_id] = []
            self._stats[container_id].append(point)

            return point
        except ImportError:
            logger.warning("Docker SDK not available")
            return None
        except Exception as e:
            logger.debug(f"Docker stats collection error: {e}")
            return None

    def _parse_stats(self, stats: Any, container_id: str, container_name: str) -> DockerStatsPoint:
        """Parse Docker stats output."""
        point = DockerStatsPoint(
            timestamp=0,
            container_id=container_id,
            container_name=container_name,
            raw=stats,
        )

        # Memory
        mem_stats = stats.get("memory_stats", {})
        point.memory_usage = mem_stats.get("usage", 0)
        point.memory_limit = mem_stats.get("limit", 1)
        point.memory_percent = (point.memory_usage / point.memory_limit * 100) if point.memory_limit > 0 else 0

        # CPU
        cpu_stats = stats.get("cpu_stats", {})
        prev_cpu = stats.get("precpu_stats", {})

        cpu_usage = cpu_stats.get("cpu_usage", {})
        point.cpu_usage_total = cpu_usage.get("total_usage", 0)
        point.cpu_usage_system = cpu_stats.get("system_cpu_usage", 0)

        # Calculate CPU percent
        if container_id in self._previous:
            prev = self._previous[container_id]
            cpu_delta = point.cpu_usage_total - prev.get("cpu_total", 0)
            sys_delta = point.cpu_usage_system - prev.get("sys_usage", 0)
            if sys_delta > 0:
                point.cpu_percent = (cpu_delta / sys_delta) * 100.0 * len(cpu_stats.get("cpu_usage", {}).get("percpu", []))
        else:
            point.cpu_percent = 0

        self._previous[container_id] = {
            "cpu_total": point.cpu_usage_total,
            "sys_usage": point.cpu_usage_system,
        }

        # Network
        networks = stats.get("networks", {})
        for iface, data in networks.items():
            point.network_rx += data.get("rx_bytes", 0)
            point.network_tx += data.get("tx_bytes", 0)

        # Block I/O
        blkio = stats.get("blkio_stats", {})
        io_serviced = blkio.get("io_serviced_recursive", [])
        for entry in io_serviced:
            op = entry.get("op", "")
            value = entry.get("value", 0)
            sector_size = 512
            bytes_val = value * sector_size
            if op == "Read":
                point.block_read += bytes_val
            elif op == "Write":
                point.block_write += bytes_val

        # PIDs
        point.pids = stats.get("pids_stats", {}).get("current", 0)

        return point

    def get_stats(self, container_id: str) -> list[DockerStatsPoint]:
        """Get stats for a container."""
        return self._stats.get(container_id, [])

    def get_latest(self, container_id: str) -> DockerStatsPoint | None:
        """Get latest stats for a container."""
        stats = self._stats.get(container_id, [])
        return stats[-1] if stats else None

    def get_summary(self, container_id: str) -> dict[str, Any]:
        """Get summary statistics for a container."""
        stats = self._stats.get(container_id, [])
        if not stats:
            return {}

        mem_utils = [s.memory_percent for s in stats]
        cpu_utils = [s.cpu_percent for s in stats]
        net_rx = [s.network_rx for s in stats]
        net_tx = [s.network_tx for s in stats]

        def avg(data: list[float]) -> float:
            return sum(data) / len(data) if data else 0

        return {
            "count": len(stats),
            "memory": {"avg_percent": avg(mem_utils), "max_percent": max(mem_utils) if mem_utils else 0},
            "cpu": {"avg_percent": avg(cpu_utils), "max_percent": max(cpu_utils) if cpu_utils else 0},
            "network_rx": {"total": net_rx[-1] if net_rx else 0},
            "network_tx": {"total": net_tx[-1] if net_tx else 0},
        }

    def clear(self) -> None:
        """Clear all collected stats."""
        self._stats.clear()
        self._previous.clear()
