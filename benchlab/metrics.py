"""
Metrics collection and aggregation for BenchLab.

Collects, aggregates, and stores benchmark metrics
with support for streaming and batched collection.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from benchlab.logging import get_logger

logger = get_logger("metrics")


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    timestamp: float
    labels: dict[str, str] = field(default_factory=dict)
    metric_type: str = "gauge"  # gauge, counter, histogram


class MetricsCollector:
    """Collect and aggregate metrics during benchmark runs."""

    def __init__(self, collection_interval: float = 1.0) -> None:
        self.collection_interval = collection_interval
        self._metrics: dict[str, list[MetricPoint]] = {}
        self._running = False
        self._task: asyncio.Task | None = None

    def record(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Record a metric value."""
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=time.time(),
            labels=labels or {},
        )

        if name not in self._metrics:
            self._metrics[name] = []

        self._metrics[name].append(point)

    def record_batch(self, points: list[MetricPoint]) -> None:
        """Record multiple metric points."""
        for point in points:
            self.record(point.name, point.value, point.labels)

    def get_metrics(self, name: str | None = None) -> list[MetricPoint]:
        """Get recorded metrics."""
        if name:
            return self._metrics.get(name, [])
        all_points: list[MetricPoint] = []
        for points in self._metrics.values():
            all_points.extend(points)
        return sorted(all_points, key=lambda p: p.timestamp)

    def get_summary(
        self,
        name: str,
        window_seconds: float = 60.0,
    ) -> dict[str, float]:
        """Get summary statistics for a metric within a time window."""
        now = time.time()
        cutoff = now - window_seconds
        points = [
            p for p in self._metrics.get(name, [])
            if p.timestamp >= cutoff
        ]

        if not points:
            return {"count": 0, "avg": 0, "min": 0, "max": 0, "p95": 0, "p99": 0}

        values = [p.value for p in points]
        values.sort()

        def percentile(data: list[float], p: float) -> float:
            if not data:
                return 0.0
            k = (len(data) - 1) * p
            f_idx = int(k)
            c_idx = f_idx + 1
            if c_idx >= len(data):
                return data[-1]
            return data[f_idx] + (k - f_idx) * (data[c_idx] - data[f_idx])

        return {
            "count": len(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
        }

    def get_latest(self, name: str) -> MetricPoint | None:
        """Get the latest value for a metric."""
        points = self._metrics.get(name, [])
        return points[-1] if points else None

    def clear(self) -> None:
        """Clear all metrics."""
        self._metrics.clear()

    def export_json(self) -> dict[str, Any]:
        """Export all metrics as JSON."""
        result: dict[str, Any] = {}
        for name, points in self._metrics.items():
            result[name] = [
                {
                    "value": p.value,
                    "timestamp": p.timestamp,
                    "labels": p.labels,
                    "metric_type": p.metric_type,
                }
                for p in points
            ]
        return result


class StreamingMetricsCollector:
    """Collect metrics in a background streaming loop."""

    def __init__(
        self,
        interval: float = 1.0,
        on_metric: Any | None = None,
    ) -> None:
        self.interval = interval
        self.on_metric = on_metric
        self._collector = MetricsCollector(interval)
        self._running = False
        self._task: asyncio.Task | None = None

    @property
    def collector(self) -> MetricsCollector:
        return self._collector

    async def start(self) -> None:
        """Start streaming collection."""
        self._running = True
        self._task = asyncio.create_task(self._stream_loop())
        logger.info(f"Streaming metrics collector started (interval={self.interval}s)")

    async def stop(self) -> None:
        """Stop streaming collection."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Streaming metrics collector stopped")

    async def _stream_loop(self) -> None:
        """Background streaming loop."""
        while self._running:
            try:
                if self.on_metric:
                    await self.on_metric(self._collector)
            except Exception as e:
                logger.error(f"Metric collection error: {e}")
            await asyncio.sleep(self.interval)


class MetricAggregator:
    """Aggregate metrics across multiple runs/configurations."""

    def __init__(self) -> None:
        self._aggregates: dict[str, dict[str, list[float]]] = {}

    def add(self, config_id: str, metric_name: str, value: float) -> None:
        """Add a metric value for a configuration."""
        if config_id not in self._aggregates:
            self._aggregates[config_id] = {}
        if metric_name not in self._aggregates[config_id]:
            self._aggregates[config_id][metric_name] = []
        self._aggregates[config_id][metric_name].append(value)

    def get_aggregate(
        self,
        config_id: str,
        metric_name: str,
    ) -> dict[str, float]:
        """Get aggregated statistics for a metric."""
        values = self._aggregates.get(config_id, {}).get(metric_name, [])
        if not values:
            return {"count": 0, "avg": 0, "min": 0, "max": 0, "p95": 0, "sum": 0}

        values.sort()
        total = sum(values)
        count = len(values)

        def percentile(data: list[float], p: float) -> float:
            if not data:
                return 0.0
            k = (len(data) - 1) * p
            f_idx = int(k)
            c_idx = f_idx + 1
            if c_idx >= len(data):
                return data[-1]
            return data[f_idx] + (k - f_idx) * (data[c_idx] - data[f_idx])

        return {
            "count": count,
            "avg": total / count,
            "min": values[0],
            "max": values[-1],
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "sum": total,
        }

    def get_all_aggregates(self) -> dict[str, dict[str, dict[str, float]]]:
        """Get all aggregated metrics."""
        result = {}
        for config_id, metrics in self._aggregates.items():
            result[config_id] = {}
            for metric_name, values in metrics.items():
                result[config_id][metric_name] = self.get_aggregate(config_id, metric_name)
        return result

    def compare_configs(
        self,
        config_ids: list[str],
        metric_name: str,
    ) -> dict[str, dict[str, float]]:
        """Compare a metric across multiple configurations."""
        return {
            cid: self.get_aggregate(cid, metric_name)
            for cid in config_ids
        }
