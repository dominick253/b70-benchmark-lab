"""
vLLM metrics scraping and parsing.

Continuously scrapes /metrics endpoint, parses Prometheus-format
metrics, and maps them to structured data.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import aiohttp

from benchlab.logging import get_logger
from benchlab.metrics import MetricsCollector

logger = get_logger("vllm_metrics")


@dataclass
class VLLMMetricPoint:
    """A single vLLM metric snapshot."""
    timestamp: float
    running_num_requests: int = 0
    waiting_num_requests: int = 0
    gpu_cache_usage_perc: float = 0.0
    prefill_total_num_tokens: int = 0
    decode_total_num_tokens: int = 0
    prompt_tokens_total: int = 0
    generation_tokens_total: int = 0
    num_requests_running: int = 0
    num_requests_swapped: int = 0
    num_requests_waiting: int = 0
    avg_prompt_throughput_toks: float = 0.0
    avg_generation_throughput_toks: float = 0.0
    prefix_cache_hit_rate_toks: float = 0.0
    gpu_cache_usage_perc_sma: float = 0.0
    time_to_first_token_seconds_sum: float = 0.0
    time_to_first_token_seconds_count: int = 0
    time_per_output_token_seconds_sum: float = 0.0
    time_per_output_token_seconds_count: int = 0
    e2e_request_latency_seconds_sum: float = 0.0
    e2e_request_latency_seconds_count: int = 0
    request_params_max_tokens: int = 0
    num_completed_requests: int = 0
    num_errored_requests: int = 0
    avg_queue_time_seconds: float = 0.0
    p99_queue_time_seconds: float = 0.0
    raw_metrics: dict[str, float] = field(default_factory=dict)


class VLLMMetricsScraper:
    """Scrape and parse vLLM Prometheus metrics."""

    def __init__(
        self,
        endpoint: str = "http://localhost:8000/metrics",
        interval: float = 1.0,
    ) -> None:
        self.endpoint = endpoint
        self.interval = interval
        self._running = False
        self._task: asyncio.Task | None = None
        self._metrics: list[VLLMMetricPoint] = []

    async def start(self, metrics_collector: MetricsCollector) -> None:
        """Start scraping."""
        self._running = True
        self._task = asyncio.create_task(self._scrape_loop(metrics_collector))
        logger.info(f"vLLM metrics scraping started ({self.endpoint}/metrics)")

    async def stop(self) -> None:
        """Stop scraping."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _scrape_loop(self, metrics_collector: MetricsCollector) -> None:
        """Background scraping loop."""
        while self._running:
            try:
                data = await self._scrape_once()
                if data:
                    self._metrics.append(data)
                    metrics_collector.record("vllm_running_requests", data.num_requests_running)
                    metrics_collector.record("vllm_waiting_requests", data.num_requests_waiting)
                    metrics_collector.record("vllm_gpu_cache_usage", data.gpu_cache_usage_perc)
                    metrics_collector.record("vllm_prefill_tokens", data.prefill_total_num_tokens)
                    metrics_collector.record("vllm_decode_tokens", data.decode_total_num_tokens)
                    metrics_collector.record("vllm_prompt_throughput", data.avg_prompt_throughput_toks)
                    metrics_collector.record("vllm_decode_throughput", data.avg_generation_throughput_toks)
                    metrics_collector.record("vllm_cache_hit_rate", data.prefix_cache_hit_rate_toks)
                    metrics_collector.record("vllm_avg_queue_time", data.avg_queue_time_seconds)
            except Exception as e:
                logger.debug(f"vLLM metrics scrape error: {e}")
            await asyncio.sleep(self.interval)

    async def _scrape_once(self) -> VLLMMetricPoint | None:
        """Scrape metrics once."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.endpoint, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status != 200:
                        return None
                    text = await resp.text()
                    return self._parse_metrics(text)
        except Exception as e:
            logger.debug(f"vLLM scrape error: {e}")
            return None

    def _parse_metrics(self, text: str) -> VLLMMetricPoint:
        """Parse Prometheus-format metrics text."""
        now = time.time()
        parsed: dict[str, float] = {}

        for line in text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            match = re.match(r"^([a-zA-Z_:][\w:]*)(?:\{(.*)\})?\s+([\d.eE+\-]+)$", line)
            if match:
                name = match.group(1)
                value = float(match.group(3))
                parsed[name] = value

        # Map to structured fields
        point = VLLMMetricPoint(timestamp=now, raw_metrics=parsed)

        point.num_requests_running = int(parsed.get("num_requests_running", 0))
        point.num_requests_swapped = int(parsed.get("num_requests_swapped", 0))
        point.num_requests_waiting = int(parsed.get("num_requests_waiting", 0))
        point.gpu_cache_usage_perc = parsed.get("gpu_cache_usage_perc", 0)
        point.prefill_total_num_tokens = int(parsed.get("prefill_total_num_tokens", 0))
        point.decode_total_num_tokens = int(parsed.get("decode_total_num_tokens", 0))
        point.prompt_tokens_total = int(parsed.get("prompt_tokens_total", 0))
        point.generation_tokens_total = int(parsed.get("generation_tokens_total", 0))

        point.avg_prompt_throughput_toks = parsed.get("avg_prompt_throughput_toks", 0)
        point.avg_generation_throughput_toks = parsed.get("avg_generation_throughput_toks", 0)
        point.prefix_cache_hit_rate_toks = parsed.get("prefix_cache_hit_rate_toks", 0)
        point.gpu_cache_usage_perc_sma = parsed.get("gpu_cache_usage_perc_sma", 0)

        # TTFT
        ttft_count = parsed.get("time_to_first_token_seconds_count", 0)
        if ttft_count > 0:
            point.time_to_first_token_seconds_sum = parsed.get("time_to_first_token_seconds_sum", 0)
            point.time_to_first_token_seconds_count = int(ttft_count)

        # TPOT
        tpot_count = parsed.get("time_per_output_token_seconds_count", 0)
        if tpot_count > 0:
            point.time_per_output_token_seconds_sum = parsed.get("time_per_output_token_seconds_sum", 0)
            point.time_per_output_token_seconds_count = int(tpot_count)

        # E2E latency
        e2e_count = parsed.get("e2e_request_latency_seconds_count", 0)
        if e2e_count > 0:
            point.e2e_request_latency_seconds_sum = parsed.get("e2e_request_latency_seconds_sum", 0)
            point.e2e_request_latency_seconds_count = int(e2e_count)

        # Queue time
        point.avg_queue_time_seconds = parsed.get("avg_queue_time_seconds", 0)
        point.p99_queue_time_seconds = parsed.get("p99_queue_time_seconds", 0)

        point.num_completed_requests = int(parsed.get("num_completed_requests", 0))
        point.num_errored_requests = int(parsed.get("num_errored_requests", 0))

        return point

    def get_metrics(self) -> list[VLLMMetricPoint]:
        return self._metrics

    def get_latest(self) -> VLLMMetricPoint | None:
        return self._metrics[-1] if self._metrics else None

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all scraped metrics."""
        if not self._metrics:
            return {}

        running = [float(m.num_requests_running) for m in self._metrics]
        waiting = [float(m.num_requests_waiting) for m in self._metrics]
        cache = [float(m.gpu_cache_usage_perc) for m in self._metrics]
        prefill_tp = [float(m.avg_prompt_throughput_toks) for m in self._metrics]
        decode_tp = [float(m.avg_generation_throughput_toks) for m in self._metrics]

        def avg(data: list[float]) -> float:
            return sum(data) / len(data) if data else 0

        def pct(data: list[float], p: float) -> float:
            if not data:
                return 0
            s = sorted(data)
            k = (len(s) - 1) * p
            f = int(k)
            c = f + 1
            if c >= len(s):
                return s[-1]
            return s[f] + (k - f) * (s[c] - s[f])

        return {
            "count": len(self._metrics),
            "running_requests": {"avg": avg(running), "max": max(running) if running else 0},
            "waiting_requests": {"avg": avg(waiting), "max": max(waiting) if waiting else 0},
            "gpu_cache": {"avg": avg(cache), "max": max(cache) if cache else 0},
            "prompt_throughput": {"avg": avg(prefill_tp), "max": max(prefill_tp) if prefill_tp else 0},
            "decode_throughput": {"avg": avg(decode_tp), "max": max(decode_tp) if decode_tp else 0},
            "cache_hit_rate": {"avg": avg(cache), "max": max(cache) if cache else 0},
        }
