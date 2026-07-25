"""
Core benchmark orchestration for BenchLab.

Coordinates Docker container lifecycle, workload execution,
telemetry collection, and result aggregation for each
configuration under test.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from benchlab.config import BenchLabConfig, DockerConfig, ModelConfig, WorkloadConfig
from benchlab.logging import get_logger
from benchlab.prompts import PromptGenerator
from benchlab.storage import Storage
from benchlab.utils import generate_id, timestamp

logger = get_logger("benchmark")


@dataclass
class BenchmarkResult:
    """Complete result for a single benchmark run."""
    benchmark_id: str
    config_id: str
    run_id: str
    config_hash: str
    workload_type: str
    workload_length: str
    start_time: str
    end_time: str
    duration: float
    status: str = "completed"
    error: str | None = None

    # Workload metrics
    avg_ttft: float = 0.0
    p95_ttft: float = 0.0
    p99_ttft: float = 0.0
    avg_tps: float = 0.0
    p95_tps: float = 0.0
    p99_tps: float = 0.0
    avg_total_tps: float = 0.0
    avg_decode_time: float = 0.0
    avg_total_time: float = 0.0
    success_rate: float = 1.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    # GPU metrics
    avg_gpu_utilization: float = 0.0
    peak_gpu_utilization: float = 0.0
    avg_gpu_memory: float = 0.0
    peak_gpu_memory: float = 0.0
    avg_power: float = 0.0
    peak_power: float = 0.0
    avg_temperature: float = 0.0
    peak_temperature: float = 0.0

    # System metrics
    avg_cpu_utilization: float = 0.0
    avg_ram_utilization: float = 0.0

    # Docker metrics
    container_memory_usage: float = 0.0
    container_cpu_usage: float = 0.0

    # Corruption
    corruption_count: int = 0
    corruption_details: list[dict[str, Any]] = field(default_factory=list)

    # Raw data
    request_metrics: list[dict[str, Any]] = field(default_factory=list)
    telemetry_data: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "config_id": self.config_id,
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "workload_type": self.workload_type,
            "workload_length": self.workload_length,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "status": self.status,
            "error": self.error,
            "avg_ttft": self.avg_ttft,
            "p95_ttft": self.p95_ttft,
            "p99_ttft": self.p99_ttft,
            "avg_tps": self.avg_tps,
            "p95_tps": self.p95_tps,
            "p99_tps": self.p99_tps,
            "avg_total_tps": self.avg_total_tps,
            "avg_decode_time": self.avg_decode_time,
            "avg_total_time": self.avg_total_time,
            "success_rate": self.success_rate,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "avg_gpu_utilization": self.avg_gpu_utilization,
            "peak_gpu_utilization": self.peak_gpu_utilization,
            "avg_gpu_memory": self.avg_gpu_memory,
            "peak_gpu_memory": self.peak_gpu_memory,
            "avg_power": self.avg_power,
            "peak_power": self.peak_power,
            "avg_temperature": self.avg_temperature,
            "peak_temperature": self.peak_temperature,
            "avg_cpu_utilization": self.avg_cpu_utilization,
            "avg_ram_utilization": self.avg_ram_utilization,
            "container_memory_usage": self.container_memory_usage,
            "container_cpu_usage": self.container_cpu_usage,
            "corruption_count": self.corruption_count,
            "corruption_details": self.corruption_details,
            "metadata": self.metadata,
        }


class BenchmarkRunner:
    """Run a single benchmark for a configuration."""

    def __init__(
        self,
        config: BenchLabConfig,
        storage: Storage,
        model_config: ModelConfig,
        docker_config: DockerConfig,
        workload_config: WorkloadConfig,
    ) -> None:
        self.config = config
        self.storage = storage
        self.model_config = model_config
        self.docker_config = docker_config
        self.workload_config = workload_config
        self._prompt_generator = PromptGenerator(seed=workload_config.seed)

    async def run(
        self,
        benchmark_id: str,
        config_id: str,
        run_id: str,
        config_hash: str,
        workload_type: str,
        workload_length: str,
    ) -> BenchmarkResult:
        """Run a single benchmark."""
        start_time = timestamp()
        start_epoch = time.time()

        result = BenchmarkResult(
            benchmark_id=benchmark_id,
            config_id=config_id,
            run_id=run_id,
            config_hash=config_hash,
            workload_type=workload_type,
            workload_length=workload_length,
            start_time=start_time,
            end_time="",
            duration=0,
        )

        try:
            logger.info(f"Starting benchmark {benchmark_id}")

            # TODO: Docker container lifecycle
            # 1. Pull image
            # 2. Create container with config
            # 3. Wait for readiness
            # 4. Start telemetry collection
            # 5. Run workload
            # 6. Stop telemetry
            # 7. Collect container logs
            # 8. Stop and remove container

            # For now, placeholder telemetry
            result.status = "completed"

            end_time = timestamp()
            result.end_time = end_time
            result.duration = time.time() - start_epoch

            logger.info(f"Completed benchmark {benchmark_id} in {result.duration:.1f}s")
            return result

        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            result.end_time = timestamp()
            result.duration = time.time() - start_epoch
            logger.error(f"Benchmark {benchmark_id} failed: {e}", exc_info=True)
            return result
