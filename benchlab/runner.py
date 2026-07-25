"""
Benchmark runner for BenchLab.

Orchestrates the full benchmark lifecycle:
1. System validation
2. Configuration generation
3. Docker container management
4. Benchmark execution
5. Telemetry collection
6. Corruption detection
7. Scoring and ranking
8. Report generation
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from benchlab.config import BenchLabConfig, ModelConfig, DockerConfig
from benchlab.config_generator import ConfigGenerator
from benchlab.corruption import CorruptionDetector
from benchlab.docker_manager import DockerManager
from benchlab.intel_gpu import IntelGPUCollector
from benchlab.logging import get_logger
from benchlab.metrics import MetricsCollector, MetricAggregator
from benchlab.config import WorkloadLength, WorkloadType
from benchlab.prompts import PromptGenerator
from benchlab.scoring import ScoringEngine
from benchlab.storage import Storage
from benchlab.telemetry import TelemetryManager
from benchlab.vllm_metrics import VLLMMetricsScraper
from benchlab.workload import WorkloadExecutor
from benchlab.validators import ConfigValidator, OutputValidator, SystemValidator
from benchlab.utils import generate_id

logger = get_logger("runner")


@dataclass
class BenchmarkResult:
    """Complete result for a single benchmark run."""
    run_id: str
    config_hash: str
    config_name: str
    status: str  # success, failed, corrupted, crashed
    start_time: float
    end_time: float
    duration: float
    metrics: dict[str, Any] = field(default_factory=dict)
    corruption_events: list[dict[str, Any]] = field(default_factory=list)
    gpu_metrics: dict[str, Any] = field(default_factory=dict)
    system_metrics: dict[str, Any] = field(default_factory=dict)
    vllm_metrics: dict[str, Any] = field(default_factory=dict)
    docker_stats: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "config_name": self.config_name,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "metrics": self.metrics,
            "corruption_events": self.corruption_events,
            "gpu_metrics": self.gpu_metrics,
            "system_metrics": self.system_metrics,
            "vllm_metrics": self.vllm_metrics,
            "docker_stats": self.docker_stats,
            "score": self.score,
            "error": self.error,
        }


class BenchmarkRunner:
    """Orchestrate the full benchmark lifecycle."""

    def __init__(self, config: BenchLabConfig) -> None:
        self.config = config
        self.storage = Storage(config.db_path)
        self.docker_manager = DockerManager(config.docker)
        self.telemetry = TelemetryManager(config.telemetry.collection_interval)
        self.gpu_collector = IntelGPUCollector()
        self.vllm_scraper = VLLMMetricsScraper(
            endpoint=f"http://localhost:{config.docker.port}",
            interval=config.telemetry.collection_interval,
        )
        self.config_generator = ConfigGenerator(
            model_config=config.model,
            docker_config=config.docker,
            max_configs=config.max_configs,
        )
        self.scoring_engine = ScoringEngine(config.scoring)
        self.corruption_detector = CorruptionDetector(config.corruption)
        self.prompt_generator = PromptGenerator()
        self.metrics_collector = MetricsCollector()
        self.metrics_aggregator = MetricAggregator()
        self._results: list[BenchmarkResult] = []
        self._running = False
        self._on_result: Callable[[BenchmarkResult], None] | None = None

    def set_on_result(self, callback: Callable[[BenchmarkResult], None]) -> None:
        """Set callback for each completed benchmark."""
        self._on_result = callback

    async def run_all(self) -> list[BenchmarkResult]:
        """Run all benchmarks in the configuration matrix."""
        self._running = True
        start_time = time.time()

        logger.info("=== B70 Benchmark Lab Starting ===")

        # Phase 1: System validation
        logger.info("Phase 1: System validation")
        system_check = SystemValidator.run_full_check()
        if not system_check["all_passed"]:
            logger.warning("System checks not fully passed:")
            for name, result in system_check["checks"].items():
                if not result["passed"]:
                    logger.warning(f"  {name}: {result['message']}")

        # Phase 2: Generate configurations
        logger.info("Phase 2: Generating configurations")
        configs = self.config_generator.generate_all()
        logger.info(f"Generated {len(configs)} configurations")

        # Phase 3: Pull Docker image
        logger.info("Phase 3: Pulling Docker image")
        await self.docker_manager.pull_image()

        # Phase 4: Run benchmarks
        logger.info("Phase 4: Running benchmarks")
        await self._run_benchmarks(configs)

        # Phase 5: Score and rank
        logger.info("Phase 5: Scoring and ranking")
        scores = self.scoring_engine.rank(
            [self._score_result(r) for r in self._results]
        )

        # Phase 6: Generate reports
        logger.info("Phase 6: Generating reports")
        await self._generate_reports(scores, start_time)

        # Phase 7: Cleanup
        logger.info("Phase 7: Cleanup")
        await self.telemetry.stop()
        await self.docker_manager.cleanup()

        logger.info(f"=== Benchmark Lab Complete ({time.time() - start_time:.1f}s) ===")
        return self._results

    async def _run_benchmarks(self, configs: list[Any]) -> None:
        """Run benchmarks for each configuration."""
        for i, config in enumerate(configs):
            if not self._running:
                logger.info("Benchmark run cancelled")
                break

            logger.info(f"[{i+1}/{len(configs)}] Running {config.name}")

            try:
                result = await self._run_single(config)
                self._results.append(result)

                # Save to storage
                await self.storage.save_run(result.to_dict())

                # Notify callback
                if self._on_result:
                    self._on_result(result)

            except Exception as e:
                logger.error(f"Benchmark failed: {e}")
                error_result = BenchmarkResult(
                    run_id=f"run-{i+1}",
                    config_hash=config.config_hash,
                    config_name=config.name,
                    status="crashed",
                    start_time=time.time(),
                    end_time=time.time(),
                    duration=0,
                    error=str(e),
                )
                self._results.append(error_result)

    async def _run_single(self, config: Any) -> BenchmarkResult:
        """Run a single benchmark configuration."""
        run_id = f"run-{config.config_hash}"
        start_time = time.time()

        # Start telemetry
        await self.telemetry.start()

        # Create container
        container = await self.docker_manager.create_container(
            model_config=self.config.model,
            run_id=run_id,
            config_id=config.config_hash,
        )

        if not container:
            return BenchmarkResult(
                run_id=run_id,
                config_hash=config.config_hash,
                config_name=config.name,
                status="failed",
                start_time=start_time,
                end_time=time.time(),
                duration=0,
                error="Failed to create container",
            )

        # Wait for readiness
        ready = await self.docker_manager.wait_for_ready(container.container_id)
        if not ready:
            await self.docker_manager.stop_container(container.container_id)
            return BenchmarkResult(
                run_id=run_id,
                config_hash=config.config_hash,
                config_name=config.name,
                status="failed",
                start_time=start_time,
                end_time=time.time(),
                duration=0,
                error="Container not ready",
            )

        # Start vLLM metrics scraping
        await self.vllm_scraper.start(self.metrics_collector)

        # Run workloads
        workload_config = self.config.workload
        workload_config.concurrency = self.config.workload.concurrency
        executor = WorkloadExecutor(
            api_url=f"http://localhost:{config.docker.port}/v1",
            model=self.config.model.name,
            config=workload_config,
        )

        workloads = [
            (WorkloadType.REASONING, WorkloadLength.SHORT),
            (WorkloadType.CODING, WorkloadLength.SHORT),
            (WorkloadType.CREATIVE_WRITING, WorkloadLength.MEDIUM),
            (WorkloadType.MATH, WorkloadLength.SHORT),
            (WorkloadType.TOOL_CALLING, WorkloadLength.SHORT),
        ]

        benchmark_metrics = {}
        corruption_events = []

        for workload_type, length in workloads:
            if not self._running:
                break

            workload_result = await executor.run_workload(
                workload_type=workload_type,
                length=length,
            )

            if workload_result:
                # Flatten WorkloadResult fields into benchmark_metrics
                benchmark_metrics[f"{workload_type.value}_{length.value}_num_requests"] = workload_result.num_requests
                benchmark_metrics[f"{workload_type.value}_{length.value}_successful"] = workload_result.successful_requests
                benchmark_metrics[f"{workload_type.value}_{length.value}_failed"] = workload_result.failed_requests
                benchmark_metrics[f"{workload_type.value}_{length.value}_avg_ttft"] = workload_result.avg_ttft
                benchmark_metrics[f"{workload_type.value}_{length.value}_p95_ttft"] = workload_result.p95_ttft
                benchmark_metrics[f"{workload_type.value}_{length.value}_avg_tps"] = workload_result.avg_tps
                benchmark_metrics[f"{workload_type.value}_{length.value}_p95_tps"] = workload_result.p95_tps
                benchmark_metrics[f"{workload_type.value}_{length.value}_avg_total_tps"] = workload_result.avg_total_tps
                benchmark_metrics[f"{workload_type.value}_{length.value}_success_rate"] = workload_result.success_rate
                benchmark_metrics[f"{workload_type.value}_{length.value}_total_tokens"] = workload_result.total_tokens
                benchmark_metrics[f"{workload_type.value}_{length.value}_total_time"] = workload_result.total_time

                # Check for corruption
                for req in workload_result.requests:
                    events = self.corruption_detector.check(
                        prompt=req.prompt,
                        output="",
                        context={"workload": workload_type.value, "length": length.value},
                    )
                    corruption_events.extend(events)

        # Collect final metrics
        duration = time.time() - start_time
        gpu_summary = self.gpu_collector.get_summary()
        system_summary = {}
        vllm_summary = self.vllm_scraper.get_summary()
        corruption_summary = self.corruption_detector.get_summary()

        status = "corrupted" if corruption_events else "success"

        result = BenchmarkResult(
            run_id=run_id,
            config_hash=config.config_hash,
            config_name=config.name,
            status=status,
            start_time=start_time,
            end_time=time.time(),
            duration=duration,
            metrics=benchmark_metrics,
            corruption_events=corruption_events,
            gpu_metrics=gpu_summary,
            vllm_metrics=vllm_summary,
            score=0,
        )

        # Cleanup
        await self.telemetry.stop()
        await self.vllm_scraper.stop()
        await self.docker_manager.stop_container(container.container_id)
        await self.docker_manager.remove_container(container.container_id)

        self.metrics_collector.clear()
        self.corruption_detector.clear()

        return result

    def _score_result(self, result: BenchmarkResult) -> Any:
        """Score a benchmark result."""
        return self.scoring_engine.score(
            config_id=result.config_hash,
            config_hash=result.config_hash,
            metrics=result.metrics,
        )

    async def _generate_reports(self, scores: list[Any], start_time: float) -> None:
        """Generate all reports."""
        from benchlab.report import ReportGenerator
        from benchlab.html_report import HTMLReportGenerator

        report_gen = ReportGenerator(self.config)
        await report_gen.generate_all(self._results, scores)

        html_gen = HTMLReportGenerator(self.config)
        await html_gen.generate(self._results, scores)

        logger.info(f"Reports generated in {self.config.reports_dir}")

    def get_results(self) -> list[BenchmarkResult]:
        """Get all benchmark results."""
        return self._results

    def get_best_config(self) -> BenchmarkResult | None:
        """Get the best performing configuration."""
        if not self._results:
            return None
        return max(self._results, key=lambda r: r.metrics.get("avg_tps", 0))

    def get_most_stable_config(self) -> BenchmarkResult | None:
        """Get the most stable configuration."""
        success_results = [r for r in self._results if r.status == "success"]
        if not success_results:
            return None
        return min(
            success_results,
            key=lambda r: r.metrics.get("stddev_tps", 0),
        )

    async def stop(self) -> None:
        """Stop all running benchmarks."""
        self._running = False
        await self.telemetry.stop()
        await self.docker_manager.cleanup()
