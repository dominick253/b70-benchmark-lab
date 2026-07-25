"""
Configuration matrix generation and benchmark scheduling.

Automatically generates all configuration combinations, assigns
unique hashes, and schedules benchmarks for execution.
"""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from benchlab.config import (
    BenchLabConfig,
    Dtype,
    GraphMode,
    KVCacheDtype,
    ModelConfig,
    PrefixCaching,
    WorkloadConfig,
    WorkloadLength,
    WorkloadType,
)
from benchlab.logging import get_logger
from benchlab.utils import generate_id, timestamp

logger = get_logger("scheduler")


@dataclass
class ConfigCombination:
    """A single configuration combination to benchmark."""
    id: str
    run_id: str
    config_hash: str
    config: dict[str, Any]
    docker_config: dict[str, Any]
    workload_type: str
    workload_length: str
    priority: int = 0
    status: str = "pending"
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "config": self.config,
            "docker_config": self.docker_config,
            "workload_type": self.workload_type,
            "workload_length": self.workload_length,
            "priority": self.priority,
            "status": self.status,
            "score": self.score,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfigCombination":
        return cls(**data)


@dataclass
class BenchmarkTask:
    """A scheduled benchmark task."""
    task_id: str
    config: ConfigCombination
    workload_type: str
    workload_length: str
    num_prompts: int
    concurrency: int
    stream: bool
    status: str = "pending"
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None


class ConfigMatrixGenerator:
    """Generate all configuration combinations."""

    def __init__(self, config: BenchLabConfig) -> None:
        self.config = config
        self._combinations: list[ConfigCombination] = []

    def generate(self) -> list[ConfigCombination]:
        """Generate all configuration combinations."""
        logger.info("Generating configuration matrix...")

        self._combinations = []

        # Generate GPU memory utilization variants
        self._generate_memory_variants()

        # Generate block size variants
        self._generate_block_size_variants()

        # Generate dtype variants
        self._generate_dtype_variants()

        # Generate KV cache dtype variants
        self._generate_kv_cache_variants()

        # Generate prefix caching variants
        self._generate_prefix_caching_variants()

        # Generate graph mode variants
        self._generate_graph_mode_variants()

        # Generate tensor parallel variants
        self._generate_tensor_parallel_variants()

        # Generate max_model_len variants
        self._generate_model_len_variants()

        # Generate quantization variants
        self._generate_quantization_variants()

        # Assign priorities
        for i, combo in enumerate(self._combinations):
            combo.priority = i

        logger.info(f"Generated {len(self._combinations)} configuration combinations")
        return self._combinations

    def _make_config(
        self,
        model_overrides: dict[str, Any],
        docker_overrides: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ConfigCombination:
        """Create a configuration combination."""
        model_config = self.config.model.model_dump()
        model_config.update(model_overrides)

        docker_config = self.config.docker.model_dump()
        if docker_overrides:
            docker_config.update(docker_overrides)

        config_hash = self.config.compute_config_hash(model_config)
        combo_id = generate_id()

        return ConfigCombination(
            id=combo_id,
            run_id=self.config.run_name,
            config_hash=config_hash,
            config=model_config,
            docker_config=docker_config,
            workload_type=self.config.workload.type.value,
            workload_length=self.config.workload.length.value,
            metadata=metadata or {},
        )

    def _generate_memory_variants(self) -> None:
        """Generate GPU memory utilization variants."""
        for gpu_util in [0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]:
            self._combinations.append(self._make_config(
                model_overrides={"gpu_memory_utilization": gpu_util},
                metadata={"variant": "gpu_memory_utilization", "value": gpu_util},
            ))

    def _generate_block_size_variants(self) -> None:
        """Generate block size variants."""
        for block_size in [16, 32, 64, 128, 256]:
            self._combinations.append(self._make_config(
                model_overrides={"block_size": block_size},
                metadata={"variant": "block_size", "value": block_size},
            ))

    def _generate_dtype_variants(self) -> None:
        """Generate dtype variants."""
        for dtype in [Dtype.AUTO, Dtype.FLOAT16, Dtype.BFLOAT16]:
            self._combinations.append(self._make_config(
                model_overrides={"dtype": dtype.value},
                metadata={"variant": "dtype", "value": dtype.value},
            ))

    def _generate_kv_cache_variants(self) -> None:
        """Generate KV cache dtype variants."""
        for kv_dtype in [KVCacheDtype.AUTO, KVCacheDtype.BFLOAT16, KVCacheDtype.FLOAT16]:
            self._combinations.append(self._make_config(
                model_overrides={"kv_cache_dtype": kv_dtype.value},
                metadata={"variant": "kv_cache_dtype", "value": kv_dtype.value},
            ))

    def _generate_prefix_caching_variants(self) -> None:
        """Generate prefix caching variants."""
        for prefix_cache in [PrefixCaching.ON, PrefixCaching.OFF]:
            self._combinations.append(self._make_config(
                model_overrides={"enable_prefix_caching": prefix_cache.value},
                metadata={"variant": "prefix_caching", "value": prefix_cache.value},
            ))

    def _generate_graph_mode_variants(self) -> None:
        """Generate graph mode variants."""
        for graph_mode in [GraphMode.PIECEWISE, GraphMode.FULL, GraphMode.EAGER, GraphMode.DISABLED]:
            self._combinations.append(self._make_config(
                model_overrides={
                    "graph_mode": graph_mode.value,
                    "enforce_eager": graph_mode == GraphMode.DISABLED or graph_mode == GraphMode.EAGER,
                },
                metadata={"variant": "graph_mode", "value": graph_mode.value},
            ))

    def _generate_tensor_parallel_variants(self) -> None:
        """Generate tensor parallel variants."""
        for tp in [1, 2]:
            self._combinations.append(self._make_config(
                model_overrides={"tensor_parallel": tp},
                metadata={"variant": "tensor_parallel", "value": tp},
            ))

    def _generate_model_len_variants(self) -> None:
        """Generate max model length variants."""
        for max_len in [4096, 8192, 16384, 32768]:
            self._combinations.append(self._make_config(
                model_overrides={"max_model_len": max_len},
                metadata={"variant": "max_model_len", "value": max_len},
            ))

    def _generate_quantization_variants(self) -> None:
        """Generate quantization variants."""
        for quant in [None, "awq", "gptq", "bitsandbytes"]:
            if quant is not None:
                self._combinations.append(self._make_config(
                    model_overrides={"quantization": quant},
                    metadata={"variant": "quantization", "value": quant},
                ))

    def get_combinations(self) -> list[ConfigCombination]:
        """Get generated combinations."""
        return self._combinations

    def filter_by_variant(self, variant: str) -> list[ConfigCombination]:
        """Filter combinations by variant type."""
        return [c for c in self._combinations if c.metadata.get("variant") == variant]

    def filter_by_status(self, status: str) -> list[ConfigCombination]:
        """Filter combinations by status."""
        return [c for c in self._combinations if c.status == status]

    def save_to_file(self, path: str | Path) -> Path:
        """Save all combinations to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "run_id": self.config.run_name,
            "total": len(self._combinations),
            "combinations": [c.to_dict() for c in self._combinations],
        }

        path.write_text(json.dumps(data, indent=2, default=str))
        return path

    def load_from_file(self, path: str | Path) -> list[ConfigCombination]:
        """Load combinations from a JSON file."""
        path = Path(path)
        data = json.loads(path.read_text())
        self._combinations = [ConfigCombination.from_dict(c) for c in data["combinations"]]
        return self._combinations


class BenchmarkScheduler:
    """Schedule and manage benchmark tasks."""

    def __init__(self, config: BenchLabConfig) -> None:
        self.config = config
        self._tasks: list[BenchmarkTask] = []
        self._completed: list[BenchmarkTask] = []

    def create_tasks(
        self,
        combinations: list[ConfigCombination],
        workload_types: list[WorkloadType] | None = None,
        workload_lengths: list[WorkloadLength] | None = None,
    ) -> list[BenchmarkTask]:
        """Create benchmark tasks from combinations."""
        if workload_types is None:
            workload_types = [WorkloadType.CODING, WorkloadType.REASONING, WorkloadType.CREATIVE_WRITING]
        if workload_lengths is None:
            workload_lengths = [WorkloadLength.SHORT, WorkloadLength.MEDIUM, WorkloadLength.LONG]

        self._tasks = []

        for combo in combinations:
            for wt in workload_types:
                for wl in workload_lengths:
                    task = BenchmarkTask(
                        task_id=generate_id(),
                        config=combo,
                        workload_type=wt.value,
                        workload_length=wl.value,
                        num_prompts=self.config.workload.num_prompts,
                        concurrency=self.config.workload.concurrency,
                        stream=self.config.workload.stream,
                    )
                    self._tasks.append(task)

        logger.info(f"Created {len(self._tasks)} benchmark tasks")
        return self._tasks

    def get_pending_tasks(self, limit: int = 1) -> list[BenchmarkTask]:
        """Get pending tasks."""
        return [t for t in self._tasks if t.status == "pending"][:limit]

    def mark_completed(self, task: BenchmarkTask, result: dict[str, Any] | None = None) -> None:
        """Mark a task as completed."""
        task.status = "completed"
        task.completed_at = timestamp()
        task.result = result
        self._completed.append(task)

    def mark_failed(self, task: BenchmarkTask, error: str) -> None:
        """Mark a task as failed."""
        task.status = "failed"
        task.completed_at = timestamp()
        task.error = error

    def mark_started(self, task: BenchmarkTask) -> None:
        """Mark a task as started."""
        task.status = "running"
        task.started_at = timestamp()

    def get_progress(self) -> dict[str, int]:
        """Get scheduling progress."""
        status_counts: dict[str, int] = {"pending": 0, "running": 0, "completed": 0, "failed": 0}
        for task in self._tasks:
            status_counts[task.status] = status_counts.get(task.status, 0) + 1
        return status_counts

    def get_completed_tasks(self) -> list[BenchmarkTask]:
        """Get all completed tasks."""
        return self._completed

    def get_tasks_by_config(self, config_id: str) -> list[BenchmarkTask]:
        """Get all tasks for a specific configuration."""
        return [t for t in self._tasks if t.config.id == config_id]

    def get_tasks_by_workload(self, workload_type: str) -> list[BenchmarkTask]:
        """Get all tasks for a specific workload type."""
        return [t for t in self._tasks if t.workload_type == workload_type]

    def export_tasks(self, path: str | Path) -> Path:
        """Export task list to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "total": len(self._tasks),
            "progress": self.get_progress(),
            "tasks": [
                {
                    "task_id": t.task_id,
                    "config_id": t.config.id,
                    "config_hash": t.config.config_hash,
                    "workload_type": t.workload_type,
                    "workload_length": t.workload_length,
                    "status": t.status,
                    "started_at": t.started_at,
                    "completed_at": t.completed_at,
                    "error": t.error,
                }
                for t in self._tasks
            ],
        }

        path.write_text(json.dumps(data, indent=2, default=str))
        return path
