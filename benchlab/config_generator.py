"""
Configuration generation for BenchLab.

Generates all configuration combinations from parameter ranges,
assigns unique hashes, and produces Docker launch scripts.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from itertools import product as iter_product
from pathlib import Path
from typing import Any

from benchlab.config import (
    DockerConfig,
    GraphMode,
    ModelConfig,
    PrefixCaching,
)
from benchlab.logging import get_logger
from benchlab.utils import generate_id

logger = get_logger("config_generator")


@dataclass
class TestConfig:
    """A complete test configuration."""
    config_id: str
    config_hash: str
    name: str
    model: ModelConfig
    docker: DockerConfig
    gpu_memory_utilization: float
    block_size: int
    dtype: str
    kv_cache: str
    prefix_caching: str
    tensor_parallel: int
    graph_mode: str
    enable_chunked_prefill: bool
    max_chunked_prefill_tokens: int | None
    max_model_len: int
    num_scheduler_steps: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "config_hash": self.config_hash,
            "name": self.name,
            "model": self.model.to_dict(),
            "docker": self.docker.to_dict(),
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "block_size": self.block_size,
            "dtype": self.dtype,
            "kv_cache": self.kv_cache,
            "prefix_caching": self.prefix_caching,
            "tensor_parallel": self.tensor_parallel,
            "graph_mode": self.graph_mode,
            "enable_chunked_prefill": self.enable_chunked_prefill,
            "max_chunked_prefill_tokens": self.max_chunked_prefill_tokens,
            "max_model_len": self.max_model_len,
            "num_scheduler_steps": self.num_scheduler_steps,
            "metadata": self.metadata,
        }

    def to_yaml(self) -> str:
        """Serialize to YAML string."""
        return json.dumps(self.to_dict(), indent=2)

    def save(self, path: str) -> None:
        """Save configuration to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class ConfigGenerator:
    """Generate test configurations from parameter ranges."""

    GPU_MEM_VALUES = [0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.95]
    BLOCK_SIZES = [16, 32, 64, 128, 256]
    DTYPES = ["auto", "float16", "bfloat16"]
    KV_CACHE_TYPES = ["auto", "bfloat16", "float16"]
    PREFIX_CACHING = ["on", "off"]
    GRAPH_MODES = ["piecewise", "full", "eager", "disabled"]
    TENSOR_PARALLEL_VALUES = [1, 2]

    def __init__(
        self,
        model_config: ModelConfig,
        docker_config: DockerConfig,
        gpu_mem_values: list[float] | None = None,
        block_sizes: list[int] | None = None,
        dtypes: list[str] | None = None,
        kv_cache_types: list[str] | None = None,
        prefix_caching_values: list[str] | None = None,
        graph_modes: list[str] | None = None,
        tensor_parallel_values: list[int] | None = None,
        max_configs: int = 100,
    ) -> None:
        self.model_config = model_config
        self.docker_config = docker_config
        self.gpu_mem_values = gpu_mem_values or self.GPU_MEM_VALUES
        self.block_sizes = block_sizes or self.BLOCK_SIZES
        self.dtypes = dtypes or self.DTYPES
        self.kv_cache_types = kv_cache_types or self.KV_CACHE_TYPES
        self.prefix_caching_values = prefix_caching_values or self.PREFIX_CACHING
        self.graph_modes = graph_modes or self.GRAPH_MODES
        self.tensor_parallel_values = tensor_parallel_values or self.TENSOR_PARALLEL_VALUES
        self.max_configs = max_configs
        self._configs: list[TestConfig] = []

    def generate_all(self) -> list[TestConfig]:
        """Generate all configuration combinations."""
        self._configs = []
        config_count = 0

        for gpu_mem in self.gpu_mem_values:
            for block_size in self.block_sizes:
                for dtype in self.dtypes:
                    for kv_cache in self.kv_cache_types:
                        for prefix_caching in self.prefix_caching_values:
                            for graph_mode in self.graph_modes:
                                for tp in self.tensor_parallel_values:
                                    if config_count >= self.max_configs:
                                        logger.warning(f"Max configs ({self.max_configs}) reached")
                                        return self._configs

                                    config = self._build_config(
                                        gpu_mem, block_size, dtype, kv_cache,
                                        prefix_caching, graph_mode, tp,
                                    )
                                    self._configs.append(config)
                                    config_count += 1

        logger.info(f"Generated {len(self._configs)} configurations")
        return self._configs

    def _build_config(
        self,
        gpu_mem: float,
        block_size: int,
        dtype: str,
        kv_cache: str,
        prefix_caching: str,
        graph_mode: str,
        tp: int,
    ) -> TestConfig:
        """Build a single test configuration."""
        config_id = f"config-{generate_id()[:8]}"

        # Generate hash from parameters
        hash_input = "|".join([
            str(gpu_mem),
            str(block_size),
            dtype,
            kv_cache,
            prefix_caching,
            graph_mode,
            str(tp),
            self.model_config.name,
        ])
        config_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        name = f"{self.model_config.name.split('/')[-1]}-gpu{int(gpu_mem*100)}-"
        name += f"bs{block_size}-{dtype}-{kv_cache}-{prefix_caching[:2]}-{graph_mode[:3]}-tp{tp}"

        return TestConfig(
            config_id=config_id,
            config_hash=config_hash,
            name=name,
            model=self.model_config,
            docker=self.docker_config,
            gpu_memory_utilization=gpu_mem,
            block_size=block_size,
            dtype=dtype,
            kv_cache=kv_cache,
            prefix_caching=prefix_caching,
            tensor_parallel=tp,
            graph_mode=graph_mode,
            enable_chunked_prefill=True,
            max_chunked_prefill_tokens=4096,
            max_model_len=8192,
            num_scheduler_steps=4,
            metadata={"generated_at": __import__("time").time()},
        )

    def generate_docker_compose(
        self,
        config: TestConfig,
        output_dir: str = "configs",
    ) -> str:
        """Generate a Docker Compose YAML for a configuration."""
        compose = {
            "version": "3.8",
            "services": {
                "benchlab": {
                    "image": f"{self.docker_config.image}:{self.docker_config.image_tag}",
                    "container_name": config.config_id,
                    "runtime": "docker",
                    "devices": ["/dev/dri:/dev/dri:rw"],
                    "environment": {
                        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
                    },
                    "command": self._build_command(config),
                    "mem_limit": self.docker_config.memory,
                    "memswap_limit": self.docker_config.memory_swap,
                    "shm_size": self.docker_config.shm_size,
                    "restart": "no",
                },
            },
        }

        path = os.path.join(output_dir, f"compose-{config.config_hash}.yml")
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        import yaml
        with open(path, "w") as f:
            yaml.dump(compose, f, default_flow_style=False)

        return path

    def _build_command(self, config: TestConfig) -> list[str]:
        """Build the vLLM command for a configuration."""
        model_path = config.model.path or config.model.name
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--tensor-parallel-size", str(config.tensor_parallel),
            "--dtype", config.dtype,
            "--gpu-memory-utilization", str(config.gpu_memory_utilization),
            "--max-model-len", str(config.max_model_len),
            "--block-size", str(config.block_size),
        ]

        if config.prefix_caching == "on":
            cmd.append("--enable-prefix-caching")
        else:
            cmd.append("--disable-prefix-caching")

        if config.graph_mode == "disabled":
            cmd.append("--enforce-eager")
        elif config.graph_mode == "full":
            cmd.extend(["--num-scheduler-steps", str(config.num_scheduler_steps)])

        if config.enable_chunked_prefill:
            cmd.append("--enable-chunked-prefill")
            if config.max_chunked_prefill_tokens:
                cmd.extend(["--max-num-batched-tokens", str(config.max_chunked_prefill_tokens)])

        return cmd

    def get_configs(self) -> list[TestConfig]:
        """Get all generated configurations."""
        return self._configs

    def get_config_by_hash(self, config_hash: str) -> TestConfig | None:
        """Get a configuration by hash."""
        for config in self._configs:
            if config.config_hash == config_hash:
                return config
        return None

    def export_all(self, output_dir: str = "configs") -> list[str]:
        """Export all configurations to JSON files."""
        paths: list[str] = []
        for config in self._configs:
            path = os.path.join(output_dir, f"config-{config.config_hash}.json")
            config.save(path)
            paths.append(path)
        return paths
