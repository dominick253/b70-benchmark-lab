"""
Docker container lifecycle management for BenchLab.

Handles pulling images, creating containers with GPU support,
waiting for readiness, collecting logs, and cleanup.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from benchlab.config import DockerConfig, ModelConfig
from benchlab.logging import get_logger
from benchlab.utils import generate_id, timestamp

logger = get_logger("docker_manager")


@dataclass
class ContainerInfo:
    """Information about a running container."""
    container_id: str
    name: str
    image: str
    status: str
    started_at: str
    port: int = 8000
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def api_url(self) -> str:
        return f"http://localhost:{self.port}/v1"


class DockerManager:
    """Manage Docker containers for benchmarking."""

    def __init__(self, config: DockerConfig) -> None:
        self.config = config
        self._containers: dict[str, ContainerInfo] = {}

    async def pull_image(self) -> bool:
        """Pull the Docker image."""
        image = f"{self.config.image}:{self.config.image_tag}"
        logger.info(f"Pulling Docker image: {image}")

        try:
            import docker as docker_sdk
            client = docker_sdk.from_env()

            if self.config.pull_image:
                client.images.pull(
                    self.config.image,
                    tag=self.config.image_tag,
                )

            logger.info(f"Image {image} ready")
            return True

        except ImportError:
            logger.warning("Docker SDK not available, using subprocess")
            return await self._pull_image_subprocess(image)
        except Exception as e:
            logger.error(f"Failed to pull image: {e}")
            return False

    async def _pull_image_subprocess(self, image: str) -> bool:
        """Pull image using subprocess."""
        try:
            process = await asyncio.create_subprocess_exec(
                "docker", "pull", image,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            if process.returncode == 0:
                logger.info(f"Image {image} pulled successfully")
                return True
            logger.error(f"Failed to pull image: {stderr.decode()}")
            return False
        except Exception as e:
            logger.error(f"Subprocess pull failed: {e}")
            return False

    async def create_container(
        self,
        model_config: ModelConfig,
        run_id: str,
        config_id: str,
    ) -> ContainerInfo | None:
        """Create a benchmarking container."""
        container_name = f"{self.config.container_name_prefix}-{generate_id()[:8]}"
        logger.info(f"Creating container: {container_name}")

        try:
            import docker as docker_sdk
            client = docker_sdk.from_env()

            env_vars = self.config.env_vars.copy()
            env_vars.update({
                "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
            })

            # Add model-specific env vars
            if model_config.quantization:
                env_vars["VLLM_DISTRIBUTED_executor_backend"] = model_config.quantization

            # Build command — entrypoint is `bash -c 'exec vllm serve "$@"' --`
            # So we pass model + options directly, NOT "vllm serve"
            model_path = model_config.path or model_config.name
            cmd = [
                model_path,
                "--tensor-parallel-size", str(model_config.tensor_parallel),
                "--dtype", model_config.dtype.value,
                "--gpu-memory-utilization", str(model_config.gpu_memory_utilization),
                "--max-model-len", str(model_config.max_model_len or 8192),
                "--block-size", str(model_config.block_size),
            ]

            if model_config.enable_prefix_caching.value == "on":
                cmd.append("--enable-prefix-caching")

            if model_config.graph_mode.value == "disabled":
                cmd.append("--enforce-eager")
            elif model_config.graph_mode.value == "full":
                cmd.extend(["--num-scheduler-steps", str(model_config.num_scheduler_steps)])

            if model_config.enable_chunked_prefill:
                cmd.append("--enable-chunked-prefill")
                if model_config.max_chunked_prefill_tokens:
                    cmd.extend(["--max-num-batched-tokens", str(model_config.max_chunked_prefill_tokens)])

            # Create container with Intel GPU support (/dev/dri mapping)
            # Also mount model directory and HF cache
            volumes = [
                "/dev/dri:/dev/dri:rw",
            ]
            
            # Mount model directory if using local model
            model_path = model_config.path or model_config.name
            if model_path and os.path.exists(model_path):
                volumes.append(f"{model_path}:{model_path}:ro")
            
            device_requests = None

            # Build port binding if using bridge network
            port_binding = None
            if self.config.network != "host":
                port_binding = {f"{self.config.port}/tcp": self.config.port}

            container = client.containers.run(
                f"{self.config.image}:{self.config.image_tag}",
                command=cmd,
                name=container_name,
                detach=True,
                environment=env_vars,
                mem_limit=self.config.memory,
                memswap_limit=self.config.memory_swap,
                cpu_count=int(float(self.config.cpus)) if self.config.cpus else None,
                shm_size=self.config.shm_size,
                network=self.config.network,
                restart_policy={"Name": self.config.restart_policy} if self.config.restart_policy != "no" else None,
                remove=self.config.remove_container,
                volumes=volumes,
                device_requests=device_requests,
                ports=port_binding,
            )

            info = ContainerInfo(
                container_id=container.id,
                name=container_name,
                image=f"{self.config.image}:{self.config.image_tag}",
                status="running",
                started_at=timestamp(),
            )

            self._containers[config_id] = info
            logger.info(f"Container {container_name} created: {container.id[:12]}")
            return info

        except ImportError:
            logger.warning("Docker SDK not available")
            return None
        except Exception as e:
            logger.error(f"Failed to create container: {e}")
            return None

    async def wait_for_ready(
        self,
        container_id: str,
        timeout: int = 120,
        interval: float = 2.0,
    ) -> bool:
        """Wait for container to be ready."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                import docker as docker_sdk
                client = docker_sdk.from_env()
                container = client.containers.get(container_id)

                if container.status != "running":
                    logger.error(f"Container {container_id} is {container.status}")
                    return False

                # Check API endpoint
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    try:
                        async with session.get(
                            f"http://localhost:8000/v1/models",
                            timeout=aiohttp.ClientTimeout(total=5),
                        ) as resp:
                            if resp.status == 200:
                                logger.info(f"Container {container_id} is ready")
                                return True
                    except Exception:
                        pass

            except Exception:
                pass

            await asyncio.sleep(interval)

        logger.error(f"Container {container_id} not ready after {timeout}s")
        return False

    async def get_container_logs(
        self,
        container_id: str,
        tail: int = 1000,
    ) -> str:
        """Get container logs."""
        try:
            import docker as docker_sdk
            client = docker_sdk.from_env()
            container = client.containers.get(container_id)
            return container.logs(tail=tail).decode("utf-8", errors="replace")
        except Exception as e:
            logger.error(f"Failed to get logs: {e}")
            return ""

    async def stop_container(self, container_id: str, timeout: int = 30) -> bool:
        """Stop a container gracefully."""
        try:
            import docker as docker_sdk
            client = docker_sdk.from_env()
            container = client.containers.get(container_id)

            if container.status == "running":
                container.stop(timeout=timeout)
                logger.info(f"Container {container_id} stopped")

            return True
        except Exception as e:
            logger.error(f"Failed to stop container: {e}")
            return False

    async def remove_container(self, container_id: str) -> bool:
        """Remove a container."""
        try:
            import docker as docker_sdk
            client = docker_sdk.from_env()
            container = client.containers.get(container_id)
            container.remove(force=True)
            logger.info(f"Container {container_id} removed")
            return True
        except Exception as e:
            logger.error(f"Failed to remove container: {e}")
            return False

    async def get_container_stats(
        self,
        container_id: str,
        one_shot: bool = True,
    ) -> dict[str, Any]:
        """Get container resource usage stats."""
        try:
            import docker as docker_sdk
            client = docker_sdk.from_env()
            container = client.containers.get(container_id)
            stats = container.stats(one_shot=one_shot)
            return {
                "memory_usage": stats.get("memory_stats", {}).get("usage", 0),
                "memory_limit": stats.get("memory_stats", {}).get("limit", 0),
                "cpu_usage": stats.get("cpu_stats", {}).get("cpu_usage", {}).get("total_usage", 0),
                "system_cpu_usage": stats.get("cpu_stats", {}).get("system_cpu_usage", 1),
                "network": stats.get("networks", {}),
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

    def get_container(self, container_id: str) -> ContainerInfo | None:
        """Get container info."""
        return self._containers.get(container_id)

    def list_containers(self) -> list[ContainerInfo]:
        """List all managed containers."""
        return list(self._containers.values())

    async def cleanup(self) -> None:
        """Clean up all managed containers."""
        for container_id, info in list(self._containers.items()):
            logger.info(f"Cleaning up container: {info.name}")
            await self.stop_container(container_id)
            await self.remove_container(container_id)

        self._containers.clear()
