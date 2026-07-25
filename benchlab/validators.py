"""
Input validation for BenchLab.

Validates configurations, outputs, and system state
before and during benchmark execution.
"""

from __future__ import annotations

import re
from typing import Any

from benchlab.config import ModelConfig, DockerConfig
from benchlab.logging import get_logger

logger = get_logger("validators")


class ConfigValidator:
    """Validate benchmark configurations."""

    @staticmethod
    def validate_model_config(config: ModelConfig) -> list[str]:
        """Validate a model configuration."""
        errors: list[str] = []

        if not config.name:
            errors.append("Model name is required")
        elif not config.path and not config.name:
            errors.append("Either model name or path must be specified")

        if config.dtype not in ("auto", "float16", "bfloat16", "float", "half"):
            errors.append(f"Invalid dtype: {config.dtype}")

        if not (0.0 < config.gpu_memory_utilization <= 1.0):
            errors.append(f"gpu_memory_utilization must be between 0 and 1, got {config.gpu_memory_utilization}")

        if config.block_size not in (16, 32, 64, 128, 256):
            errors.append(f"Invalid block_size: {config.block_size}")

        if config.tensor_parallel < 1:
            errors.append(f"tensor_parallel must be >= 1, got {config.tensor_parallel}")

        if config.max_model_len < 128:
            errors.append(f"max_model_len must be >= 128, got {config.max_model_len}")

        if config.quantization and config.quantization not in (
            "awq", "bitsandbytes", "gptq", "compressed-tensors", "fp8", "fp8_e4m3", "fp8_e5m2",
        ):
            errors.append(f"Invalid quantization: {config.quantization}")

        return errors

    @staticmethod
    def validate_docker_config(config: DockerConfig) -> list[str]:
        """Validate a Docker configuration."""
        errors: list[str] = []

        if config.gpus < 1:
            errors.append("gpus must be >= 1")

        if config.gpus > 8:
            errors.append("gpus must be <= 8")

        if config.cpus < 1:
            errors.append("cpus must be >= 1")

        if config.memory and config.memory.endswith("G"):
            mem_gb = int(config.memory[:-1])
            if mem_gb < 4:
                errors.append("memory should be >= 4G")

        if config.shm_size and config.shm_size.endswith("G"):
            shm_gb = int(config.shm_size[:-1])
            if shm_gb < 2:
                errors.append("shm_size should be >= 2G for multi-GPU")

        return errors

    @staticmethod
    def validate_config_combination(
        model_config: ModelConfig,
        docker_config: DockerConfig,
    ) -> list[str]:
        """Validate a complete configuration combination."""
        errors: list[str] = []
        errors.extend(ConfigValidator.validate_model_config(model_config))
        errors.extend(ConfigValidator.validate_docker_config(docker_config))

        # Cross-validation
        if docker_config.gpus > 1 and model_config.tensor_parallel < docker_config.gpus:
            errors.append(
                f"tensor_parallel ({model_config.tensor_parallel}) < gpus ({docker_config.gpus})"
            )

        if docker_config.gpus == 1 and model_config.tensor_parallel > 1:
            errors.append(
                f"tensor_parallel ({model_config.tensor_parallel}) > gpus ({docker_config.gpus})"
            )

        return errors


class OutputValidator:
    """Validate benchmark outputs."""

    @staticmethod
    def validate_json_output(output: str) -> tuple[bool, str]:
        """Validate that output is valid JSON."""
        import json
        try:
            json.loads(output)
            return True, ""
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"

    @staticmethod
    def validate_utf8(output: str) -> tuple[bool, str]:
        """Validate UTF-8 encoding."""
        try:
            output.encode("utf-8")
            return True, ""
        except UnicodeEncodeError as e:
            return False, f"Invalid UTF-8: {e}"

    @staticmethod
    def validate_repetition(output: str, max_repeat: int = 10) -> tuple[bool, str]:
        """Check for excessive repetition."""
        words = output.split()
        if len(words) < max_repeat:
            return True, ""

        for i in range(len(words) - max_repeat + 1):
            if all(w == words[i] for w in words[i:i + max_repeat]):
                return False, f"Token '{words[i]}' repeated {max_repeat} times"

        return True, ""

    @staticmethod
    def validate_length(output: str, min_len: int = 1, max_len: int = 100000) -> tuple[bool, str]:
        """Validate output length."""
        if len(output) < min_len:
            return False, f"Output too short ({len(output)} chars)"
        if len(output) > max_len:
            return False, f"Output too long ({len(output)} chars)"
        return True, ""

    @staticmethod
    def validate_nan(output: str) -> tuple[bool, str]:
        """Check for NaN values."""
        if "NaN" in output or "nan" in output:
            return False, "NaN values detected"
        return True, ""

    @staticmethod
    def validate(output: str) -> dict[str, Any]:
        """Run all validations."""
        results = {
            "valid": True,
            "checks": {},
        }

        checks = [
            ("json", lambda: OutputValidator.validate_json_output(output)),
            ("utf8", lambda: OutputValidator.validate_utf8(output)),
            ("repetition", lambda: OutputValidator.validate_repetition(output)),
            ("length", lambda: OutputValidator.validate_length(output)),
            ("nan", lambda: OutputValidator.validate_nan(output)),
        ]

        for name, check_fn in checks:
            valid, msg = check_fn()
            results["checks"][name] = {"valid": valid, "message": msg}
            if not valid:
                results["valid"] = False

        return results


class SystemValidator:
    """Validate system state."""

    @staticmethod
    def check_gpu_available() -> tuple[bool, str]:
        """Check if Intel GPU is available."""
        try:
            import subprocess
            result = subprocess.run(
                ["lspci", "|", "grep", "-i", "vga"],
                shell=True,
                capture_output=True,
                text=True,
            )
            if "intel" in result.stdout.lower():
                return True, "Intel GPU detected"
            return False, "No Intel GPU detected"
        except Exception as e:
            return False, f"GPU check failed: {e}"

    @staticmethod
    def check_docker_available() -> tuple[bool, str]:
        """Check if Docker is available."""
        try:
            import subprocess
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return True, "Docker is available"
            return False, f"Docker check failed: {result.stderr}"
        except FileNotFoundError:
            return False, "Docker not installed"
        except Exception as e:
            return False, f"Docker check failed: {e}"

    @staticmethod
    def check_intel_gpu_tools() -> tuple[bool, str]:
        """Check if intel_gpu_top is available."""
        try:
            import shutil
            if shutil.which("intel_gpu_top"):
                return True, "intel_gpu_top available"
            return False, "intel_gpu_top not found"
        except Exception as e:
            return False, f"Check failed: {e}"

    @staticmethod
    def check_disk_space(path: str = "/") -> tuple[bool, str]:
        """Check available disk space."""
        import shutil
        total, used, free = shutil.disk_usage(path)
        if free < 10 * 1024 * 1024 * 1024:  # 10GB
            return False, f"Low disk space: {free / (1024**3):.1f}GB free"
        return True, f"{free / (1024**3):.1f}GB free"

    @staticmethod
    def check_memory_available() -> tuple[bool, str]:
        """Check available system memory."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            if mem.available < 4 * 1024 * 1024 * 1024:  # 4GB
                return False, f"Low memory: {mem.available / (1024**3):.1f}GB available"
            return True, f"{mem.available / (1024**3):.1f}GB available"
        except Exception as e:
            return False, f"Memory check failed: {e}"

    @staticmethod
    def run_full_check() -> dict[str, Any]:
        """Run all system checks."""
        checks = [
            ("gpu", SystemValidator.check_gpu_available),
            ("docker", SystemValidator.check_docker_available),
            ("intel_gpu_tools", SystemValidator.check_intel_gpu_tools),
            ("disk_space", SystemValidator.check_disk_space),
            ("memory", SystemValidator.check_memory_available),
        ]

        results = {}
        all_passed = True
        for name, check_fn in checks:
            passed, msg = check_fn()
            results[name] = {"passed": passed, "message": msg}
            if not passed:
                all_passed = False

        return {
            "all_passed": all_passed,
            "checks": results,
        }
