"""
Utility functions for BenchLab.
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import re
import signal
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Generator, TypeVar

import psutil

T = TypeVar("T")


def generate_id() -> str:
    """Generate a unique identifier."""
    return uuid.uuid4().hex[:12]


def generate_run_id() -> str:
    """Generate a run identifier."""
    return f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def timestamp() -> str:
    """Get current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()


def timestamp_ms() -> int:
    """Get current time as milliseconds since epoch."""
    return int(time.time() * 1000)


def parse_duration(duration_str: str) -> float:
    """Parse a duration string into seconds.

    Supports formats: '30m', '1h', '2h', '4h', '8h', '12h', '24h', '3600', '1.5h'
    """
    if isinstance(duration_str, (int, float)):
        return float(duration_str)

    duration_str = duration_str.strip().lower()

    # Pure number = seconds
    try:
        return float(duration_str)
    except ValueError:
        pass

    multipliers = {
        "s": 1,
        "m": 60,
        "h": 3600,
        "d": 86400,
    }

    match = re.match(r"^([\d.]+)\s*([smhd]?)$", duration_str)
    if not match:
        raise ValueError(f"Invalid duration: {duration_str}")

    value = float(match.group(1))
    unit = match.group(2) or "s"
    return value * multipliers[unit]


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists and return its Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> dict[str, Any]:
    """Read and parse a JSON file."""
    with open(path) as f:
        return json.load(f)


def write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    """Write data as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=str)


def read_text(path: str | Path) -> str:
    """Read a text file."""
    with open(path) as f:
        return f.read()


def write_text(path: str | Path, content: str) -> None:
    """Write text to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def retry_async(
    func: Callable[..., T],
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[..., T]:
    """Async retry decorator with exponential backoff."""
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        last_exception: Exception | None = None
        current_delay = delay

        for attempt in range(1, max_attempts + 1):
            try:
                return await func(*args, **kwargs)  # type: ignore[misc]
            except exceptions as e:
                last_exception = e
                if attempt < max_attempts:
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff

        assert last_exception is not None
        raise last_exception

    wrapper = functools.wraps(func)(wrapper)
    return wrapper  # type: ignore[return-value]


def retry(
    func: Callable[..., T],
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[..., T]:
    """Sync retry decorator with exponential backoff."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        last_exception: Exception | None = None
        current_delay = delay

        for attempt in range(1, max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                last_exception = e
                if attempt < max_attempts:
                    time.sleep(current_delay)
                    current_delay *= backoff

        assert last_exception is not None
        raise last_exception

    return wrapper


class TimeoutError(Exception):
    """Custom timeout error."""

    pass


def timeout(seconds: float) -> "TimeoutContext":
    """Context manager with timeout using signals."""
    return TimeoutContext(seconds)


class TimeoutContext:
    """Context manager with timeout using signals."""

    def __init__(self, seconds: float) -> None:
        self._seconds = seconds

    def __enter__(self) -> "TimeoutContext":
        def _handler(signum: int, frame: Any) -> None:
            raise TimeoutError(f"Operation timed out after {self._seconds} seconds")

        self._old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(int(self._seconds))
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        signal.signal(signal.SIGALRM, self._old_handler)
        signal.alarm(0)


def format_bytes(num_bytes: int | float) -> str:
    """Format bytes into human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"


def truncate_string(s: str, max_length: int = 200) -> str:
    """Truncate a string to max_length with ellipsis."""
    if len(s) <= max_length:
        return s
    return s[: max_length - 3] + "..."


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename."""
    name = re.sub(r"[^\w\-_.]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name[:200]


def get_gpu_info() -> dict[str, Any]:
    """Get basic GPU information."""
    info: dict[str, Any] = {"available": False, "gpus": []}

    # Check for Intel GPUs via lshw or lsmod
    try:
        result = os.popen("lspci | grep -i vga").read().strip()
        if result:
            info["available"] = True
            for line in result.split("\n"):
                info["gpus"].append(line.strip())
    except Exception:
        pass

    # Check for Intel specific info
    try:
        if os.path.exists("/sys/class/drm/card*/device/vendor"):
            cards = Path("/sys/class/drm").glob("card*")
            for card in cards:
                try:
                    vendor_file = Path(card.parent, "device", "vendor")
                    if vendor_file.exists():
                        vendor = vendor_file.read_text().strip()
                        if vendor == "0x8086":  # Intel
                            info["intel_gpus"] = info.get("intel_gpus", 0) + 1
                except Exception:
                    pass
    except Exception:
        pass

    return info


def get_system_info() -> dict[str, Any]:
    """Get system information."""
    return {
        "hostname": os.uname().nodename,
        "platform": sys.platform,
        "python_version": sys.version,
        "cpu_count": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "memory_total": psutil.virtual_memory().total,
        "memory_available": psutil.virtual_memory().available,
        "boot_time": psutil.boot_time(),
        "uptime": time.time() - psutil.boot_time(),
    }


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


class AsyncSemaphore:
    """Async semaphore for rate limiting."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._semaphore = asyncio.Semaphore(limit)

    async def acquire(self) -> None:
        await self._semaphore.acquire()

    async def release(self) -> None:
        self._semaphore.release()

    async def __aenter__(self) -> "AsyncSemaphore":
        await self.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.release()
