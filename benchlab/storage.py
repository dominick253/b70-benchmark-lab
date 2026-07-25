"""
Persistent storage layer for BenchLab.

SQLite-backed storage with tables for:
- Runs
- Configurations
- Benchmarks
- Metrics
- Telemetry
- Corruption events
- Docker containers
- Results

All data is stored in a single SQLite database for simplicity
and portability.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from benchlab.utils import ensure_dir, generate_id, timestamp


class Storage:
    """SQLite storage backend for BenchLab."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn.execute("PRAGMA busy_timeout=5000")
        return self._local.conn

    @contextmanager
    def _transaction(self) -> Any:
        """Context manager for database transactions."""
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_db(self) -> None:
        """Initialize the database schema."""
        conn = self._get_conn()

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT NOT NULL DEFAULT 'running',
                config_hash TEXT,
                total_configs INTEGER DEFAULT 0,
                completed_configs INTEGER DEFAULT 0,
                system_info TEXT,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS configurations (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                config_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                score REAL,
                rank INTEGER,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                error TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            );

            CREATE TABLE IF NOT EXISTS benchmarks (
                id TEXT PRIMARY KEY,
                config_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                workload_type TEXT NOT NULL,
                workload_length TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                duration REAL,
                status TEXT NOT NULL DEFAULT 'running',
                error TEXT,
                FOREIGN KEY (config_id) REFERENCES configurations(id),
                FOREIGN KEY (run_id) REFERENCES runs(id)
            );

            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                benchmark_id TEXT NOT NULL,
                config_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_type TEXT NOT NULL DEFAULT 'gauge',
                timestamp TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY (benchmark_id) REFERENCES benchmarks(id),
                FOREIGN KEY (config_id) REFERENCES configurations(id),
                FOREIGN KEY (run_id) REFERENCES runs(id)
            );

            CREATE INDEX IF NOT EXISTS idx_metrics_benchmark ON metrics(benchmark_id);
            CREATE INDEX IF NOT EXISTS idx_metrics_config ON metrics(config_id);
            CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id);
            CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name);
            CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);

            CREATE TABLE IF NOT EXISTS telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                benchmark_id TEXT NOT NULL,
                config_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                source TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                data TEXT NOT NULL,
                FOREIGN KEY (benchmark_id) REFERENCES benchmarks(id),
                FOREIGN KEY (config_id) REFERENCES configurations(id),
                FOREIGN KEY (run_id) REFERENCES runs(id)
            );

            CREATE INDEX IF NOT EXISTS idx_telemetry_benchmark ON telemetry(benchmark_id);
            CREATE INDEX IF NOT EXISTS idx_telemetry_config ON telemetry(config_id);
            CREATE INDEX IF NOT EXISTS idx_telemetry_run ON telemetry(run_id);
            CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON telemetry(timestamp);

            CREATE TABLE IF NOT EXISTS corruption_events (
                id TEXT PRIMARY KEY,
                benchmark_id TEXT NOT NULL,
                config_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL DEFAULT 'high',
                timestamp TEXT NOT NULL,
                prompt TEXT,
                output TEXT,
                evidence TEXT,
                logs TEXT,
                FOREIGN KEY (benchmark_id) REFERENCES benchmarks(id),
                FOREIGN KEY (config_id) REFERENCES configurations(id),
                FOREIGN KEY (run_id) REFERENCES runs(id)
            );

            CREATE INDEX IF NOT EXISTS idx_corruption_benchmark ON corruption_events(benchmark_id);
            CREATE INDEX IF NOT EXISTS idx_corruption_config ON corruption_events(config_id);
            CREATE INDEX IF NOT EXISTS idx_corruption_run ON corruption_events(run_id);

            CREATE TABLE IF NOT EXISTS docker_containers (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                config_id TEXT NOT NULL,
                container_id TEXT NOT NULL,
                container_name TEXT NOT NULL,
                image TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                exit_code INTEGER,
                logs_path TEXT,
                metrics_path TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(id),
                FOREIGN KEY (config_id) REFERENCES configurations(id)
            );

            CREATE TABLE IF NOT EXISTS results (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                config_id TEXT NOT NULL,
                benchmark_id TEXT,
                result_type TEXT NOT NULL,
                result_data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(id),
                FOREIGN KEY (config_id) REFERENCES configurations(id),
                FOREIGN KEY (benchmark_id) REFERENCES benchmarks(id)
            );

            CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id);
            CREATE INDEX IF NOT EXISTS idx_results_config ON results(config_id);
            CREATE INDEX IF NOT EXISTS idx_results_type ON results(result_type);
        """)

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def create_run(self, run_id: str, name: str, metadata: dict[str, Any] | None = None) -> None:
        """Create a new run."""
        with self._transaction() as conn:
            conn.execute(
                """INSERT INTO runs (id, name, created_at, status, metadata)
                   VALUES (?, ?, ?, 'running', ?)""",
                (run_id, name, timestamp(), json.dumps(metadata or {})),
            )

    def update_run_status(self, run_id: str, status: str, completed_at: str | None = None) -> None:
        """Update run status."""
        with self._transaction() as conn:
            if completed_at:
                conn.execute(
                    "UPDATE runs SET status = ?, completed_at = ? WHERE id = ?",
                    (status, completed_at, run_id),
                )
            else:
                conn.execute(
                    "UPDATE runs SET status = ? WHERE id = ?",
                    (status, run_id),
                )

    def increment_completed_configs(self, run_id: str) -> None:
        """Increment completed configs counter."""
        with self._transaction() as conn:
            conn.execute(
                "UPDATE runs SET completed_configs = completed_configs + 1 WHERE id = ?",
                (run_id,),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Get run by ID."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row:
            return dict(row)
        return None

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """List all runs."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def save_run(self, run_data: dict[str, Any]) -> str:
        """Save a complete benchmark run result."""
        run_id = run_data.get("run_id", f"run-{generate_id()}")
        config_hash = run_data.get("config_hash", "")
        status = run_data.get("status", "unknown")
        duration = run_data.get("duration", 0)
        start_time = run_data.get("start_time", 0)
        end_time = run_data.get("end_time", 0)
        metrics = run_data.get("metrics", {})
        gpu_metrics = run_data.get("gpu_metrics", {})
        vllm_metrics = run_data.get("vllm_metrics", {})
        corruption_events = run_data.get("corruption_events", [])
        score = run_data.get("score", 0)

        # Insert into runs table
        self.create_run(run_id, config_hash)
        self.update_run_status(run_id, status, timestamp())

        # Insert into configurations table
        config_id = f"config-{run_id}-{config_hash[:8]}"
        self.create_configuration(config_id, run_id, config_hash, run_data)

        # Record metrics
        metric_records = []
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                metric_records.append((
                    run_id, config_id, run_id, key, float(val), "gauge", None
                ))
        if metric_records:
            self.record_metrics_batch(metric_records)

        # Store score
        self.set_config_score(config_id, score)

        logger.info(f"Saved run {run_id} ({status}, score={score:.4f})")
        return run_id

    # ------------------------------------------------------------------
    # Configurations
    # ------------------------------------------------------------------

    def create_configuration(
        self,
        config_id: str,
        run_id: str,
        config_hash: str,
        config: dict[str, Any],
    ) -> None:
        """Create a new configuration."""
        with self._transaction() as conn:
            conn.execute(
                """INSERT INTO configurations (id, run_id, config_hash, config_json, created_at, status)
                   VALUES (?, ?, ?, ?, ?, 'pending')""",
                (config_id, run_id, config_hash, json.dumps(config), timestamp()),
            )

    def update_config_status(self, config_id: str, status: str, error: str | None = None) -> None:
        """Update configuration status."""
        with self._transaction() as conn:
            if error:
                conn.execute(
                    """UPDATE configurations SET status = ?, error = ?, completed_at = ?
                       WHERE id = ?""",
                    (status, error, timestamp(), config_id),
                )
            else:
                conn.execute(
                    """UPDATE configurations SET status = ?, completed_at = ?
                       WHERE id = ?""",
                    (status, timestamp(), config_id),
                )

    def set_config_score(self, config_id: str, score: float) -> None:
        """Set configuration score."""
        with self._transaction() as conn:
            conn.execute(
                "UPDATE configurations SET score = ? WHERE id = ?",
                (score, config_id),
            )

    def set_config_rank(self, config_id: str, rank: int) -> None:
        """Set configuration rank."""
        with self._transaction() as conn:
            conn.execute(
                "UPDATE configurations SET rank = ? WHERE id = ?",
                (rank, config_id),
            )

    def get_config(self, config_id: str) -> dict[str, Any] | None:
        """Get configuration by ID."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM configurations WHERE id = ?", (config_id,)).fetchone()
        if row:
            return dict(row)
        return None

    def get_configs_for_run(self, run_id: str) -> list[dict[str, Any]]:
        """Get all configurations for a run."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM configurations WHERE run_id = ? ORDER BY score DESC NULLS LAST",
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_pending_configs(self, run_id: str, limit: int = 1) -> list[dict[str, Any]]:
        """Get pending configurations for a run."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT * FROM configurations
               WHERE run_id = ? AND status = 'pending'
               ORDER BY created_at ASC
               LIMIT ?""",
            (run_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Benchmarks
    # ------------------------------------------------------------------

    def create_benchmark(
        self,
        benchmark_id: str,
        config_id: str,
        run_id: str,
        workload_type: str,
        workload_length: str,
    ) -> None:
        """Create a new benchmark."""
        with self._transaction() as conn:
            conn.execute(
                """INSERT INTO benchmarks (id, config_id, run_id, workload_type, workload_length, start_time, status)
                   VALUES (?, ?, ?, ?, ?, ?, 'running')""",
                (benchmark_id, config_id, run_id, workload_type, workload_length, timestamp()),
            )

    def complete_benchmark(
        self,
        benchmark_id: str,
        duration: float,
        status: str = "completed",
        error: str | None = None,
    ) -> None:
        """Complete a benchmark."""
        with self._transaction() as conn:
            conn.execute(
                """UPDATE benchmarks SET status = ?, end_time = ?, duration = ?, error = ?
                   WHERE id = ?""",
                (status, timestamp(), duration, error, benchmark_id),
            )

    def get_benchmark(self, benchmark_id: str) -> dict[str, Any] | None:
        """Get benchmark by ID."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM benchmarks WHERE id = ?", (benchmark_id,)).fetchone()
        if row:
            return dict(row)
        return None

    def get_benchmarks_for_config(self, config_id: str) -> list[dict[str, Any]]:
        """Get all benchmarks for a configuration."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM benchmarks WHERE config_id = ? ORDER BY start_time ASC",
            (config_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def record_metric(
        self,
        benchmark_id: str,
        config_id: str,
        run_id: str,
        metric_name: str,
        metric_value: float,
        metric_type: str = "gauge",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a metric."""
        with self._transaction() as conn:
            conn.execute(
                """INSERT INTO metrics (benchmark_id, config_id, run_id, metric_name, metric_value, metric_type, timestamp, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    benchmark_id,
                    config_id,
                    run_id,
                    metric_name,
                    metric_value,
                    metric_type,
                    timestamp(),
                    json.dumps(metadata or {}),
                ),
            )

    def record_metrics_batch(
        self,
        metrics: list[tuple[str, str, str, str, float, str, dict[str, Any] | None]],
    ) -> None:
        """Record multiple metrics in a batch."""
        with self._transaction() as conn:
            conn.executemany(
                """INSERT INTO metrics (benchmark_id, config_id, run_id, metric_name, metric_value, metric_type, timestamp, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        benchmark_id,
                        config_id,
                        run_id,
                        name,
                        value,
                        metric_type,
                        timestamp(),
                        json.dumps(metadata or {}),
                    )
                    for benchmark_id, config_id, run_id, name, value, metric_type, metadata in metrics
                ],
            )

    def get_metrics(
        self,
        benchmark_id: str | None = None,
        config_id: str | None = None,
        run_id: str | None = None,
        metric_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get metrics with optional filters."""
        conn = self._get_conn()
        query = "SELECT * FROM metrics WHERE 1=1"
        params: list[Any] = []

        if benchmark_id:
            query += " AND benchmark_id = ?"
            params.append(benchmark_id)
        if config_id:
            query += " AND config_id = ?"
            params.append(config_id)
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        if metric_name:
            query += " AND metric_name = ?"
            params.append(metric_name)

        query += " ORDER BY timestamp ASC"
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_metric_summary(
        self,
        config_id: str,
        metric_name: str,
    ) -> dict[str, float]:
        """Get summary statistics for a metric."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT
                  COUNT(*) as count,
                  AVG(metric_value) as avg,
                  MIN(metric_value) as min,
                  MAX(metric_value) as max
               FROM metrics
               WHERE config_id = ? AND metric_name = ?""",
            (config_id, metric_name),
        ).fetchall()
        if rows:
            row = rows[0]
            # SQLite doesn't support PERCENTILE_CONT, compute p95 manually
            data_rows = conn.execute(
                "SELECT metric_value FROM metrics WHERE config_id = ? AND metric_name = ? ORDER BY metric_value",
                (config_id, metric_name),
            ).fetchall()
            p95 = 0.0
            if data_rows:
                vals = [r[0] for r in data_rows]
                k = (len(vals) - 1) * 0.95
                f_idx = int(k)
                c_idx = f_idx + 1
                if c_idx >= len(vals):
                    p95 = vals[-1]
                else:
                    p95 = vals[f_idx] + (k - f_idx) * (vals[c_idx] - vals[f_idx])
            return {"count": row["count"], "avg": row["avg"], "min": row["min"], "max": row["max"], "p95": p95}
        return {"count": 0, "avg": 0, "min": 0, "max": 0, "p95": 0}

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def record_telemetry(
        self,
        benchmark_id: str,
        config_id: str,
        run_id: str,
        source: str,
        timestamp_val: str,
        data: dict[str, Any],
    ) -> None:
        """Record telemetry data."""
        with self._transaction() as conn:
            conn.execute(
                """INSERT INTO telemetry (benchmark_id, config_id, run_id, source, timestamp, data)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (benchmark_id, config_id, run_id, source, timestamp_val, json.dumps(data)),
            )

    def record_telemetry_batch(
        self,
        telemetry: list[tuple[str, str, str, str, str, dict[str, Any]]],
    ) -> None:
        """Record multiple telemetry entries in a batch."""
        with self._transaction() as conn:
            conn.executemany(
                """INSERT INTO telemetry (benchmark_id, config_id, run_id, source, timestamp, data)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [
                    (b, c, r, s, t, json.dumps(d))
                    for b, c, r, s, t, d in telemetry
                ],
            )

    def get_telemetry(
        self,
        benchmark_id: str | None = None,
        config_id: str | None = None,
        run_id: str | None = None,
        source: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get telemetry data with optional filters."""
        conn = self._get_conn()
        query = "SELECT * FROM telemetry WHERE 1=1"
        params: list[Any] = []

        if benchmark_id:
            query += " AND benchmark_id = ?"
            params.append(benchmark_id)
        if config_id:
            query += " AND config_id = ?"
            params.append(config_id)
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        if source:
            query += " AND source = ?"
            params.append(source)

        query += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Corruption Events
    # ------------------------------------------------------------------

    def record_corruption_event(
        self,
        event_id: str,
        benchmark_id: str,
        config_id: str,
        run_id: str,
        event_type: str,
        severity: str = "high",
        prompt: str | None = None,
        output: str | None = None,
        evidence: str | None = None,
        logs: str | None = None,
    ) -> None:
        """Record a corruption event."""
        with self._transaction() as conn:
            conn.execute(
                """INSERT INTO corruption_events (id, benchmark_id, config_id, run_id, event_type, severity, timestamp, prompt, output, evidence, logs)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event_id,
                    benchmark_id,
                    config_id,
                    run_id,
                    event_type,
                    severity,
                    timestamp(),
                    prompt,
                    output,
                    evidence,
                    logs,
                ),
            )

    def get_corruption_events(
        self,
        benchmark_id: str | None = None,
        config_id: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get corruption events with optional filters."""
        conn = self._get_conn()
        query = "SELECT * FROM corruption_events WHERE 1=1"
        params: list[Any] = []

        if benchmark_id:
            query += " AND benchmark_id = ?"
            params.append(benchmark_id)
        if config_id:
            query += " AND config_id = ?"
            params.append(config_id)
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)

        query += " ORDER BY timestamp DESC"
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Docker Containers
    # ------------------------------------------------------------------

    def create_container(
        self,
        container_id: str,
        run_id: str,
        config_id: str,
        docker_id: str,
        name: str,
        image: str,
        status: str,
    ) -> None:
        """Create a container record."""
        with self._transaction() as conn:
            conn.execute(
                """INSERT INTO docker_containers (id, run_id, config_id, container_id, container_name, image, status, started_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (container_id, run_id, config_id, docker_id, name, image, status, timestamp()),
            )

    def update_container_status(
        self,
        container_id: str,
        status: str,
        exit_code: int | None = None,
        logs_path: str | None = None,
        metrics_path: str | None = None,
    ) -> None:
        """Update container status."""
        with self._transaction() as conn:
            updates = ["status = ?", "completed_at = ?"]
            params: list[Any] = [status, timestamp()]

            if exit_code is not None:
                updates.append("exit_code = ?")
                params.append(exit_code)
            if logs_path is not None:
                updates.append("logs_path = ?")
                params.append(logs_path)
            if metrics_path is not None:
                updates.append("metrics_path = ?")
                params.append(metrics_path)

            updates.append("id = ?")
            params.append(container_id)

            conn.execute(
                f"UPDATE docker_containers SET {', '.join(updates)} WHERE id = ?",
                params,
            )

    def get_container(self, container_id: str) -> dict[str, Any] | None:
        """Get container by ID."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM docker_containers WHERE id = ?", (container_id,)).fetchone()
        if row:
            return dict(row)
        return None

    def get_containers_for_run(self, run_id: str) -> list[dict[str, Any]]:
        """Get all containers for a run."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM docker_containers WHERE run_id = ? ORDER BY started_at ASC",
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def save_result(
        self,
        result_id: str,
        run_id: str,
        config_id: str,
        benchmark_id: str | None,
        result_type: str,
        result_data: dict[str, Any],
    ) -> None:
        """Save a result."""
        with self._transaction() as conn:
            conn.execute(
                """INSERT INTO results (id, run_id, config_id, benchmark_id, result_type, result_data, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    result_id,
                    run_id,
                    config_id,
                    benchmark_id,
                    result_type,
                    json.dumps(result_data),
                    timestamp(),
                ),
            )

    def get_results(
        self,
        run_id: str | None = None,
        config_id: str | None = None,
        result_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get results with optional filters."""
        conn = self._get_conn()
        query = "SELECT * FROM results WHERE 1=1"
        params: list[Any] = []

        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        if config_id:
            query += " AND config_id = ?"
            params.append(config_id)
        if result_type:
            query += " AND result_type = ?"
            params.append(result_type)

        query += " ORDER BY created_at DESC"
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None

    def __del__(self) -> None:
        """Destructor to ensure connection is closed."""
        self.close()

    def __enter__(self) -> Storage:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
