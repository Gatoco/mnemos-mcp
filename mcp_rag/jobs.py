"""In-memory async job registry for long-running index operations (ADMIN-003).

A single-worker `ThreadPoolExecutor` serializes index jobs so Qdrant and
Ollama are never contended by concurrent indexing. Jobs live only in memory —
they die on restart (acceptable: local, single-user admin page).
"""
from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field


@dataclass
class Job:
    id: str
    status: str = "running"  # "running" | "done" | "error"
    progress: float = 0.0
    result: dict | None = None
    error: str | None = None
    _lock: object = field(default_factory=threading.Lock, repr=False, compare=False)

    def as_dict(self) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "status": self.status,
                "progress": self.progress,
                "result": self.result,
                "error": self.error,
            }


class JobRegistry:
    def __init__(self, max_workers: int = 1):
        # max_workers=1: serialized index jobs avoid qdrant/ollama contention.
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, fn, *a, **k) -> str:
        job_id = uuid.uuid4().hex
        job = Job(id=job_id)
        with self._lock:
            self._jobs[job_id] = job
        self._pool.submit(self._run, job, fn, a, k)
        return job_id

    def _run(self, job: Job, fn, a, k) -> None:
        try:
            result = fn(*a, **k)
            job.result = result
            job.progress = 1.0
            job.status = "done"
        except Exception as exc:  # noqa: BLE001 - surface any failure
            job.error = str(exc)
            job.status = "error"
        finally:
            pass  # job stays in registry for polling

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)
