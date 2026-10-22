"""The job queue. Long work runs in one worker thread; tools return a job id at once.

Every job is a JSON file under ``jobs/`` and every finished mesh a folder of files under
``out/`` (GLB, OBJ, cutout, turntable), so nothing is lost when the server stops. On start,
jobs that were running under a dead process are re-queued. Exactly one process per data
directory runs the worker (``worker.lock``); the others submit jobs and read status.
"""
# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import json
import os
import queue
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from shapestudio.config import Settings
from shapestudio.engine.base import EngineError, MeshEngine, MeshSpec, Progress

STATUSES = ("queued", "running", "done", "failed", "cancelled")
FINAL = ("done", "failed", "cancelled")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def human(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    s = round(seconds)
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min"
    return f"{s // 3600} h {(s % 3600) // 60:02d} min"


@dataclass
class Job:
    id: str
    tool: str  # reconstruct | import
    input_path: str
    source_name: str  # the original file name, for titles
    model: str
    tier: str
    device: str
    preset: str
    resolution: int
    foreground_ratio: float
    remove_background: bool
    bg_model: str | None
    estimate_seconds: float
    estimate_basis: str
    fit_note: str | None
    warnings: list[str]
    status: str = "queued"
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    phase: str = "queued"
    progress: float = 0.0
    message: str = ""
    glb_path: str | None = None
    obj_path: str | None = None
    preview_path: str | None = None
    thumb_path: str | None = None
    cutout_path: str | None = None
    vertices: int | None = None
    faces: int | None = None
    error: str | None = None
    load_seconds: float | None = None
    network_seconds: float | None = None
    grid_seconds: float | None = None
    total_seconds: float | None = None
    peak_rss_bytes: int | None = None
    threads: int | None = None
    over_budget: bool = False
    worker_pid: int | None = None
    cancel_requested: bool = False
    favorite: bool = False
    deleted_at: str | None = None
    project: str | None = None
    note: str = ""  # free text the user may attach

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Job:
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)

    # -- derived -------------------------------------------------------------------------

    def elapsed_seconds(self) -> float:
        if not self.started_at:
            return 0.0
        end = datetime.fromisoformat(self.finished_at) if self.finished_at else datetime.now(UTC)
        return (end - datetime.fromisoformat(self.started_at)).total_seconds()

    def eta_seconds(self, queue_ahead_seconds: float = 0.0) -> float | None:
        """Remaining seconds from the estimate; None once the estimate is spent (then the
        UI says 'measuring' instead of inventing a number)."""
        if self.status in FINAL:
            return 0.0
        if self.status == "queued":
            return queue_ahead_seconds + self.estimate_seconds
        if self.over_budget:
            return None
        elapsed = self.elapsed_seconds()
        if self.progress >= 0.3 and elapsed > 5:
            # phases after the network pass scale with the grid; extrapolate from progress
            projected = elapsed / max(self.progress, 1e-3)
            remaining = projected - elapsed
            return max(0.0, remaining)
        remaining = self.estimate_seconds - elapsed
        if self.estimate_seconds <= 0 or remaining < 0.1 * self.estimate_seconds:
            return None
        return remaining

    def summary(
        self, queue_position: int | None = None, eta: float | None = None, paused: bool = False
    ) -> str:
        """One honest line for every surface (card, header chip, tray)."""
        if self.status == "queued":
            if paused:
                return "queued · paused"
            pos = f"queued #{queue_position} · " if queue_position else "queued · "
            return pos + (f"≈ {human(eta)}" if eta is not None else "estimate unknown")
        if self.status == "done":
            return f"done in {human(self.total_seconds or 0)}"
        if self.status in ("failed", "cancelled"):
            return self.status
        label = {
            "loading": "loading weights",
            "preprocessing": "cutting out the object",
            "encoding": "network pass",
            "extracting": "extracting the surface",
            "colouring": "colouring",
            "exporting": "writing GLB and OBJ",
            "preview": "rendering the turntable",
            "saving": "finishing",
            "queued": "starting",
        }.get(self.phase, self.phase)
        el = self.elapsed_seconds()
        if eta is not None:
            return f"{label} · ≈ {human(eta)} left"
        slow = (
            " · longer than estimated"
            if self.estimate_seconds and el > 0.9 * self.estimate_seconds
            else ""
        )
        swap = " · swapping" if self.over_budget else ""
        return f"{label} · {human(el)} so far{slow}{swap} · measuring"


class JobStore:
    """Jobs as JSON files. Several processes may share one directory: exactly one runs a
    worker, the others submit jobs and read status. ``refresh`` picks up files changed by
    another process; jobs this process is executing (``owned``) are never overwritten."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._mtimes: dict[str, float] = {}
        self.owned: set[str] = set()
        self.settings.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.refresh()

    def refresh(self) -> None:
        with self._lock:
            for p in self.settings.jobs_dir.glob("*.json"):
                job_id = p.stem
                if job_id in self.owned:
                    continue
                try:
                    mtime = p.stat().st_mtime
                except OSError:
                    continue
                if self._mtimes.get(job_id) == mtime:
                    continue
                try:
                    job = Job.from_dict(json.loads(p.read_text(encoding="utf-8")))
                except (OSError, ValueError, TypeError):
                    continue  # half-written by another process; next refresh gets it
                self._jobs[job.id] = job
                self._mtimes[job_id] = mtime

    def pause_marker(self) -> Path:
        return self.settings.jobs_dir / "queue.paused"

    def is_paused(self) -> bool:
        return self.pause_marker().exists()

    def set_paused(self, paused: bool) -> None:
        if paused:
            self.pause_marker().write_text("paused", encoding="utf-8")
        else:
            self.pause_marker().unlink(missing_ok=True)

    def exit_marker(self) -> Path:
        """Another process asks the worker to stop between jobs (handover to a newer build)."""
        return self.settings.jobs_dir / "worker.exit"

    def cancel_marker(self, job_id: str) -> Path:
        return self.settings.jobs_dir / f"{job_id}.cancel"

    def new_id(self) -> str:
        return "job_" + secrets.token_hex(6)

    def put(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job
            tmp = self.settings.jobs_dir / f"{job.id}.json.tmp"
            tmp.write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")
            final = self.settings.jobs_dir / f"{job.id}.json"
            tmp.replace(final)
            with contextlib.suppress(OSError):
                self._mtimes[job.id] = final.stat().st_mtime

    def forget(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)
            self._mtimes.pop(job_id, None)
            (self.settings.jobs_dir / f"{job_id}.json").unlink(missing_ok=True)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            self.refresh()
            return self._jobs.get(job_id)

    def all(self, include_deleted: bool = False) -> list[Job]:
        with self._lock:
            self.refresh()
            jobs = [j for j in self._jobs.values() if include_deleted or not j.deleted_at]
            return sorted(jobs, key=lambda j: j.created_at)

    def queued(self) -> list[Job]:
        return [j for j in self.all() if j.status == "queued"]

    def running(self) -> Job | None:
        return next((j for j in self.all() if j.status == "running"), None)


def _pid_alive(pid: int) -> bool:
    try:
        return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


_HELD_IN_PROCESS: set[Path] = set()  # several Studios in one process share one pid


class WorkerLock:
    """One worker per data directory. The lock file holds the owner's pid; a dead owner's
    lock is taken over."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.held = False

    def info(self) -> dict | None:
        try:
            raw = self.path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        try:
            d = json.loads(raw) if raw.startswith("{") else {"pid": int(raw)}
        except ValueError:
            return None
        pid = int(d.get("pid", 0))
        return d if pid and _pid_alive(pid) else None

    def owner(self) -> int | None:
        d = self.info()
        return int(d["pid"]) if d else None

    def acquire(self) -> bool:
        if self.path in _HELD_IN_PROCESS:
            return False
        owner = self.owner()
        if owner is not None and owner != os.getpid():
            return False
        _HELD_IN_PROCESS.add(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        from shapestudio import __version__

        self.path.write_text(
            json.dumps({"pid": os.getpid(), "version": __version__, "started_at": _now()}),
            encoding="utf-8",
        )
        self.held = True
        return True

    def release(self) -> None:
        if self.held:
            _HELD_IN_PROCESS.discard(self.path)
            with contextlib.suppress(OSError):
                if self.owner() in (None, os.getpid()):
                    self.path.unlink()
            self.held = False


class Worker:
    """One background thread draining the queue through the engine, in FIFO order."""

    def __init__(
        self,
        store: JobStore,
        engines: dict[str, MeshEngine],
        on_finished=None,
        spec_builder=None,
    ) -> None:
        self.store = store
        self.engines = engines  # engine name -> engine
        self.spec_builder = spec_builder  # Job -> MeshSpec (resolves paths; set by the Studio)
        self.on_finished = on_finished
        self._q: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="shapestudio-worker", daemon=True)
        self._current: str | None = None
        self._current_engine: MeshEngine | None = None
        self._seen: set[str] = set()
        self.lock = WorkerLock(store.settings.worker_lock_file)
        self.poll_seconds = 2.0

    def start(self) -> bool:
        if self._thread.is_alive():
            return True
        if not self.lock.acquire():
            return False
        for job in self.store.all():
            if job.status == "running":
                if job.worker_pid and job.worker_pid != os.getpid() and _pid_alive(job.worker_pid):
                    continue  # another live process is working on it; leave it alone
                job.status, job.phase, job.message = "queued", "queued", "re-queued after restart"
                job.worker_pid = None
                job.progress = 0.0
                self.store.put(job)
        for job in self.store.queued():
            self._enqueue(job.id)
        self._thread.start()
        return True

    @property
    def active(self) -> bool:
        return self._thread.is_alive()

    def _enqueue(self, job_id: str) -> None:
        if job_id not in self._seen:
            self._seen.add(job_id)
            self._q.put(job_id)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._current and self._current_engine is not None:
            self._current_engine.cancel()
        self._q.put("")  # wake
        if self._thread.is_alive():
            self._thread.join(timeout)
        self.lock.release()

    def submit(self, job: Job) -> None:
        self.store.put(job)
        if self.active:
            self._enqueue(job.id)

    def cancel(self, job_id: str) -> bool:
        job = self.store.get(job_id)
        if job is None or job.status in FINAL:
            return False
        if job.status == "running":
            if job.id in self.store.owned and self._current_engine is not None:
                self._current_engine.cancel()
            else:
                job.cancel_requested = True
                self.store.cancel_marker(job.id).write_text("cancel", encoding="utf-8")
            return True
        job.status, job.phase, job.finished_at = "cancelled", "cancelled", _now()
        self.store.put(job)
        return True

    def queue_seconds_ahead(self, job_id: str) -> float:
        total = 0.0
        running = self.store.running()
        if running and running.id != job_id:
            total += running.eta_seconds() or 0.0
        for j in self.store.queued():
            if j.id == job_id:
                break
            total += j.estimate_seconds
        return total

    def _run(self) -> None:
        while not self._stop.is_set():
            if self.store.exit_marker().exists():
                self.store.exit_marker().unlink(missing_ok=True)
                self._stop.set()
                self.lock.release()
                break
            if self.store.is_paused():
                time.sleep(min(1.0, self.poll_seconds))
                continue
            try:
                job_id = self._q.get(timeout=self.poll_seconds)
            except queue.Empty:
                for job in self.store.queued():  # submitted by another process
                    self._enqueue(job.id)
                continue
            if not job_id or self._stop.is_set():
                continue
            if self.store.is_paused():
                self._seen.discard(job_id)
                time.sleep(min(1.0, self.poll_seconds))
                continue
            job = self.store.get(job_id)
            if job is None or job.status != "queued":
                continue
            self._current = job_id
            self._execute(job)
            self._current = None

    def _execute(self, job: Job) -> None:
        self.store.owned.add(job.id)
        job.status, job.phase, job.started_at = "running", "loading", _now()
        job.worker_pid = os.getpid()
        job.progress = 0.0
        self.store.put(job)
        last_write = 0.0

        def on_progress(p: Progress) -> None:
            nonlocal last_write
            job.phase = p.phase
            job.progress = max(job.progress, min(1.0, float(p.fraction)))
            if p.message:
                job.message = p.message
            now = time.monotonic()
            if now - last_write > 0.5 or p.phase in ("loading", "exporting", "saving"):
                last_write = now
                self.store.put(job)

        from shapestudio.registry import MODELS

        engine_name = MODELS[job.model].engine if job.model in MODELS else job.model
        engine = self.engines.get(engine_name)
        stop_watch = threading.Event()

        def watch_cancel() -> None:
            marker = self.store.cancel_marker(job.id)
            while not stop_watch.wait(0.25):
                if marker.exists():
                    job.cancel_requested = True
                    if engine is not None:
                        engine.cancel()
                    return

        threading.Thread(target=watch_cancel, name=f"cancel-watch-{job.id}", daemon=True).start()
        self._current_engine = engine
        try:
            if engine is None:
                raise EngineError(f"no engine for model {job.model!r}")
            if self.spec_builder is None:
                raise EngineError("worker has no spec builder")
            spec: MeshSpec = self.spec_builder(job)
            result = engine.reconstruct(spec, on_progress)
        except EngineError as e:
            stop_watch.set()
            if str(e) == "cancelled":
                job.status, job.phase, job.message = "cancelled", "cancelled", "cancelled by caller"
            else:
                job.status, job.phase, job.error = "failed", "failed", str(e)
        except Exception as e:
            stop_watch.set()
            job.status, job.phase = "failed", "failed"
            job.error = (
                f"The engine hit an internal error ({type(e).__name__}: {e}). This is a bug "
                "in the studio, not in your picture; please report it from Settings."
            )
        else:
            stop_watch.set()
            job.status, job.phase, job.progress = "done", "done", 1.0
            job.glb_path = str(result.glb)
            job.obj_path = str(result.obj)
            job.preview_path = str(result.preview) if result.preview else None
            thumb = result.glb.with_name(f"{result.glb.stem}.thumb.png")
            job.thumb_path = str(thumb) if thumb.exists() else job.preview_path
            job.cutout_path = str(result.cutout) if result.cutout else None
            job.vertices, job.faces = result.vertices, result.faces
            job.load_seconds = result.load_seconds
            job.network_seconds = result.network_seconds
            job.grid_seconds = result.grid_seconds
            job.total_seconds = result.total_seconds
            job.peak_rss_bytes = result.peak_rss_bytes or None
            job.message = "finished"
        finally:
            self._current_engine = None
        job.finished_at = _now()
        self.store.put(job)
        self.store.owned.discard(job.id)
        with contextlib.suppress(OSError):
            self.store.cancel_marker(job.id).unlink()
        if self.on_finished:
            with contextlib.suppress(Exception):
                self.on_finished(job)
