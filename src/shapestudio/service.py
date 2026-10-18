"""The studio: wires hardware, models, calibration, planner, engines and the job queue.
The MCP server, the CLI and the web UI all talk to this one object."""
# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import contextlib
import io
import json
import re
import secrets
import shutil
import threading
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shapestudio import __version__, runtime
from shapestudio.calibration import Calibration, human_duration
from shapestudio.config import ENV_ALLOW_OVER_BUDGET, Settings
from shapestudio.hardware import GIB, Hardware, detect
from shapestudio.jobs import FINAL, Job, JobStore, Worker
from shapestudio.models import ModelStore
from shapestudio.planner import Plan, PlanError, Planner, cuda_gpu, fit_for, memory_budget
from shapestudio.registry import (
    DEFAULT_MODEL,
    EXCLUDED_BY_POLICY,
    LICENCES,
    MODELS,
    TOOL_MODELS,
    MeshModel,
    default_tool_model,
)

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


class StudioError(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def job_title(job: Job) -> str:
    try:
        when = datetime.fromisoformat(job.finished_at or job.created_at).astimezone()
        stamp = when.strftime("%b %d, %H:%M")
    except ValueError:
        stamp = job.created_at
    name = Path(job.source_name).stem[:32] if job.source_name else job.id
    return f"{stamp} · {job.preset} · {job.resolution}³ · {name}"


class BackgroundTask:
    """A long operation started from the UI (download, runtime install, benchmark)."""

    def __init__(self, kind: str, label: str) -> None:
        self.kind = kind
        self.label = label
        self.status = "running"
        self.done_bytes = 0
        self.total_bytes = 0
        self.current_file: str | None = None
        self.message = ""
        self.error: str | None = None
        self.result: dict[str, Any] | None = None
        self.started_at = time.time()
        self.finished_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "done_bytes": self.done_bytes,
            "total_bytes": self.total_bytes,
            "progress": (self.done_bytes / self.total_bytes) if self.total_bytes else None,
            "current_file": self.current_file,
            "message": self.message,
            "error": self.error,
            "result": self.result,
            "elapsed_seconds": round((self.finished_at or time.time()) - self.started_at),
        }


class Studio:
    def __init__(
        self,
        settings: Settings | None = None,
        engines: dict[str, Any] | None = None,
        hardware: Hardware | None = None,
        start_worker: bool = True,
        background_engine=None,
        cuda_runtime: bool | None = None,
    ) -> None:
        self.settings = settings or Settings.load()
        self.settings.ensure_dirs()
        self.hardware = hardware or detect(self.settings.hardware_cache_file)
        self.models = ModelStore(self.settings)
        self.calibration = Calibration(self.settings.calibration_file, self.hardware)
        if cuda_runtime is None:
            cuda_runtime = runtime.torch_status().cuda_available
        self._cuda_runtime = cuda_runtime
        self.planner = Planner(
            self.hardware,
            self.models,
            self.calibration,
            allow_over_budget=self.settings.allow_over_budget,
            cap_bytes=self._cap_bytes(),
            device_preference=self.settings.device_preference,
            cuda_runtime=cuda_runtime,
        )
        if background_engine is None:
            from shapestudio.engine.background import BackgroundEngine

            background_engine = BackgroundEngine()
        self.background_engine = background_engine
        if engines is None:
            from shapestudio.engine.trellis import TrellisEngine
            from shapestudio.engine.triposr import TripoSREngine

            engines = {"triposr": TripoSREngine(background_engine), "trellis": TrellisEngine()}
        self.engines = engines
        self.jobs = JobStore(self.settings)
        self.worker = Worker(
            self.jobs, self.engines, on_finished=self._on_finished, spec_builder=self._spec_for
        )
        self.worker_wanted = start_worker
        self.tasks: dict[str, BackgroundTask] = {}
        if start_worker:
            self.worker.start()

    # -- lifecycle ----------------------------------------------------------------------------

    @property
    def worker_active(self) -> bool:
        return self.worker.active

    @property
    def worker_owner_pid(self) -> int | None:
        return self.worker.lock.owner()

    def try_start_worker(self) -> bool:
        return self.worker.start()

    def request_worker_handover(self) -> dict[str, Any]:
        owner = self.worker.lock.owner()
        if owner is None or self.worker.active:
            return {"requested": False, "reason": "no other worker"}
        self.jobs.exit_marker().write_text("exit", encoding="utf-8")
        return {"requested": True, "owner_pid": owner}

    def close(self) -> None:
        self.worker.stop()

    def _cap_bytes(self) -> int | None:
        cap = self.settings.max_memory_gib
        return int(cap * GIB) if cap else None

    def _on_finished(self, job: Job) -> None:
        if job.status != "done" or job.tool != "reconstruct":
            return
        model = MODELS.get(job.model)
        if model is None or job.network_seconds is None:
            return
        self.calibration.record(
            model,
            job.device,
            job.resolution,
            job.load_seconds or 0.0,
            job.network_seconds or 0.0,
            job.grid_seconds or 0.0,
        )

    # -- inputs -------------------------------------------------------------------------------

    def save_upload(self, data: bytes, name: str = "upload.png") -> Path:
        """Store an uploaded picture under the data dir and return its path."""
        from PIL import Image

        try:
            with Image.open(io.BytesIO(data)) as im:
                im.load()
                fmt = (im.format or "PNG").lower()
        except Exception as e:
            raise StudioError(f"not an image: {e}") from e
        ext = {"jpeg": ".jpg", "png": ".png", "webp": ".webp"}.get(fmt, ".png")
        self.settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(name).stem)[:40] or "image"
        p = self.settings.uploads_dir / f"{stem}-{secrets.token_hex(4)}{ext}"
        p.write_bytes(data)
        return p

    def _resolve_input(self, image: str) -> tuple[Path, str]:
        """A job id (reuse its input), or a path to a PNG/JPEG/WebP on this machine. The file
        is copied into the uploads folder so the gallery keeps the source."""
        if image in {j.id for j in self.jobs.all(include_deleted=True)}:
            job = next(j for j in self.jobs.all(include_deleted=True) if j.id == image)
            return Path(job.input_path), job.source_name
        p = Path(image).expanduser()
        if not p.is_file():
            raise StudioError(f"image not found: {image}")
        if p.suffix.lower() not in IMAGE_SUFFIXES:
            raise StudioError(f"{p.name}: use a PNG, JPEG or WebP picture")
        if self.settings.uploads_dir in p.parents:
            return p, p.name
        stored = self.save_upload(p.read_bytes(), p.name)
        return stored, p.name

    # -- planning and queuing -----------------------------------------------------------------

    def plan(
        self,
        model: str | None = None,
        preset: str = "best",
        resolution: int | None = None,
        foreground_ratio: float | None = None,
        remove_background: bool = True,
    ) -> Plan:
        try:
            return self.planner.plan(model, preset, resolution, foreground_ratio, remove_background)
        except PlanError as e:
            raise StudioError(str(e)) from e

    def reconstruct(
        self,
        image: str,
        model: str | None = None,
        preset: str = "best",
        resolution: int | None = None,
        foreground_ratio: float | None = None,
        remove_background: bool = True,
        bg_model: str | None = None,
        threads: int | None = None,
        project: str | None = None,
        note: str = "",
    ) -> Job:
        input_path, source_name = self._resolve_input(image)
        plan = self.plan(model, preset, resolution, foreground_ratio, remove_background)
        engine = self.engines.get(plan.model.engine)
        if engine is None or not engine.available():
            reason = ""
            status = getattr(engine, "status", None)
            if callable(status):
                reason = " " + str(status().get("reason", ""))
            if plan.model.engine == "triposr":
                st = runtime.torch_status()
                reason = f" PyTorch is not importable: {st.error}" if not st.installed else ""
            raise StudioError(f"the {plan.model.display_name} engine is not available.{reason}")
        if not plan.installed:
            raise StudioError(
                f"{plan.model.display_name} is not downloaded "
                f"({plan.model.download_bytes / GIB:.2f} GiB, {plan.model.licence}): run "
                f"`shapestudio download-model {plan.model.name} --yes` or use Settings."
            )
        tool_name = None
        if remove_background:
            tool = TOOL_MODELS.get(bg_model) if bg_model else default_tool_model("background")
            if tool is None:
                raise StudioError(f"unknown background model {bg_model!r}")
            if not self.models.tool_installed(tool) and not self._input_has_alpha(input_path):
                raise StudioError(
                    f"background removal needs {tool.display_name} "
                    f"({tool.file.size / 2**20:.0f} MB, "
                    f"{tool.licence}): `shapestudio download-model {tool.name} --yes`, or pass a "
                    "picture that already has a transparent background, or remove_background=false."
                )
            tool_name = tool.name
        job = Job(
            id=self.jobs.new_id(),
            tool="reconstruct",
            input_path=str(input_path),
            source_name=source_name,
            model=plan.model.name,
            tier=plan.model.tier,
            device=plan.device,
            preset=plan.preset.name,
            resolution=plan.resolution,
            foreground_ratio=plan.foreground_ratio,
            remove_background=remove_background,
            bg_model=tool_name,
            estimate_seconds=plan.estimate.seconds,
            estimate_basis=plan.estimate.basis,
            fit_note=plan.fit_note,
            warnings=list(plan.warnings),
            threads=threads or self.settings.default_threads,
            over_budget=plan.over_budget,
            project=project or self.current_project(),
            note=note or "",
        )
        self.worker.submit(job)
        return job

    @staticmethod
    def _input_has_alpha(path: Path) -> bool:
        try:
            from PIL import Image

            from shapestudio.engine.tsr.preprocess import has_alpha_cutout

            with Image.open(path) as im:
                return has_alpha_cutout(im)
        except Exception:
            return False

    def reconstruct_upload(self, data: bytes, name: str, **kw) -> Job:
        p = self.save_upload(data, name)
        return self.reconstruct(str(p), **kw)

    def _spec_for(self, job: Job):
        from shapestudio.engine.base import MeshSpec

        model = MODELS[job.model]
        tool = TOOL_MODELS.get(job.bg_model or "")
        out_dir = self.settings.out_dir / job.id
        return MeshSpec(
            input=Path(job.input_path),
            out_dir=out_dir,
            stem=job.id,
            model_paths=self.models.model_paths(model),
            device=job.device,
            resolution=job.resolution,
            foreground_ratio=job.foreground_ratio,
            remove_background=job.remove_background,
            bg_model_path=self.models.path_for(tool.file) if tool else None,
            bg_model_name=tool.name if tool else "",
            threads=job.threads,
        )

    # -- status -------------------------------------------------------------------------------

    def job_status(self, job_id: str) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if job is None:
            raise StudioError(f"no such job {job_id!r}")
        ahead = self.worker.queue_seconds_ahead(job_id) if job.status == "queued" else 0.0
        eta = job.eta_seconds(ahead)
        position = None
        if job.status == "queued":
            ids = [j.id for j in self.jobs.queued()]
            position = ids.index(job_id) + 1 if job_id in ids else None
        return {
            "id": job.id,
            "tool": job.tool,
            "title": job_title(job),
            "source_name": job.source_name,
            "status": job.status,
            "phase": job.phase,
            "progress": round(job.progress if job.status != "done" else 1.0, 3),
            "message": job.message,
            "summary": job.summary(position, eta, self.jobs.is_paused()),
            "eta_seconds": None if eta is None else round(eta),
            "eta_human": None if eta is None else human_duration(eta),
            "elapsed_seconds": round(job.elapsed_seconds()) if job.status == "running" else None,
            "estimate_seconds": round(job.estimate_seconds),
            "estimate_basis": job.estimate_basis,
            "queue_position": position,
            "model": job.model,
            "tier": job.tier,
            "device": job.device,
            "preset": job.preset,
            "resolution": job.resolution,
            "foreground_ratio": job.foreground_ratio,
            "remove_background": job.remove_background,
            "bg_model": job.bg_model,
            "fit_note": job.fit_note,
            "over_budget": job.over_budget,
            "warnings": job.warnings,
            "error": job.error,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "glb_path": job.glb_path,
            "obj_path": job.obj_path,
            "preview_path": job.preview_path,
            "cutout_path": job.cutout_path,
            "input_path": job.input_path,
            "vertices": job.vertices,
            "faces": job.faces,
            "total_seconds": job.total_seconds,
            "load_seconds": job.load_seconds,
            "network_seconds": job.network_seconds,
            "grid_seconds": job.grid_seconds,
            "peak_rss_gib": round(job.peak_rss_bytes / GIB, 2) if job.peak_rss_bytes else None,
            "favorite": job.favorite,
            "project": job.project,
            "note": job.note,
            "deleted_at": job.deleted_at,
            "threads": job.threads,
        }

    def job_result(self, job_id: str, as_base64: bool = False, fmt: str = "glb") -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if job is None:
            raise StudioError(f"no such job {job_id!r}")
        if job.status != "done":
            return {
                "id": job.id,
                "status": job.status,
                "ready": False,
                "status_detail": self.job_status(job_id),
            }
        if fmt not in ("glb", "obj"):
            raise StudioError("format must be glb or obj")
        path = Path((job.glb_path if fmt == "glb" else job.obj_path) or "")
        if not path.exists():
            raise StudioError(f"job {job_id} finished but its {fmt.upper()} is missing at {path}")
        out: dict[str, Any] = {
            "id": job.id,
            "status": "done",
            "ready": True,
            "format": fmt,
            "path": str(path),
            "mime_type": "model/gltf-binary" if fmt == "glb" else "model/obj",
            "glb_path": job.glb_path,
            "obj_path": job.obj_path,
            "preview_path": job.preview_path,
            "cutout_path": job.cutout_path,
            "vertices": job.vertices,
            "faces": job.faces,
            "bytes": path.stat().st_size,
            "model": job.model,
            "tier": job.tier,
            "device": job.device,
            "preset": job.preset,
            "resolution": job.resolution,
            "total_seconds": job.total_seconds,
            "fit_note": job.fit_note,
        }
        if as_base64:
            out["base64"] = base64.b64encode(path.read_bytes()).decode("ascii")
        return out

    def list_jobs(self, include_deleted: bool = False) -> list[dict[str, Any]]:
        return [
            {
                "id": j.id,
                "status": j.status,
                "title": job_title(j),
                "source_name": j.source_name,
                "model": j.model,
                "tier": j.tier,
                "device": j.device,
                "preset": j.preset,
                "resolution": j.resolution,
                "created_at": j.created_at,
                "finished_at": j.finished_at,
                "glb_path": j.glb_path,
                "obj_path": j.obj_path,
                "vertices": j.vertices,
                "faces": j.faces,
                "favorite": j.favorite,
                "project": j.project,
                "deleted_at": j.deleted_at,
                "total_seconds": j.total_seconds,
                "error": j.error,
            }
            for j in reversed(self.jobs.all(include_deleted=include_deleted))
        ]

    def cancel(self, job_id: str) -> bool:
        return self.worker.cancel(job_id)

    # -- gallery management -------------------------------------------------------------------

    def _job_files(self, job: Job) -> list[Path]:
        folder = self.settings.out_dir / job.id
        return [p for p in folder.glob("*") if p.is_file()] if folder.exists() else []

    def delete_job(self, job_id: str) -> dict[str, Any]:
        """Soft delete: the output folder moves to trash/ and the job is hidden."""
        job = self.jobs.get(job_id)
        if job is None:
            raise StudioError(f"no such job {job_id!r}")
        if job.status in ("queued", "running"):
            self.cancel(job_id)
        src = self.settings.out_dir / job.id
        dst = self.settings.trash_dir / job.id
        if src.exists():
            self.settings.trash_dir.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                shutil.rmtree(dst, ignore_errors=True)
            src.replace(dst)
            self._repoint(job, src, dst)
        job.deleted_at = now_iso()
        self.jobs.put(job)
        return {"deleted": job_id, "trash": str(self.settings.trash_dir)}

    def _repoint(self, job: Job, old: Path, new: Path) -> None:
        for attr in ("glb_path", "obj_path", "preview_path", "thumb_path", "cutout_path"):
            v = getattr(job, attr)
            if v and Path(v).parent == old:
                setattr(job, attr, str(new / Path(v).name))

    def restore_job(self, job_id: str) -> dict[str, Any]:
        job = next((j for j in self.jobs.all(include_deleted=True) if j.id == job_id), None)
        if job is None or not job.deleted_at:
            raise StudioError(f"{job_id!r} is not in the trash")
        src = self.settings.trash_dir / job.id
        dst = self.settings.out_dir / job.id
        if src.exists():
            src.replace(dst)
            self._repoint(job, src, dst)
        job.deleted_at = None
        self.jobs.put(job)
        return {"restored": job_id}

    def trash(self) -> list[dict[str, Any]]:
        return [r for r in self.list_jobs(include_deleted=True) if r["deleted_at"]]

    def empty_trash(self) -> int:
        n = 0
        for j in self.jobs.all(include_deleted=True):
            if j.deleted_at:
                shutil.rmtree(self.settings.trash_dir / j.id, ignore_errors=True)
                self.jobs.forget(j.id)
                n += 1
        return n

    def set_favorite(self, job_id: str, value: bool) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if job is None:
            raise StudioError(f"no such job {job_id!r}")
        job.favorite = bool(value)
        self.jobs.put(job)
        return {"id": job_id, "favorite": job.favorite}

    def set_note(self, job_id: str, note: str) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if job is None:
            raise StudioError(f"no such job {job_id!r}")
        job.note = (note or "").strip()[:500]
        self.jobs.put(job)
        return {"id": job_id, "note": job.note}

    # -- projects -----------------------------------------------------------------------------

    def _read_projects(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self.settings.projects_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [p for p in data if isinstance(p, dict) and p.get("id") and p.get("name")]

    def _write_projects(self, items: list[dict[str, Any]]) -> None:
        tmp = self.settings.projects_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(items, indent=2), encoding="utf-8")
        tmp.replace(self.settings.projects_file)

    def projects(self) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        for j in self.jobs.all():
            if j.project and j.status == "done":
                counts[j.project] = counts.get(j.project, 0) + 1
        return [{**p, "count": counts.get(p["id"], 0)} for p in self._read_projects()]

    def project(self, project_id: str) -> dict[str, Any] | None:
        return next((p for p in self._read_projects() if p["id"] == project_id), None)

    def add_project(self, name: str, description: str = "") -> dict[str, Any]:
        name = " ".join((name or "").split())
        if not name:
            raise StudioError("a project needs a name")
        items = self._read_projects()
        if any(p["name"].lower() == name.lower() for p in items):
            raise StudioError(f"a project called {name!r} already exists")
        entry = {
            "id": "prj_" + secrets.token_hex(4),
            "name": name[:80],
            "description": (description or "").strip()[:500],
            "created_at": now_iso(),
        }
        items.append(entry)
        self._write_projects(items)
        self.set_current_project(entry["id"])
        return {**entry, "count": 0}

    def update_project(
        self, project_id: str, name: str | None = None, description: str | None = None
    ):
        items = self._read_projects()
        entry = next((p for p in items if p["id"] == project_id), None)
        if entry is None:
            raise StudioError(f"no such project {project_id!r}")
        if name is not None:
            name = " ".join(name.split())
            if not name:
                raise StudioError("a project needs a name")
            entry["name"] = name[:80]
        if description is not None:
            entry["description"] = description.strip()[:500]
        self._write_projects(items)
        return entry

    def remove_project(self, project_id: str) -> dict[str, Any]:
        items = self._read_projects()
        if not any(p["id"] == project_id for p in items):
            raise StudioError(f"no such project {project_id!r}")
        self._write_projects([p for p in items if p["id"] != project_id])
        unfiled = 0
        for j in self.jobs.all(include_deleted=True):
            if j.project == project_id:
                j.project = None
                self.jobs.put(j)
                unfiled += 1
        if self.current_project() == project_id:
            self.set_current_project(None)
        return {"removed": project_id, "unfiled": unfiled}

    def current_project(self) -> str | None:
        pid = self.settings.config().get("current_project")
        if not pid:
            return None
        return pid if any(p["id"] == pid for p in self._read_projects()) else None

    def set_current_project(self, project_id: str | None) -> dict[str, Any]:
        if project_id and not any(p["id"] == project_id for p in self._read_projects()):
            raise StudioError(f"no such project {project_id!r}")
        cfg = self.settings.config()
        if project_id:
            cfg["current_project"] = project_id
        else:
            cfg.pop("current_project", None)
        self.settings.write_config(cfg)
        return {"current_project": project_id or None}

    def set_job_project(self, job_id: str, project_id: str | None) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if job is None:
            raise StudioError(f"no such job {job_id!r}")
        if project_id and not any(p["id"] == project_id for p in self._read_projects()):
            raise StudioError(f"no such project {project_id!r}")
        job.project = project_id or None
        self.jobs.put(job)
        return {"id": job_id, "project": job.project}

    def export_project(self, project_id: str) -> tuple[str, bytes]:
        entry = self.project(project_id)
        if entry is None:
            raise StudioError(f"no such project {project_id!r}")
        jobs = [j for j in self.jobs.all() if j.project == project_id and j.status == "done"]
        if not jobs:
            raise StudioError("this project has no finished meshes yet")
        return self._zip_of(jobs, entry["name"], entry)

    def export_jobs(self, job_ids: list[str], title: str = "meshes") -> tuple[str, bytes]:
        jobs = [j for jid in job_ids if (j := self.jobs.get(str(jid))) and j.status == "done"]
        if not jobs:
            raise StudioError("none of the chosen meshes is finished")
        return self._zip_of(jobs, title, {"name": title})

    def _zip_of(self, jobs: list[Job], title: str, entry: dict[str, Any]) -> tuple[str, bytes]:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", title).strip("-") or "meshes"
        buf = io.BytesIO()
        manifest = []
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for n, j in enumerate(jobs, 1):
                folder = f"{n:03d}-{Path(j.source_name).stem[:24] or j.id}"
                for attr in ("glb_path", "obj_path", "preview_path", "cutout_path", "input_path"):
                    v = getattr(j, attr)
                    if v and Path(v).exists():
                        z.write(v, f"{folder}/{Path(v).name}")
                manifest.append(
                    {
                        "folder": folder,
                        "job": j.id,
                        "source": j.source_name,
                        "model": j.model,
                        "tier": j.tier,
                        "device": j.device,
                        "preset": j.preset,
                        "resolution": j.resolution,
                        "vertices": j.vertices,
                        "faces": j.faces,
                        "seconds": j.total_seconds,
                        "note": j.note,
                    }
                )
            z.writestr(
                "project.json",
                json.dumps(
                    {**entry, "app": f"Hivemind Shape Studio {__version__}", "meshes": manifest},
                    indent=2,
                ),
            )
        return f"{slug}.zip", buf.getvalue()

    # -- components, downloads, runtime -------------------------------------------------------

    def components(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in MODELS.values():
            files = [self.models.file_info(f) for f in m.files]
            out.append(
                {
                    "kind": "model",
                    "name": m.name,
                    "display_name": m.display_name,
                    "tier": m.tier,
                    "licence": m.licence,
                    "size": m.download_bytes,
                    "bytes": sum(f["bytes"] for f in files),
                    "installed": all(f["installed"] for f in files),
                    "path": str(self.settings.models_dir / m.folder),
                    "files": files,
                    "verified": m.verified,
                    "note": m.note,
                    "homepage": m.homepage,
                    "in_use": any(
                        j.model == m.name and j.status in ("queued", "running")
                        for j in self.jobs.all()
                    ),
                }
            )
        for t in TOOL_MODELS.values():
            info = self.models.file_info(t.file)
            out.append(
                {
                    "kind": "tool",
                    "name": t.name,
                    "display_name": t.display_name,
                    "tier": "tool",
                    "licence": t.licence,
                    "size": t.file.size,
                    "bytes": info["bytes"],
                    "installed": info["installed"],
                    "path": info["path"],
                    "files": [info],
                    "note": t.note,
                    "default": t.default,
                    "in_use": False,
                }
            )
        rt = runtime.report(self.settings)
        out.append(
            {
                "kind": "runtime",
                "name": "torch",
                "display_name": f"PyTorch {rt['torch'].get('version') or '(not importable)'}",
                "tier": "runtime",
                "licence": "BSD-3-Clause",
                "installed": rt["torch"]["installed"],
                "flavor": rt["torch"]["flavor"],
                "path": rt["torch"].get("location"),
                "cuda": rt["torch"]["cuda_available"],
                "device": rt["torch"].get("device_name"),
                "cuda_install": rt["cuda_install"],
                "nvidia_gpu": cuda_gpu(self.hardware) is not None,
                "runtime_installed": rt["runtime_installed"],
                "in_use": False,
            }
        )
        return out

    def remove_component(self, kind: str, name: str) -> dict[str, Any]:
        if kind == "model":
            m = MODELS.get(name)
            if m is None:
                raise StudioError(f"unknown model {name!r}")
            if any(j.model == name and j.status in ("queued", "running") for j in self.jobs.all()):
                raise StudioError(f"{m.display_name} is in use by a queued or running job")
            freed = self.models.remove_files(m.files)
            shutil.rmtree(self.settings.models_dir / m.folder, ignore_errors=True)
            return {"removed": name, "freed_bytes": freed}
        if kind == "tool":
            t = TOOL_MODELS.get(name)
            if t is None:
                raise StudioError(f"unknown tool model {name!r}")
            return {"removed": name, "freed_bytes": self.models.remove_files([t.file])}
        if kind == "runtime":
            return {
                "removed": "cuda runtime",
                "freed_bytes": runtime.remove_cuda_runtime(self.settings),
            }
        raise StudioError(f"unknown component kind {kind!r}")

    def download_info(self, name: str) -> dict[str, Any]:
        """Exact files, sizes and licence before any consent."""
        m = MODELS.get(name)
        if m is not None:
            missing = [f for f in m.files if not self.models.file_installed(f)]
            return {
                "model": m.name,
                "kind": "model",
                "tier": m.tier,
                "display_name": m.display_name,
                "licence": m.licence,
                "files": [
                    {
                        "filename": f.filename,
                        "size_gib": round(f.size / GIB, 3),
                        "source": f.source,
                        "sha256": f.sha256,
                    }
                    for f in missing
                ],
                "total_gib": round(sum(f.size for f in missing) / GIB, 2),
                "total_bytes": sum(f.size for f in missing),
                "already_installed": not missing,
                "verified": m.verified,
                "note": m.note,
            }
        t = TOOL_MODELS.get(name)
        if t is not None:
            inst = self.models.tool_installed(t)
            return {
                "model": t.name,
                "kind": "tool",
                "display_name": t.display_name,
                "licence": t.licence,
                "files": []
                if inst
                else [
                    {
                        "filename": t.file.filename,
                        "size_gib": round(t.file.size / GIB, 3),
                        "source": t.file.source,
                        "sha256": t.file.sha256,
                    }
                ],
                "total_gib": 0.0 if inst else round(t.file.size / GIB, 2),
                "total_bytes": 0 if inst else t.file.size,
                "already_installed": inst,
            }
        raise StudioError(f"unknown model {name!r}; known: {', '.join([*MODELS, *TOOL_MODELS])}")

    def install(self, name: str, consent, progress=None) -> bool:
        """Blocking download with a consent callback (CLI, MCP)."""
        m = MODELS.get(name)
        if m is not None:
            return self.models.install_model(m, consent, progress)
        t = TOOL_MODELS.get(name)
        if t is not None:
            return self.models.install_tool(t, consent, progress)
        raise StudioError(f"unknown model {name!r}")

    def start_download(self, name: str) -> BackgroundTask:
        """Download in a background thread after the UI has shown consent."""
        info = self.download_info(name)
        key = f"download:{name}"
        existing = self.tasks.get(key)
        if existing and existing.status == "running":
            return existing
        task = BackgroundTask("download", info["display_name"])
        task.total_bytes = info["total_bytes"]
        self.tasks[key] = task
        files = MODELS[name].files if name in MODELS else (TOOL_MODELS[name].file,)

        def progress(fname: str, done: int, total: int) -> None:
            task.current_file = fname
            finished = sum(f.size for f in files if self.models.file_installed(f))
            task.done_bytes = min(task.total_bytes, finished + done) if task.total_bytes else done

        def run() -> None:
            try:
                self.install(name, lambda _d, _b: True, progress)
                task.done_bytes = task.total_bytes
                task.status, task.message = "done", "installed"
            except Exception as e:
                task.status, task.error = "failed", str(e)
            task.finished_at = time.time()

        threading.Thread(target=run, name=key, daemon=True).start()
        return task

    def start_runtime_install(self) -> BackgroundTask:
        key = "runtime:cuda"
        existing = self.tasks.get(key)
        if existing and existing.status == "running":
            return existing
        plan = runtime.cuda_install_plan(self.settings)
        if not plan["possible"]:
            raise StudioError(plan["reason"])
        task = BackgroundTask("runtime", "PyTorch CUDA runtime")
        task.total_bytes = int(plan.get("bytes") or 0)
        self.tasks[key] = task

        def progress(done: int, total: int) -> None:
            task.done_bytes, task.total_bytes = done, total or task.total_bytes

        def run() -> None:
            try:
                r = runtime.install_cuda_runtime(self.settings, lambda _d, _b: True, progress)
                task.result = r
                task.status = "done" if r.get("installed") else "failed"
                task.message = r.get("message", "installed")
                self._cuda_runtime = runtime.torch_status(refresh=True).cuda_available
                self.planner.cuda_runtime = self._cuda_runtime
            except Exception as e:
                task.status, task.error = "failed", str(e)
            task.finished_at = time.time()

        threading.Thread(target=run, name=key, daemon=True).start()
        return task

    def task_list(self) -> dict[str, dict[str, Any]]:
        return {k: t.to_dict() for k, t in self.tasks.items()}

    # -- benchmark ----------------------------------------------------------------------------

    def sample_image(self) -> Path:
        """A synthetic test object (a shaded sphere on white) drawn with Pillow, so the
        benchmark needs no download and every machine measures the same input."""
        from PIL import Image, ImageDraw

        p = self.settings.data_dir / "benchmark-input.png"
        if p.exists():
            return p
        size = 512
        img = Image.new("RGB", (size, size), (255, 255, 255))
        d = ImageDraw.Draw(img)
        cx, cy, r = size // 2, size // 2, size // 3
        for i in range(r, 0, -2):
            t = i / r
            shade = int(60 + 180 * (1 - t) ** 0.7)
            d.ellipse(
                (
                    cx - i - int(20 * t),
                    cy - i - int(20 * t),
                    cx + i - int(20 * t),
                    cy + i - int(20 * t),
                ),
                fill=(shade, int(shade * 0.55), int(shade * 0.25)),
            )
        img.save(p, "PNG")
        return p

    def benchmark(self, preset: str = "fast", wait: float = 2.0, on_status=None) -> dict[str, Any]:
        """One real reconstruction of the sample image, then write profile.json with what
        was measured and the expected time per preset from the new calibration."""
        job = self.reconstruct(str(self.sample_image()), preset=preset, note="benchmark")
        while True:
            st = self.job_status(job.id)
            if on_status:
                on_status(st)
            if st["status"] in FINAL:
                break
            time.sleep(wait)
        if st["status"] != "done":
            raise StudioError(f"benchmark {st['status']}: {st['error'] or st['message']}")
        model = MODELS[job.model]
        expected = {}
        for name in model.presets:
            try:
                expected[name] = round(self.plan(model.name, name).estimate.seconds)
            except StudioError:
                expected[name] = None
        profile = {
            "measured_at": now_iso(),
            "studio_version": __version__,
            "cpu": self.hardware.cpu_name,
            "physical_cores": self.hardware.physical_cores,
            "ram_total_gib": round(self.hardware.ram_total_bytes / GIB, 1),
            "gpu": (cuda_gpu(self.hardware).name if cuda_gpu(self.hardware) else None),
            "model": job.model,
            "device": job.device,
            "preset": job.preset,
            "resolution": job.resolution,
            "job_id": job.id,
            "load_seconds": st["load_seconds"],
            "network_seconds": st["network_seconds"],
            "grid_seconds": st["grid_seconds"],
            "wall_seconds": st["total_seconds"],
            "peak_rss_gib": st["peak_rss_gib"],
            "vertices": st["vertices"],
            "faces": st["faces"],
            "expected_seconds_per_preset": expected,
        }
        self.settings.profile_file.write_text(json.dumps(profile, indent=2), encoding="utf-8")
        return profile

    def start_benchmark(self, preset: str = "fast") -> BackgroundTask:
        key = "benchmark"
        existing = self.tasks.get(key)
        if existing and existing.status == "running":
            return existing
        task = BackgroundTask("benchmark", f"benchmark ({preset})")
        self.tasks[key] = task

        def status(st):
            task.message = st["summary"]
            task.result = {"job_id": st["id"]}

        def run() -> None:
            try:
                profile = self.benchmark(preset, wait=1.0, on_status=status)
                task.result = {"profile": profile, "job_id": profile["job_id"]}
                task.status, task.message = "done", "profile written"
            except Exception as e:
                task.status, task.error = "failed", str(e)
            task.finished_at = time.time()

        threading.Thread(target=run, name=key, daemon=True).start()
        return task

    def profile(self) -> dict[str, Any] | None:
        try:
            return json.loads(self.settings.profile_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    # -- config ------------------------------------------------------------------------------

    def get_config(self) -> dict[str, Any]:
        cfg = self.settings.config()
        import os

        return {
            "allow_over_budget": self.settings.allow_over_budget,
            "allow_over_budget_from_env": os.environ.get(ENV_ALLOW_OVER_BUDGET, "").lower()
            in ("1", "true", "yes"),
            "max_memory_gib": self.settings.max_memory_gib,
            "memory_budget_gib": round(
                memory_budget(self.hardware, "cpu", self._cap_bytes()) / GIB, 1
            ),
            "memory_budget_auto_gib": round(memory_budget(self.hardware, "cpu") / GIB, 1),
            "device": self.settings.device_preference,
            "default_threads": self.settings.default_threads,
            "paused": self.jobs.is_paused(),
            "current_project": self.current_project(),
            "runtime": cfg.get("runtime", "cpu"),
        }

    def set_config(self, changes: dict[str, Any]) -> dict[str, Any]:
        cfg = self.settings.config()
        if "allow_over_budget" in changes:
            cfg["allow_over_budget"] = bool(changes["allow_over_budget"])
        if "max_memory_gib" in changes:
            v = changes["max_memory_gib"]
            if v in (None, "", 0, "0"):
                cfg.pop("max_memory_gib", None)
            else:
                v = float(v)
                if not 2 <= v <= 1024:
                    raise StudioError("memory limit must be between 2 and 1024 GiB")
                cfg["max_memory_gib"] = v
        if "device" in changes:
            d = str(changes["device"] or "auto").lower()
            if d not in ("auto", "cpu", "cuda"):
                raise StudioError("device must be auto, cpu or cuda")
            if d == "auto":
                cfg.pop("device", None)
            else:
                cfg["device"] = d
        if "default_threads" in changes:
            v = changes["default_threads"]
            if v in (None, "", 0, "0"):
                cfg.pop("default_threads", None)
            else:
                cfg["default_threads"] = max(1, int(v))
        if "runtime" in changes:
            r = str(changes["runtime"]).lower()
            if r not in ("cpu", "cuda"):
                raise StudioError("runtime must be cpu or cuda")
            cfg["runtime"] = r
        if "paused" in changes:
            self.jobs.set_paused(bool(changes["paused"]))
        self.settings.write_config(cfg)
        self.planner.allow_over_budget = self.settings.allow_over_budget
        self.planner.cap_bytes = self._cap_bytes()
        self.planner.device_preference = self.settings.device_preference
        return self.get_config()

    # -- capabilities -------------------------------------------------------------------------

    def model_report(self, model: MeshModel) -> dict[str, Any]:
        cpu = fit_for(model, self.hardware, "cpu", 256, self._cap_bytes(), self._cuda_runtime)
        gpu = fit_for(model, self.hardware, "cuda", 256, self._cap_bytes(), self._cuda_runtime)
        device, chosen = self.planner.device_for(model, 256)
        engine = self.engines.get(model.engine)
        return {
            "name": model.name,
            "display_name": model.display_name,
            "tier": model.tier,
            "licence": model.licence,
            "quality_rank": model.quality_rank,
            "description": model.description,
            "homepage": model.homepage,
            "note": model.note,
            "verified": model.verified,
            "installed": self.models.model_installed(model),
            "download_gib": round(model.download_bytes / GIB, 2),
            "engine": model.engine,
            "engine_available": bool(engine and engine.available()),
            "engine_status": (engine.status() if engine and hasattr(engine, "status") else None),
            "fits_this_machine": chosen.fits,
            "device": device,
            "fit_note": None
            if chosen.fits
            else (
                chosen.reason
                or (
                    f"needs about {chosen.required_bytes / GIB:.1f} GiB; this machine's budget is "
                    f"{chosen.budget_bytes / GIB:.1f} GiB"
                )
            ),
            "cpu": {
                "possible": model.cpu_capable,
                "fits": cpu.fits,
                "needs_gib_at_256": round(cpu.required_bytes / GIB, 1),
                "budget_gib": round(cpu.budget_bytes / GIB, 1),
                "reason": cpu.reason,
            },
            "cuda": {
                "fits": gpu.fits,
                "needs_gib_at_256": round(gpu.required_bytes / GIB, 1),
                "budget_gib": round(gpu.budget_bytes / GIB, 1),
                "min_vram_gib": round(model.min_vram_bytes / GIB, 0),
                "reason": gpu.reason,
            },
            "presets": self.planner.preset_table(model),
        }

    def capabilities(self) -> dict[str, Any]:
        rt = runtime.report(self.settings)
        running = self.jobs.running()
        lock = self.worker.lock.info() or {}
        return {
            "studio_version": __version__,
            "policy": {
                "default_preset": "best",
                "default_model": DEFAULT_MODEL,
                "quality": "Quality is a setting, never a consequence: 'best' is the finest "
                "surface grid on the strongest permissively licensed model that fits this machine; "
                "the machine decides how long it takes. Nothing downgrades silently; when the "
                "automatic choice is not the top tier, fit_note says why.",
                "faster_only_with_consent": "'balanced' and 'fast' trade surface detail for "
                "extraction time and must be chosen explicitly; each preset states its trade-off.",
                "licences": "Permissive only (MIT, Apache-2.0, BSD). Non-commercial and "
                "revenue-gated weights are excluded by policy and listed as such.",
                "network": "Models and the GPU runtime download once, each after explicit "
                "consent; reconstruction is fully offline.",
                "measured": "Every timing comes from this machine's own runs; until the first "
                "run, estimates are labelled 'not measured yet'.",
            },
            "hardware": self.hardware.to_dict(),
            "runtime": rt,
            "engines": {
                name: {"name": e.name, "available": e.available(), "version": e.version()}
                | ({"status": e.status()} if hasattr(e, "status") else {})
                for name, e in self.engines.items()
            },
            "background_engine": {
                "name": self.background_engine.name,
                "available": self.background_engine.available(),
                "version": self.background_engine.version(),
                "models": [
                    {
                        "name": t.name,
                        "display_name": t.display_name,
                        "licence": t.licence,
                        "size_mb": round(t.file.size / 2**20),
                        "installed": self.models.tool_installed(t),
                        "default": t.default,
                        "note": t.note,
                    }
                    for t in TOOL_MODELS.values()
                ],
            },
            "models": [self.model_report(m) for m in MODELS.values()],
            "excluded_by_policy": [{"model": m, "reason": r} for m, r in EXCLUDED_BY_POLICY],
            "licences": list(LICENCES),
            "paths": {
                "data_dir": str(self.settings.data_dir),
                "models_dir": str(self.settings.models_dir),
                "out_dir": str(self.settings.out_dir),
                "runtime_dir": str(self.settings.runtime_dir),
            },
            "profile": self.profile(),
            "config": self.get_config(),
            "queue": {
                "running": running.id if running else None,
                "queued": len(self.jobs.queued()),
                "worker_in_this_process": self.worker_active,
                "paused": self.jobs.is_paused(),
                "worker_pid": self.worker_owner_pid,
                "worker_version": lock.get("version"),
                "worker_started_at": lock.get("started_at"),
                "worker_outdated": bool(lock and lock.get("version") != __version__),
            },
        }

    # -- misc helpers for the UI --------------------------------------------------------------

    def memory_users(self, limit: int = 8) -> dict[str, Any]:
        import psutil

        vm = psutil.virtual_memory()
        rows = []
        for p in psutil.process_iter(["pid", "name", "memory_info"]):
            try:
                rss = p.info["memory_info"].rss if p.info["memory_info"] else 0
            except (psutil.Error, AttributeError):
                continue
            rows.append(
                {"pid": p.info["pid"], "name": p.info["name"] or "?", "gib": round(rss / GIB, 2)}
            )
        rows.sort(key=lambda r: -r["gib"])
        return {
            "total_gib": round(vm.total / GIB, 1),
            "available_gib": round(vm.available / GIB, 1),
            "top": rows[:limit],
            "measured_at": now_iso(),
        }

    def reveal(self, path: str) -> None:
        """Open the system file manager on a path inside the data dir (never elsewhere)."""
        import os
        import subprocess
        import sys

        p = Path(path).resolve()
        if (
            self.settings.data_dir.resolve() not in p.parents
            and p != self.settings.data_dir.resolve()
        ):
            raise StudioError("only paths inside the studio's data folder can be shown")
        target = p if p.is_dir() else p.parent
        with contextlib.suppress(Exception):
            if sys.platform == "win32":
                if p.is_file():
                    subprocess.Popen(["explorer", "/select,", str(p)])
                else:
                    os.startfile(str(target))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R" if p.is_file() else "", str(p)])
            else:
                subprocess.Popen(["xdg-open", str(target)])
