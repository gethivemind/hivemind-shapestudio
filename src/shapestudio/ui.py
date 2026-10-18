"""The studio's own local web UI: one HTML page plus a small JSON API over the same Studio
the MCP tools use. Home, Create, Gallery (with a 3D viewer) and Settings."""
# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from shapestudio import DISPLAY_NAME, PRODUCT_NAME, __version__
from shapestudio.service import Studio, StudioError

STATIC = Path(__file__).parent / "static"
FILE_KINDS = {
    "glb": ("glb_path", "model/gltf-binary"),
    "obj": ("obj_path", "model/obj"),
    "preview": ("preview_path", "image/png"),
    "thumb": ("thumb_path", "image/png"),
    "cutout": ("cutout_path", "image/png"),
    "input": ("input_path", None),
}


def _draw_icon(size: int) -> bytes:
    """The studio mark (hexagon with a faceted cube) as a PNG, drawn at 4x and downsampled."""
    from PIL import Image, ImageDraw

    s = size * 4
    u = s / 64
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, s - 1, s - 1), radius=int(14 * u), fill="#0f1012")
    hexagon = [(32, 6), (54, 19), (54, 45), (32, 58), (10, 45), (10, 19)]
    d.polygon([(x * u, y * u) for x, y in hexagon], fill="#f0b429")
    # an isometric cube: top, left, right faces
    top = [(32, 20), (44, 27), (32, 34), (20, 27)]
    left = [(20, 27), (32, 34), (32, 48), (20, 41)]
    right = [(32, 34), (44, 27), (44, 41), (32, 48)]
    d.polygon([(x * u, y * u) for x, y in top], fill="#f7f6f2")
    d.polygon([(x * u, y * u) for x, y in left], fill="#1b1b19")
    d.polygon([(x * u, y * u) for x, y in right], fill="#3a3a37")
    out = io.BytesIO()
    img.resize((size, size), Image.LANCZOS).save(out, "PNG", optimize=True)
    return out.getvalue()


ICON_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
    "<path d='M32 6 54 19v26L32 58 10 45V19z' fill='#f0b429'/>"
    "<path d='M32 20l12 7-12 7-12-7z' fill='#f7f6f2'/>"
    "<path d='M20 27l12 7v14l-12-7z' fill='#1b1b19'/>"
    "<path d='M44 27l-12 7v14l12-7z' fill='#3a3a37'/></svg>"
)


def _int(v: Any) -> int | None:
    return int(v) if v not in (None, "", "auto") else None


def _float(v: Any) -> float | None:
    return float(v) if v not in (None, "") else None


def _str(v: Any) -> str | None:
    return v if v not in (None, "", "auto") else None


def _bool(v: Any, default: bool = True) -> bool:
    if v is None or v == "":
        return default
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "on")


def build_app(studio: Studio) -> Starlette:
    def err(e: Exception, code: int = 400) -> JSONResponse:
        return JSONResponse({"error": str(e)}, status_code=code)

    async def index(_: Request) -> HTMLResponse:
        return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))

    async def manifest(_: Request) -> JSONResponse:
        return JSONResponse(
            {
                "name": DISPLAY_NAME,
                "short_name": "Shape Studio",
                "start_url": "/",
                "display": "standalone",
                "background_color": "#0f1012",
                "theme_color": "#0f1012",
                "icons": [
                    {"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml"},
                    {"src": "/icon-192.png", "sizes": "192x192", "type": "image/png"},
                    {
                        "src": "/icon-512.png",
                        "sizes": "512x512",
                        "type": "image/png",
                        "purpose": "any maskable",
                    },
                ],
            },
            media_type="application/manifest+json",
        )

    async def icon(_: Request) -> Response:
        return Response(ICON_SVG, media_type="image/svg+xml")

    _png_icons: dict[int, bytes] = {}

    async def icon_png(request: Request) -> Response:
        size = max(16, min(1024, int(request.path_params["size"])))
        if size not in _png_icons:
            _png_icons[size] = _draw_icon(size)
        return Response(_png_icons[size], media_type="image/png")

    async def about(_: Request) -> JSONResponse:
        import sys

        return JSONResponse(
            {
                "product": PRODUCT_NAME,
                "display_name": DISPLAY_NAME,
                "version": __version__,
                "credit": "made with ♥ by Ratul Roy (royal-programmer)",
                "licence": "Apache-2.0",
                "homepage": "https://gethivemind.github.io/hivemind-shapestudio/",
                "source": "https://github.com/gethivemind/hivemind-shapestudio",
                "mcp_command": (
                    f"{sys.executable} cli serve"
                    if getattr(sys, "frozen", False)
                    else "shapestudio serve"
                ),
            }
        )

    # -- capabilities, plan, config ---------------------------------------------------------

    async def capabilities(_: Request) -> JSONResponse:
        return JSONResponse(await run_in_threadpool(studio.capabilities))

    async def plan(request: Request) -> JSONResponse:
        q = request.query_params
        try:
            p = studio.plan(
                _str(q.get("tier")) or _str(q.get("model")),
                q.get("preset", "best"),
                _int(q.get("resolution")),
                _float(q.get("foreground")),
                _bool(q.get("remove_background"), True),
            )
        except StudioError as e:
            return err(e)
        return JSONResponse(p.to_dict())

    async def get_config(_: Request) -> JSONResponse:
        return JSONResponse(studio.get_config())

    async def set_config(request: Request) -> JSONResponse:
        try:
            return JSONResponse(studio.set_config(await request.json()))
        except (StudioError, ValueError) as e:
            return err(e)

    # -- jobs -------------------------------------------------------------------------------

    def _job_kwargs(form: Any) -> dict[str, Any]:
        return {
            "model": _str(form.get("tier")) or _str(form.get("model")),
            "preset": form.get("preset") or "best",
            "resolution": _int(form.get("resolution")),
            "foreground_ratio": _float(form.get("foreground")),
            "remove_background": _bool(form.get("remove_background"), True),
            "bg_model": _str(form.get("bg_model")),
            "threads": _int(form.get("threads")),
            "project": _str(form.get("project")),
            "note": form.get("note") or "",
        }

    async def reconstruct(request: Request) -> JSONResponse:
        ctype = request.headers.get("content-type", "")
        try:
            if ctype.startswith("multipart/form-data"):
                form = await request.form()
                upload = form.get("file")
                kw = _job_kwargs(form)
                if upload is not None and hasattr(upload, "read"):
                    data = await upload.read()
                    job = await run_in_threadpool(
                        studio.reconstruct_upload, data, upload.filename or "upload.png", **kw
                    )
                elif form.get("image"):
                    job = await run_in_threadpool(studio.reconstruct, form.get("image"), **kw)
                else:
                    return err(StudioError("send a file or an image path"))
            else:
                body = await request.json()
                if not body.get("image"):
                    return err(StudioError("image path required"))
                job = await run_in_threadpool(
                    studio.reconstruct, body["image"], **_job_kwargs(body)
                )
        except StudioError as e:
            return err(e)
        return JSONResponse(studio.job_status(job.id))

    async def upload(request: Request) -> JSONResponse:
        form = await request.form()
        f = form.get("file")
        if f is None or not hasattr(f, "read"):
            return err(StudioError("no file"))
        try:
            p = studio.save_upload(await f.read(), f.filename or "upload.png")
        except StudioError as e:
            return err(e)
        return JSONResponse({"path": str(p), "name": f.filename})

    async def jobs(request: Request) -> JSONResponse:
        rows = [studio.job_status(r["id"]) for r in studio.list_jobs()]
        return JSONResponse(
            {
                "jobs": rows,
                "worker_in_this_process": studio.worker_active,
                "worker_pid": studio.worker_owner_pid,
                "paused": studio.jobs.is_paused(),
            }
        )

    async def job(request: Request) -> JSONResponse:
        try:
            return JSONResponse(studio.job_status(request.path_params["job_id"]))
        except StudioError as e:
            return err(e, 404)

    async def cancel(request: Request) -> JSONResponse:
        return JSONResponse({"cancelled": studio.cancel(request.path_params["job_id"])})

    async def delete_job(request: Request) -> JSONResponse:
        try:
            return JSONResponse(studio.delete_job(request.path_params["job_id"]))
        except StudioError as e:
            return err(e, 404)

    async def restore_job(request: Request) -> JSONResponse:
        try:
            return JSONResponse(studio.restore_job(request.path_params["job_id"]))
        except StudioError as e:
            return err(e, 404)

    async def favorite(request: Request) -> JSONResponse:
        body = await request.json()
        try:
            return JSONResponse(
                studio.set_favorite(request.path_params["job_id"], bool(body.get("favorite")))
            )
        except StudioError as e:
            return err(e, 404)

    async def note(request: Request) -> JSONResponse:
        body = await request.json()
        try:
            return JSONResponse(
                studio.set_note(request.path_params["job_id"], body.get("note", ""))
            )
        except StudioError as e:
            return err(e, 404)

    async def job_project(request: Request) -> JSONResponse:
        body = await request.json()
        try:
            return JSONResponse(
                studio.set_job_project(request.path_params["job_id"], _str(body.get("project")))
            )
        except StudioError as e:
            return err(e)

    async def file(request: Request) -> Response:
        kind = request.path_params["kind"]
        if kind not in FILE_KINDS:
            return err(StudioError(f"unknown file kind {kind!r}"), 404)
        job_id = request.path_params["job_id"]
        j = next((x for x in studio.jobs.all(include_deleted=True) if x.id == job_id), None)
        if j is None:
            return err(StudioError("no such job"), 404)
        attr, media = FILE_KINDS[kind]
        p = getattr(j, attr)
        if not p or not Path(p).exists():
            return err(StudioError(f"no {kind} for this job"), 404)
        p = Path(p)
        headers = {}
        if request.query_params.get("download"):
            stem = Path(j.source_name).stem or j.id
            headers["Content-Disposition"] = f'attachment; filename="{stem}-{j.id}{p.suffix}"'
        return FileResponse(str(p), media_type=media, headers=headers)

    async def trash(_: Request) -> JSONResponse:
        return JSONResponse({"jobs": studio.trash()})

    async def empty_trash(_: Request) -> JSONResponse:
        return JSONResponse({"emptied": studio.empty_trash()})

    async def export_jobs(request: Request) -> Response:
        body = await request.json()
        try:
            name, data = studio.export_jobs(
                [str(i) for i in body.get("ids", [])], body.get("title") or "meshes"
            )
        except StudioError as e:
            return err(e)
        return Response(
            data,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{name}"'},
        )

    # -- projects ---------------------------------------------------------------------------

    async def projects(_: Request) -> JSONResponse:
        return JSONResponse({"projects": studio.projects(), "current": studio.current_project()})

    async def project_add(request: Request) -> JSONResponse:
        body = await request.json()
        try:
            return JSONResponse(
                studio.add_project(body.get("name", ""), body.get("description", ""))
            )
        except StudioError as e:
            return err(e)

    async def project_update(request: Request) -> JSONResponse:
        body = await request.json()
        try:
            return JSONResponse(
                studio.update_project(body.get("id", ""), body.get("name"), body.get("description"))
            )
        except StudioError as e:
            return err(e)

    async def project_remove(request: Request) -> JSONResponse:
        body = await request.json()
        try:
            return JSONResponse(studio.remove_project(body.get("id", "")))
        except StudioError as e:
            return err(e)

    async def project_current(request: Request) -> JSONResponse:
        body = await request.json()
        try:
            return JSONResponse(studio.set_current_project(_str(body.get("id"))))
        except StudioError as e:
            return err(e)

    async def project_export(request: Request) -> Response:
        try:
            name, data = studio.export_project(request.path_params["project_id"])
        except StudioError as e:
            return err(e)
        return Response(
            data,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{name}"'},
        )

    # -- components, downloads, runtime, benchmark -----------------------------------------

    async def components(_: Request) -> JSONResponse:
        return JSONResponse({"components": await run_in_threadpool(studio.components)})

    async def remove_component(request: Request) -> JSONResponse:
        body = await request.json()
        try:
            return JSONResponse(studio.remove_component(body.get("kind", ""), body.get("name", "")))
        except StudioError as e:
            return err(e)

    async def reveal(request: Request) -> JSONResponse:
        body = await request.json()
        try:
            studio.reveal(body.get("path", ""))
        except StudioError as e:
            return err(e)
        return JSONResponse({"ok": True})

    async def download_info(request: Request) -> JSONResponse:
        try:
            return JSONResponse(studio.download_info(request.query_params.get("name", "")))
        except StudioError as e:
            return err(e)

    async def start_download(request: Request) -> JSONResponse:
        body = await request.json()
        if not body.get("consent"):
            return err(StudioError("consent required"))
        try:
            return JSONResponse(studio.start_download(body.get("name", "")).to_dict())
        except StudioError as e:
            return err(e)

    async def runtime_info(_: Request) -> JSONResponse:
        from shapestudio import runtime

        return JSONResponse(await run_in_threadpool(runtime.report, studio.settings))

    async def runtime_install(request: Request) -> JSONResponse:
        body = await request.json()
        if not body.get("consent"):
            return err(StudioError("consent required"))
        try:
            return JSONResponse(studio.start_runtime_install().to_dict())
        except StudioError as e:
            return err(e)

    async def start_benchmark(request: Request) -> JSONResponse:
        body = await request.json() if await request.body() else {}
        try:
            return JSONResponse(studio.start_benchmark(body.get("preset", "fast")).to_dict())
        except StudioError as e:
            return err(e)

    async def tasks(_: Request) -> JSONResponse:
        return JSONResponse(studio.task_list())

    async def memory(_: Request) -> JSONResponse:
        return JSONResponse(await run_in_threadpool(studio.memory_users))

    async def worker_handover(_: Request) -> JSONResponse:
        return JSONResponse(studio.request_worker_handover())

    async def worker_start(_: Request) -> JSONResponse:
        return JSONResponse({"started": studio.try_start_worker(), "active": studio.worker_active})

    return Starlette(
        routes=[
            Route("/", index),
            Route("/manifest.webmanifest", manifest),
            Route("/icon.svg", icon),
            Route("/icon-{size}.png", icon_png),
            Route("/api/about", about),
            Route("/api/capabilities", capabilities),
            Route("/api/plan", plan),
            Route("/api/config", get_config),
            Route("/api/config", set_config, methods=["POST"]),
            Route("/api/reconstruct", reconstruct, methods=["POST"]),
            Route("/api/upload", upload, methods=["POST"]),
            Route("/api/jobs", jobs),
            Route("/api/jobs/{job_id}", job),
            Route("/api/jobs/{job_id}", delete_job, methods=["DELETE"]),
            Route("/api/jobs/{job_id}/cancel", cancel, methods=["POST"]),
            Route("/api/jobs/{job_id}/restore", restore_job, methods=["POST"]),
            Route("/api/jobs/{job_id}/favorite", favorite, methods=["POST"]),
            Route("/api/jobs/{job_id}/note", note, methods=["POST"]),
            Route("/api/jobs/{job_id}/project", job_project, methods=["POST"]),
            Route("/api/files/{job_id}/{kind}", file),
            Route("/api/trash", trash),
            Route("/api/trash/empty", empty_trash, methods=["POST"]),
            Route("/api/export", export_jobs, methods=["POST"]),
            Route("/api/projects", projects),
            Route("/api/projects", project_add, methods=["POST"]),
            Route("/api/projects/update", project_update, methods=["POST"]),
            Route("/api/projects/remove", project_remove, methods=["POST"]),
            Route("/api/projects/current", project_current, methods=["POST"]),
            Route("/api/projects/{project_id}/export", project_export),
            Route("/api/components", components),
            Route("/api/components/remove", remove_component, methods=["POST"]),
            Route("/api/reveal", reveal, methods=["POST"]),
            Route("/api/models/download", download_info),
            Route("/api/models/download", start_download, methods=["POST"]),
            Route("/api/runtime", runtime_info),
            Route("/api/runtime/install", runtime_install, methods=["POST"]),
            Route("/api/benchmark", start_benchmark, methods=["POST"]),
            Route("/api/tasks", tasks),
            Route("/api/memory", memory),
            Route("/api/worker/handover", worker_handover, methods=["POST"]),
            Route("/api/worker/start", worker_start, methods=["POST"]),
        ],
    )


def serve_ui(host: str = "127.0.0.1", port: int = 8776, start_worker: bool = True) -> None:
    import uvicorn

    studio = Studio(start_worker=start_worker)
    mode = "running jobs" if studio.worker_active else "submitting to the running worker"
    print(f"{PRODUCT_NAME} UI at http://{host}:{port}  ({mode})", flush=True)
    try:
        uvicorn.run(build_app(studio), host=host, port=port, log_level="warning")
    finally:
        studio.close()


__all__ = ["build_app", "json", "serve_ui"]
