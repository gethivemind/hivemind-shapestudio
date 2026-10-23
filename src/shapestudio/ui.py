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


BRAND = STATIC / "brand"


def _draw_icon(size: int) -> bytes:
    """The Presidium Shape Studio tile (coral, light mark) as a PNG, drawn at 8x and
    downsampled; same geometry as static/brand/shape-tile.svg from the brand kit."""
    from PIL import Image, ImageDraw

    s = size * 8
    u = s / 64
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, s - 1, s - 1), radius=int(14 * u), fill="#D5582F")
    ink = "#F4F4F5"
    for cx, cy in ((53.1, 18.2), (58.3, 41), (43.7, 59.3), (20.3, 59.3), (5.7, 41), (10.9, 18.2)):
        d.ellipse(((cx - 3.6) * u, (cy - 3.6) * u, (cx + 3.6) * u, (cy + 3.6) * u), fill=ink)
    d.ellipse(((32 - 5.5) * u, (8 - 5.5) * u, (32 + 5.5) * u, (8 + 5.5) * u), fill=ink)
    outline = [(32, 24), (42, 29.5), (42, 40.5), (32, 46), (22, 40.5), (22, 29.5)]
    pts = [(x * u, y * u) for x, y in outline]
    d.line([*pts, pts[0]], fill=ink, width=max(1, round(3 * u)), joint="curve")
    for seg in ([(22, 29.5), (32, 35), (42, 29.5)], [(32, 35), (32, 46)]):
        d.line(
            [(x * u, y * u) for x, y in seg], fill=ink, width=max(1, round(2.6 * u)), joint="curve"
        )
    r = 1.5 * u
    for x, y in [*outline, (22, 29.5), (42, 29.5), (32, 46)]:
        d.ellipse((x * u - r, y * u - r, x * u + r, y * u + r), fill=ink)
    out = io.BytesIO()
    img.resize((size, size), Image.LANCZOS).save(out, "PNG", optimize=True)
    return out.getvalue()


def _brand_svg(name: str) -> str:
    return (BRAND / f"{name}.svg").read_text(encoding="utf-8")


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
                "description": "Presidium Shape Studio: free, local image-to-3D.",
                "start_url": "/",
                "display": "standalone",
                "background_color": "#0f1012",
                "theme_color": "#D5582F",
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
        return Response(_brand_svg("shape-tile"), media_type="image/svg+xml")

    async def brand(request: Request) -> Response:
        name = request.path_params["name"]
        if name not in (
            "shape-tile",
            "shape-mark",
            "shape-mono",
            "shape-mono-light",
            "shape-spinner",
        ):
            return JSONResponse({"error": "no such brand asset"}, status_code=404)
        return Response(_brand_svg(name), media_type="image/svg+xml")

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
                "family": "Presidium",
                "display_name": DISPLAY_NAME,
                "version": __version__,
                "credit": "made with ♥ by Ratul Roy (royal-programmer)",
                "licence": "Apache-2.0",
                "homepage": "https://presidiumdev.github.io/presidium-shapestudio/",
                "source": "https://github.com/presidiumdev/presidium-shapestudio",
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
                uploads = [u for u in form.getlist("file") if hasattr(u, "read")]
                kw = _job_kwargs(form)
                if len(uploads) > 1:
                    jobs_out = []
                    for u in uploads:
                        data = await u.read()
                        j = await run_in_threadpool(
                            studio.reconstruct_upload, data, u.filename or "upload.png", **kw
                        )
                        jobs_out.append(studio.job_status(j.id))
                    return JSONResponse({"jobs": jobs_out, **jobs_out[0]})
                upload = uploads[0] if uploads else None
                if upload is not None:
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

    async def derived(request: Request) -> Response:
        try:
            p = studio.derived_path(request.path_params["job_id"], request.path_params["name"])
        except StudioError as e:
            return err(e, 404)
        media = {
            ".glb": "model/gltf-binary",
            ".obj": "model/obj",
            ".stl": "model/stl",
        }.get(p.suffix, "application/octet-stream")
        return FileResponse(
            str(p),
            media_type=media,
            headers={"Content-Disposition": f'attachment; filename="{p.name}"'},
        )

    async def mesh_tool(request: Request) -> JSONResponse:
        body = await request.json()
        try:
            r = await run_in_threadpool(
                studio.mesh_tool,
                request.path_params["job_id"],
                str(body.get("op", "")),
                body.get("params") or {},
            )
        except StudioError as e:
            return err(e)
        return JSONResponse(r)

    async def sample(request: Request) -> Response:
        try:
            p = studio.sample_image(request.path_params["name"])
        except StudioError as e:
            return err(e, 404)
        return FileResponse(str(p), media_type="image/png")

    async def report(request: Request) -> JSONResponse:
        return JSONResponse(
            await run_in_threadpool(studio.report_bundle, request.query_params.get("job"))
        )

    async def restart(_: Request) -> JSONResponse:
        return JSONResponse(studio.request_restart())

    async def update_state(request: Request) -> JSONResponse:
        force = request.query_params.get("force") in ("1", "true")
        return JSONResponse(await run_in_threadpool(studio.update_state, force))

    async def update_consent(request: Request) -> JSONResponse:
        body = await request.json()
        return JSONResponse(
            await run_in_threadpool(studio.set_update_check, bool(body.get("enabled")))
        )

    async def update_install(request: Request) -> JSONResponse:
        body = await request.json()
        if not body.get("consent"):
            return err(StudioError("consent required"))
        try:
            return JSONResponse(studio.start_update_install().to_dict())
        except StudioError as e:
            return err(e)

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
            Route("/brand/{name}.svg", brand),
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
            Route("/api/derived/{job_id}/{name}", derived),
            Route("/api/jobs/{job_id}/mesh", mesh_tool, methods=["POST"]),
            Route("/api/sample/{name}", sample),
            Route("/api/report", report),
            Route("/api/restart", restart, methods=["POST"]),
            Route("/api/update", update_state),
            Route("/api/update/consent", update_consent, methods=["POST"]),
            Route("/api/update/install", update_install, methods=["POST"]),
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
