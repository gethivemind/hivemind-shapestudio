"""The MCP server. Tool descriptions are written for an AI caller: what it does, when to use
it, what comes back."""
# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from typing import Any

from shapestudio.registry import MODELS, PRESET_NAMES, TIERS, TOOL_MODELS
from shapestudio.service import Studio, StudioError

try:  # mcp >= 2: FastMCP was renamed MCPServer
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:  # pragma: no cover - mcp 1.x
    from mcp.server.fastmcp import FastMCP  # type: ignore[no-redef]
from mcp.types import ToolAnnotations


def _hints(
    *, read_only: bool = False, destructive: bool = False, idempotent: bool = False
) -> ToolAnnotations:
    return ToolAnnotations(
        readOnlyHint=read_only,
        destructiveHint=destructive,
        idempotentHint=idempotent,
        openWorldHint=False,
    )


INSTRUCTIONS = """Hivemind Shape Studio: free, local, offline image-to-3D.

Policy: quality is a setting, never a consequence. reconstruct() defaults to preset "best"
(the finest surface grid) on the strongest permissively licensed model that fits this
machine: TripoSR (MIT, runs on a CPU) or, on a 16 GB NVIDIA GPU, TRELLIS (MIT; its engine is
not integrated yet and says so). The machine decides how long a mesh takes, never how good it
is; on a CPU expect a minute or more per object and an estimate labelled "not measured yet"
until the first run. Faster presets ("balanced", "fast") exist only when the caller chooses
them, with the trade-off stated.

Workflow: call capabilities() once to learn hardware, installed models and expected times.
Call reconstruct(image) with a PNG/JPEG/WebP path: it returns a job id and an ETA at once.
Poll job_status() every 10-15 seconds until status is "done", then call job_result() for the
GLB path (and OBJ path, turntable PNG). Never block waiting inside a single tool call.
"""


def build_server(studio: Studio | None = None) -> FastMCP:
    studio = studio or Studio()
    mcp = FastMCP(name="hivemind-shapestudio", instructions=INSTRUCTIONS)

    def _err(e: Exception) -> dict[str, Any]:
        return {"error": str(e)}

    @mcp.tool(
        name="capabilities",
        annotations=_hints(read_only=True, idempotent=True),
        description=(
            "Describe this studio: the hardware found (CPU cores, RAM, GPU and VRAM if any), the "
            "PyTorch runtime (CPU or CUDA, and whether the CUDA runtime can be installed here), "
            "every catalogue model with its licence, tier, download size, whether it is "
            "downloaded, whether it fits this machine and why not, each preset with its "
            "trade-off in words and the expected time on this machine (marked 'not measured "
            "yet' until a run has happened), the background-removal models, the quality policy, "
            "the licence list and the weights excluded by policy. Call it once before the first "
            "reconstruct() to set expectations. No parameters."
        ),
    )
    def capabilities() -> dict[str, Any]:
        return studio.capabilities()

    @mcp.tool(
        name="plan",
        annotations=_hints(read_only=True, idempotent=True),
        description=(
            "Dry run of reconstruct(): same parameters, nothing is queued. Returns the resolved "
            "plan {model, tier, device cpu|cuda, preset, resolution, estimate {seconds, human, "
            "calibrated, basis}, load {score percent of memory budget, label, required_gib, "
            "budget_gib}, fit_note, installed, warnings, over_budget}. Use it to explain to the "
            "human what will run and how long it is expected to take, or why a tier is refused."
        ),
    )
    def plan(
        tier: str | None = None, preset: str = "best", resolution: int | None = None
    ) -> dict[str, Any]:
        try:
            return studio.plan(tier, preset, resolution).to_dict()
        except StudioError as e:
            return _err(e)

    @mcp.tool(
        name="reconstruct",
        annotations=_hints(),
        description=(
            "Queue an image-to-3D job and return its job id immediately. image is a path to a "
            "PNG, JPEG or WebP on this machine (one object, preferably centred; the background is "
            "removed automatically with IS-Net unless remove_background=false and the picture "
            "already has a transparent background). tier: omit for the strongest tier that is "
            "downloaded and fits ("
            + ", ".join(f"'{t}'" for t in TIERS)
            + "), or a model name ("
            + ", ".join(f"'{m}'" for m in MODELS)
            + "). preset: "
            + ", ".join(f"'{p}'" for p in PRESET_NAMES)
            + " ('best' = 256³ surface grid, 'balanced' = 192³, 'fast' = 128³; the network pass "
            "is identical, only the extracted surface detail and the extraction time change). "
            "resolution (64-512, multiple of 32) overrides the preset's grid. foreground_ratio "
            "0.5-1.0 (default 0.85) is how much of the frame the object fills before the network "
            "sees it. Returns {job_id, status, estimate_seconds, estimate_human, estimate_basis, "
            "model, tier, device, preset, resolution, fit_note, warnings}. Then poll "
            "job_status(job_id) every 10-15 seconds and fetch job_result(job_id) when status is "
            "'done'. Errors (model not downloaded, PyTorch missing, GPU-only tier on a machine "
            "without the GPU, bad preset) come back as {error} with the fix spelled out."
        ),
    )
    def reconstruct(
        image: str,
        tier: str | None = None,
        preset: str = "best",
        resolution: int | None = None,
        remove_background: bool = True,
        foreground_ratio: float | None = None,
    ) -> dict[str, Any]:
        try:
            job = studio.reconstruct(
                image,
                model=tier,
                preset=preset,
                resolution=resolution,
                foreground_ratio=foreground_ratio,
                remove_background=remove_background,
            )
        except StudioError as e:
            return _err(e)
        st = studio.job_status(job.id)
        return {
            "job_id": job.id,
            "status": job.status,
            "estimate_seconds": round(job.estimate_seconds),
            "estimate_human": st["eta_human"],
            "estimate_basis": job.estimate_basis,
            "queue_position": st["queue_position"],
            "model": job.model,
            "tier": job.tier,
            "device": job.device,
            "preset": job.preset,
            "resolution": job.resolution,
            "fit_note": job.fit_note,
            "over_budget": job.over_budget,
            "warnings": job.warnings,
        }

    @mcp.tool(
        name="job_status",
        annotations=_hints(read_only=True, idempotent=True),
        description=(
            "Report progress of a job started by reconstruct(). Poll every 10-15 seconds, not "
            "continuously. Returns {id, status (queued|running|done|failed|cancelled), phase "
            "(loading|preprocessing|encoding|extracting|colouring|exporting|preview|saving), "
            "progress 0-1, summary (one honest line), eta_seconds, eta_human (null once the "
            "estimate is spent: the studio says 'measuring' rather than inventing a number), "
            "estimate_basis, queue_position, message, error, and when done glb_path, obj_path, "
            "preview_path, vertices, faces, total_seconds}."
        ),
    )
    def job_status(job_id: str) -> dict[str, Any]:
        try:
            return studio.job_status(job_id)
        except StudioError as e:
            return _err(e)

    @mcp.tool(
        name="job_result",
        annotations=_hints(read_only=True, idempotent=True),
        description=(
            "Fetch the finished mesh of a job. Call it when job_status reports 'done'. Returns "
            "{ready: true, format, path (the GLB on this machine's disk, kept under the studio's "
            "output folder), mime_type, glb_path, obj_path, preview_path (a PNG turntable sheet "
            "rendered on the CPU), cutout_path, vertices, faces, bytes, model, tier, device, "
            "preset, resolution, total_seconds} and, when as_base64 is true, 'base64' with the "
            "file bytes (large; use it only when the caller cannot read local files). format "
            "'glb' (default, vertex colours, for viewers and game engines) or 'obj' (plain text "
            "with v x y z r g b lines). If the job is not finished it returns {ready: false, "
            "status, status_detail} instead of blocking."
        ),
    )
    def job_result(job_id: str, as_base64: bool = False, format: str = "glb") -> dict[str, Any]:
        try:
            return studio.job_result(job_id, as_base64, format)
        except StudioError as e:
            return _err(e)

    @mcp.tool(
        name="jobs",
        annotations=_hints(read_only=True, idempotent=True),
        description=(
            "List all jobs known to the studio, newest first, with id, status, title, source "
            "image name, model, tier, device, preset, resolution, vertex and face counts and the "
            "GLB path when done. Use it to recover a job id or to see what is queued or running."
        ),
    )
    def jobs() -> dict[str, Any]:
        return {"jobs": studio.list_jobs()}

    @mcp.tool(
        name="cancel_job",
        annotations=_hints(destructive=True, idempotent=True),
        description=(
            "Cancel a queued or running job. Returns {cancelled: bool}. A running job stops at "
            "the next chunk of work (seconds); its partial work is discarded."
        ),
    )
    def cancel_job(job_id: str) -> dict[str, Any]:
        return {"cancelled": studio.cancel(job_id)}

    @mcp.tool(
        name="download_model",
        annotations=ToolAnnotations(
            readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=True
        ),
        description=(
            "Download a catalogue model's weights into the studio's models folder so "
            "reconstruct() can use it offline afterwards. This is the only tool that uses the "
            "network and it needs explicit consent: call it first with consent=false to get the "
            "exact files, total size in GiB, SHA-256 pins and licence, then again with "
            "consent=true after the human agrees. name is one of "
            + ", ".join(f"'{m}'" for m in MODELS)
            + " or a background-removal model: "
            + ", ".join(f"'{t}'" for t in TOOL_MODELS)
            + ". TripoSR is 1.56 GiB (MIT); TRELLIS is 3.1 GiB (MIT) and its engine is not "
            "integrated yet (download_model still works so the files are ready). Downloads are "
            "resumable and SHA-256 verified. The call blocks until the download finishes; for "
            "slow networks prefer the CLI `shapestudio download-model <name> --yes`."
        ),
    )
    def download_model(name: str, consent: bool = False) -> dict[str, Any]:
        try:
            info = studio.download_info(name)
        except StudioError as e:
            return _err(e)
        if info["already_installed"]:
            return info | {"downloaded": False, "message": "already installed"}
        if not consent:
            return info | {
                "downloaded": False,
                "message": "Call again with consent=true after the human approves this download.",
            }
        try:
            ok = studio.install(name, lambda _d, _b: True)
        except Exception as e:  # surfaced to the caller
            return info | {"downloaded": False, "error": str(e)}
        return info | {"downloaded": ok}

    return mcp


def serve(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8775) -> None:
    server = build_server()
    if transport == "stdio":
        server.run("stdio")
    else:
        server.settings.host = host  # type: ignore[attr-defined]
        server.settings.port = port  # type: ignore[attr-defined]
        server.run("streamable-http")


if __name__ == "__main__":  # pragma: no cover
    serve(sys.argv[1] if len(sys.argv) > 1 else "stdio")
