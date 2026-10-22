"""A fake mesh engine for tests and dry runs: writes a tiny valid GLB and OBJ (a coloured
cube), a cutout PNG and a turntable preview, reports every phase, needs no weights, no torch
and no network."""
# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import struct
import time
import zlib

from shapestudio.engine.base import (
    BackgroundResult,
    EngineError,
    EngineResult,
    MeshSpec,
    Progress,
    ProgressFn,
)
from shapestudio.engine.meshio import cube, write_glb, write_obj


def tiny_png(width: int = 8, height: int = 8, rgb: tuple[int, int, int] = (90, 120, 200)) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


class FakeMeshEngine:
    name = "fake"

    def __init__(self, phase_delay: float = 0.0, fail: bool = False) -> None:
        self.phase_delay = phase_delay
        self.fail = fail
        self.calls: list[MeshSpec] = []
        self._cancel = False

    def available(self) -> bool:
        return True

    def version(self) -> str | None:
        return "fake-0"

    def cancel(self) -> None:
        self._cancel = True

    def _step(self, on_progress: ProgressFn | None, phase: str, fraction: float, msg: str = ""):
        if self.phase_delay:
            time.sleep(self.phase_delay)
        if self._cancel:
            raise EngineError("cancelled")
        if on_progress:
            on_progress(Progress(phase, fraction, msg))

    def reconstruct(self, spec: MeshSpec, on_progress: ProgressFn | None = None) -> EngineResult:
        self.calls.append(spec)
        self._cancel = False
        t0 = time.monotonic()
        self._step(on_progress, "loading", 0.05, "fake weights")
        if self.fail:
            raise RuntimeError("fake engine failure")
        t1 = time.monotonic()
        spec.out_dir.mkdir(parents=True, exist_ok=True)
        cutout = None
        if spec.remove_background:
            self._step(on_progress, "preprocessing", 0.15, "fake cutout")
            cutout = spec.out_dir / f"{spec.stem}.cutout.png"
            cutout.write_bytes(tiny_png(16, 16, (10, 200, 10)))
        self._step(on_progress, "encoding", 0.4, "fake network pass")
        t2 = time.monotonic()
        self._step(on_progress, "extracting", 0.7, f"fake {spec.resolution}³ grid")
        self._step(on_progress, "colouring", 0.8)
        t3 = time.monotonic()
        self._step(on_progress, "exporting", 0.9)
        v, f, c = cube()
        glb = write_glb(spec.out_dir / f"{spec.stem}.glb", v, f, c)
        obj = write_obj(spec.out_dir / f"{spec.stem}.obj", v, f, c)
        preview = None
        if spec.preview:
            self._step(on_progress, "preview", 0.97)
            preview = spec.out_dir / f"{spec.stem}.preview.png"
            preview.write_bytes(tiny_png(24, 16, (200, 180, 90)))
        self._step(on_progress, "saving", 1.0)
        return EngineResult(
            glb=glb,
            obj=obj,
            preview=preview,
            cutout=cutout,
            vertices=int(v.shape[0]),
            faces=int(f.shape[0]),
            load_seconds=t1 - t0,
            network_seconds=t2 - t1,
            grid_seconds=t3 - t2,
            total_seconds=time.monotonic() - t0,
            log_tail=["fake engine"],
        )


class FakeBackgroundEngine:
    name = "fake-bg"

    def __init__(self) -> None:
        self.calls: list = []

    def available(self) -> bool:
        return True

    def version(self) -> str | None:
        return "fake-bg-0"

    def remove_background(self, spec, on_progress=None) -> BackgroundResult:
        self.calls.append(spec)
        spec.output.parent.mkdir(parents=True, exist_ok=True)
        spec.output.write_bytes(tiny_png(16, 16, (10, 200, 10)))
        return BackgroundResult(spec.output, 0.0, 0.0, 0.0, ["fake-bg"])
