"""The engine interface. TripoSR is the first implementation; TRELLIS sits behind the same
protocol as a stub until it is verified on a GPU machine."""
# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

PHASES = (
    "loading",  # weights from disk
    "preprocessing",  # background removal, crop, resize
    "encoding",  # image tokens -> triplane (the network pass)
    "extracting",  # density grid + marching cubes
    "colouring",  # vertex colours from the field
    "exporting",  # GLB / OBJ
    "preview",  # turntable PNG
    "saving",
)


@dataclass
class MeshSpec:
    """Everything an engine needs to reconstruct one object. Paths are already resolved."""

    input: Path
    out_dir: Path
    stem: str  # output files are <stem>.glb, <stem>.obj, <stem>.preview.png, <stem>.cutout.png
    model_paths: dict[str, Path]  # basename -> weight file
    device: str = "cpu"
    resolution: int = 256
    foreground_ratio: float = 0.85
    remove_background: bool = True
    bg_model_path: Path | None = None
    bg_model_name: str = ""
    threads: int | None = None
    chunk_size: int = 8192
    preview: bool = True
    preview_frames: int = 6


@dataclass
class Progress:
    phase: str
    fraction: float = 0.0  # 0-1 within the whole job, monotonic
    message: str = ""


@dataclass
class EngineResult:
    glb: Path
    obj: Path
    preview: Path | None
    cutout: Path | None
    vertices: int
    faces: int
    load_seconds: float
    network_seconds: float
    grid_seconds: float
    total_seconds: float
    log_tail: list[str] = field(default_factory=list)
    peak_rss_bytes: int = 0


@dataclass
class BackgroundResult:
    """What the background remover produced: the RGBA cutout and its timings."""

    output: Path
    load_seconds: float
    matte_seconds: float
    total_seconds: float
    log_tail: list[str] = field(default_factory=list)


ProgressFn = Callable[[Progress], None]


class EngineError(RuntimeError):
    pass


class MeshEngine(Protocol):
    name: str

    def available(self) -> bool: ...

    def version(self) -> str | None: ...

    def reconstruct(
        self, spec: MeshSpec, on_progress: ProgressFn | None = None
    ) -> EngineResult: ...

    def cancel(self) -> None: ...
