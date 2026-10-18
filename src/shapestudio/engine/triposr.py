"""The TripoSR engine: loads the pinned checkpoint from the models folder (once, lazily),
removes the background with the same ONNX models as Image Studio, runs one network pass,
extracts the surface, writes GLB and OBJ with vertex colours and a CPU-rendered turntable.

Timing is reported in three parts (load, network, grid) so the calibration can learn each
on this machine. Cancellation is checked between chunks of the grid query, which is where a
CPU run spends most of its time."""
# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from shapestudio import runtime
from shapestudio.engine.base import EngineError, EngineResult, MeshSpec, Progress, ProgressFn
from shapestudio.engine.meshio import write_glb, write_obj
from shapestudio.engine.preview import write_previews
from shapestudio.engine.tsr.preprocess import composite_grey, has_alpha_cutout, resize_foreground

MAX_INPUT_SIDE = 4096


class TripoSREngine:
    name = "triposr"

    def __init__(self, background_engine=None) -> None:
        self._model = None
        self._model_path: Path | None = None
        self._device = "cpu"
        self._lock = threading.Lock()
        self._cancel = False
        self.background_engine = background_engine

    # -- protocol ----------------------------------------------------------------------------

    def available(self) -> bool:
        return runtime.torch_status().installed

    def version(self) -> str | None:
        st = runtime.torch_status()
        if not st.installed:
            return None
        return (
            f"torch {st.version} ({'cuda ' + str(st.cuda_version) if st.cuda_available else 'cpu'})"
        )

    def cancel(self) -> None:
        self._cancel = True

    def _check(self) -> None:
        if self._cancel:
            raise EngineError("cancelled")

    # -- model -------------------------------------------------------------------------------

    def _load(
        self, weight_path: Path, device: str, threads: int | None, on_progress: ProgressFn | None
    ):
        import torch

        from shapestudio.engine.tsr import TSR

        if threads:
            torch.set_num_threads(int(threads))
        if device == "cuda" and not torch.cuda.is_available():
            raise EngineError(
                "the plan asked for the GPU but this PyTorch cannot see a CUDA device; run "
                "`shapestudio doctor` and `shapestudio install-runtime --cuda`"
            )
        with self._lock:
            if self._model is None or self._model_path != weight_path or self._device != device:
                if on_progress:
                    on_progress(Progress("loading", 0.02, f"loading TripoSR weights ({device})"))
                model = TSR.from_checkpoint(weight_path)
                model.renderer.set_chunk_size(8192)
                model.to(device)
                if device == "cuda":
                    model.half()
                self._model, self._model_path, self._device = model, weight_path, device
            return self._model

    # -- input -------------------------------------------------------------------------------

    def _prepare(
        self, spec: MeshSpec, on_progress: ProgressFn | None
    ) -> tuple[np.ndarray, Path | None]:
        try:
            with Image.open(spec.input) as im:
                im.load()
                image = im.copy()
        except Exception as e:
            raise EngineError(f"cannot read the input image: {e}") from e
        if max(image.size) > MAX_INPUT_SIDE:
            image.thumbnail((MAX_INPUT_SIDE, MAX_INPUT_SIDE))
        cutout_path = None
        if spec.remove_background and not has_alpha_cutout(image):
            if self.background_engine is None or spec.bg_model_path is None:
                raise EngineError(
                    "background removal needs a segmentation model; download one with "
                    "`shapestudio download-model isnet-general-use --yes` or pass "
                    "remove_background=false with a picture that already has a transparent "
                    "background"
                )
            if on_progress:
                on_progress(
                    Progress(
                        "preprocessing", 0.08, f"removing the background ({spec.bg_model_name})"
                    )
                )
            from shapestudio.engine.background import BackgroundSpec

            tmp_in = spec.out_dir / f"{spec.stem}.input.png"
            image.convert("RGBA").save(tmp_in, "PNG")
            cutout_path = spec.out_dir / f"{spec.stem}.cutout.png"
            self.background_engine.remove_background(
                BackgroundSpec(tmp_in, cutout_path, spec.bg_model_path, spec.bg_model_name)
            )
            tmp_in.unlink(missing_ok=True)
            with Image.open(cutout_path) as cut:
                cut.load()
                image = cut.convert("RGBA")
        elif has_alpha_cutout(image):
            cutout_path = spec.out_dir / f"{spec.stem}.cutout.png"
            image.convert("RGBA").save(cutout_path, "PNG")
        self._check()
        if image.mode == "RGBA" and has_alpha_cutout(image):
            try:
                image = resize_foreground(image, spec.foreground_ratio)
            except ValueError as e:
                raise EngineError(str(e)) from e
            rgb = composite_grey(image)
        else:
            rgb = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
        return rgb, cutout_path

    # -- run ---------------------------------------------------------------------------------

    def reconstruct(self, spec: MeshSpec, on_progress: ProgressFn | None = None) -> EngineResult:
        self._cancel = False
        import psutil

        proc = psutil.Process(os.getpid())
        peak = proc.memory_info().rss
        t0 = time.monotonic()
        weight = spec.model_paths.get("model.ckpt")
        if weight is None or not Path(weight).exists():
            raise EngineError(
                "TripoSR weights are not installed (shapestudio download-model triposr --yes)"
            )
        spec.out_dir.mkdir(parents=True, exist_ok=True)
        model = self._load(Path(weight), spec.device, spec.threads, on_progress)
        t1 = time.monotonic()
        self._check()
        rgb, cutout = self._prepare(spec, on_progress)
        if on_progress:
            on_progress(Progress("encoding", 0.2, "network pass: image to triplane"))
        import torch

        with torch.no_grad():
            codes = model.scene_codes(rgb, spec.device)
            if spec.device == "cuda":
                codes = codes.half()
        peak = max(peak, proc.memory_info().rss)
        t2 = time.monotonic()
        self._check()

        def stage(name: str) -> None:
            if on_progress is None:
                return
            on_progress(
                {
                    "density": Progress(
                        "extracting", 0.35, f"querying the field on a {spec.resolution}³ grid"
                    ),
                    "marching": Progress("extracting", 0.7, "marching cubes"),
                    "colour": Progress("colouring", 0.8, "vertex colours from the field"),
                }[name]
            )

        try:
            v, f, c = model.extract_mesh(
                codes[0].float() if spec.device == "cpu" else codes[0],
                resolution=spec.resolution,
                check=self._check,
                on_stage=stage,
            )
        except ValueError as e:
            raise EngineError(str(e)) from e
        peak = max(peak, proc.memory_info().rss)
        t3 = time.monotonic()
        self._check()
        if on_progress:
            on_progress(Progress("exporting", 0.86, f"{len(v)} vertices, {len(f)} faces"))
        glb = write_glb(spec.out_dir / f"{spec.stem}.glb", v, f, c, name=spec.stem)
        obj = write_obj(spec.out_dir / f"{spec.stem}.obj", v, f, c)
        preview = None
        if spec.preview:
            if on_progress:
                on_progress(Progress("preview", 0.92, "rendering the turntable"))
            preview = spec.out_dir / f"{spec.stem}.preview.png"
            write_previews(
                v, f, c, preview, spec.out_dir / f"{spec.stem}.thumb.png", spec.preview_frames
            )
        if on_progress:
            on_progress(Progress("saving", 1.0))
        return EngineResult(
            glb=glb,
            obj=obj,
            preview=preview,
            cutout=cutout,
            vertices=len(v),
            faces=len(f),
            load_seconds=t1 - t0,
            network_seconds=t2 - t1,
            grid_seconds=t3 - t2,
            total_seconds=time.monotonic() - t0,
            log_tail=[f"triposr {spec.device} grid {spec.resolution}"],
            peak_rss_bytes=int(peak),
        )
