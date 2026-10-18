"""Background removal with ONNX Runtime (MIT) and a permissively licensed segmentation model.

Two models are in the catalogue: IS-Net general use (Apache-2.0, 1024 px input, the better
one) and U²-Net (Apache-2.0, 320 px input, smaller and faster). The image is resized to the
model's input, the predicted matte is scaled back to the original size and used as the alpha
channel. Nothing leaves the machine. onnxruntime is an optional dependency (``pip install
hivemind-shapestudio``).
"""
# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from shapestudio.engine.base import EngineError, EngineResult, Progress, ProgressFn


@dataclass
class BackgroundSpec:
    input: Path
    output: Path
    model_path: Path
    model_name: str  # "isnet-general-use" | "u2net"
    feather: int = 2  # soften the matte edge a little; 0 = hard


def _ensure_runtime():
    try:
        import numpy as np
        import onnxruntime as ort
        from PIL import Image
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise EngineError(
            "background removal needs onnxruntime, numpy and Pillow: "
            "pip install 'hivemind-shapestudio'"
        ) from e
    return np, ort, Image


class BackgroundEngine:
    """Removes the background of one image. Stateless apart from a cached session per model."""

    name = "onnxruntime"

    def __init__(self) -> None:
        self._sessions: dict[str, object] = {}

    def available(self) -> bool:
        try:
            _ensure_runtime()
        except EngineError:
            return False
        return True

    def version(self) -> str | None:
        try:
            _np, ort, _img = _ensure_runtime()
        except EngineError:
            return None
        return f"onnxruntime {ort.__version__}"

    def _session(self, model_path: Path):
        key = str(model_path)
        if key not in self._sessions:
            _np, ort, _img = _ensure_runtime()
            self._sessions[key] = ort.InferenceSession(key, providers=["CPUExecutionProvider"])
        return self._sessions[key]

    def remove_background(
        self, spec: BackgroundSpec, on_progress: ProgressFn | None = None
    ) -> EngineResult:
        np, _ort, image_mod = _ensure_runtime()
        t0 = time.monotonic()
        if on_progress:
            on_progress(Progress("loading", message=f"loading {spec.model_name}"))
        sess = self._session(spec.model_path)
        inp = sess.get_inputs()[0]
        t1 = time.monotonic()
        with image_mod.open(spec.input) as src:
            im = src.convert("RGB")
        if spec.model_name.startswith("isnet"):
            size, mean, std = 1024, (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)
        else:
            size, mean, std = 320, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        small = im.resize((size, size), image_mod.BILINEAR)
        x = np.asarray(small, dtype=np.float32) / 255.0
        x = (x - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
        x = x.transpose(2, 0, 1)[None]
        if on_progress:
            on_progress(Progress("sampling", 1, 1, message="segmenting"))
        out = sess.run(None, {inp.name: x})[0]
        matte = out[0][0]
        matte = (matte - matte.min()) / (matte.max() - matte.min() + 1e-8)
        mask = image_mod.fromarray((matte * 255).astype(np.uint8)).resize(
            im.size, image_mod.BILINEAR
        )
        if spec.feather > 0:
            from PIL import ImageFilter

            mask = mask.filter(ImageFilter.GaussianBlur(spec.feather))
        rgba = im.copy()
        rgba.putalpha(mask)
        spec.output.parent.mkdir(parents=True, exist_ok=True)
        rgba.save(spec.output, "PNG")
        t2 = time.monotonic()
        if on_progress:
            on_progress(Progress("saving", 1, 1))
        return EngineResult(
            output=spec.output,
            load_seconds=t1 - t0,
            sampling_seconds=t2 - t1,
            total_seconds=t2 - t0,
            log_tail=[f"{spec.model_name} {size}px matte"],
        )
