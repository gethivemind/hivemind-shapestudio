"""TRELLIS: the ``best`` tier. Catalogue entry, fit check and planner tier are real; the
engine itself is a labelled stub that refuses with the exact reason until it has been
verified on a GPU machine. It never pretends and never falls back to another model."""
# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from shapestudio import runtime
from shapestudio.engine.base import EngineError, EngineResult, MeshSpec, ProgressFn
from shapestudio.registry import TRELLIS

STATUS = (
    "not integrated yet: TRELLIS needs a CUDA-only runtime stack (spconv, xformers or "
    "flash-attention, nvdiffrast) and the DINOv2 ViT-L/14 encoder that this build does not "
    "package. The weights can be downloaded and hash-verified now; the engine will be enabled "
    "once it has been run on a machine with an NVIDIA GPU of 16 GB or more."
)


class TrellisEngine:
    name = "trellis"

    def available(self) -> bool:
        return False

    def version(self) -> str | None:
        return None

    def cancel(self) -> None:
        return None

    def status(self) -> dict:
        st = runtime.torch_status()
        return {
            "integrated": False,
            "reason": STATUS,
            "needs": list(TRELLIS.extra_runtime),
            "cuda_available": st.cuda_available,
            "device": st.device_name,
        }

    def reconstruct(self, spec: MeshSpec, on_progress: ProgressFn | None = None) -> EngineResult:
        st = runtime.torch_status()
        if not st.cuda_available:
            raise EngineError(
                f"TRELLIS needs an NVIDIA GPU with at least 16 GB; this PyTorch sees no CUDA "
                f"device ({st.version or 'torch not installed'}). {STATUS}"
            )
        raise EngineError(f"TRELLIS {STATUS}")
