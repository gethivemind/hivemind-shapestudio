"""TripoSR, vendored and trimmed for Shape Studio.

Upstream: https://github.com/VAST-AI-Research/TripoSR (MIT, Copyright (c) 2024 Tripo AI &
Stability AI); the transformer blocks there derive from Hugging Face diffusers (Apache-2.0).
Changes made here, all behaviour-preserving for inference:

* no ``omegaconf`` (the architecture of the published checkpoint is a plain dict),
* no ``einops`` (reshape/permute spelled out),
* no ``transformers`` and no ``huggingface_hub``: the DINO ViT-B/16 image tokenizer is
  reimplemented in ``vit.py`` with the same parameter names, so the published ``model.ckpt``
  loads unchanged,
* no ``torchmcubes`` (a C++/CUDA extension): marching cubes comes from scikit-image, so
  Windows needs no compiler,
* ``trimesh`` is not needed: the studio writes GLB/OBJ itself (``engine/meshio.py``).

The licence texts are in ``third_party/``.
"""

from shapestudio.engine.tsr.system import TRIPOSR_CONFIG, TSR

__all__ = ["TRIPOSR_CONFIG", "TSR"]
