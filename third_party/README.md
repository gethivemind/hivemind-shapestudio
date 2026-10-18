# Third-party code in this repository

| Where | Origin | Licence |
| --- | --- | --- |
| `src/shapestudio/engine/tsr/` (system, nerf, isosurface, preprocess) | [VAST-AI-Research/TripoSR](https://github.com/VAST-AI-Research/TripoSR), Copyright (c) 2024 Tripo AI & Stability AI | MIT, see `TripoSR-LICENSE` |
| `src/shapestudio/engine/tsr/transformer.py` | TripoSR's `tsr/models/transformer/*`, which are modified copies of [Hugging Face diffusers](https://github.com/huggingface/diffusers) attention blocks, Copyright 2023 The HuggingFace Team | Apache-2.0 (diffusers) plus MIT (TripoSR's modifications), see `HuggingFace-diffusers-NOTICE` |
| `src/shapestudio/engine/tsr/vit.py` | a reimplementation of the DINO ViT-B/16 architecture ([facebookresearch/dino](https://github.com/facebookresearch/dino), Apache-2.0) following the parameter layout of Hugging Face transformers' `ViTModel` (Apache-2.0) | Apache-2.0 (this file is original code) |

Everything else in the repository is Copyright 2026 Ratul Roy (royal-programmer) and the
Hivemind contributors, Apache-2.0. Model weights are downloaded by the user and keep their own
licences (see the Licences table in the README).
