from __future__ import annotations

import json
from pathlib import Path

import pytest

from shapestudio.config import Settings
from shapestudio.engine.fake import FakeBackgroundEngine, FakeMeshEngine
from shapestudio.hardware import Gpu, Hardware
from shapestudio.registry import TOOL_MODELS, TRELLIS, TRIPOSR, MeshModel, ToolModel
from shapestudio.service import Studio

GIB = 1024**3


def laptop(ram_gib: float = 16, cores: int = 10) -> Hardware:
    """The admin's machine: i5-1235U, 16 GB, Intel integrated graphics, no CUDA."""
    return Hardware(
        os="Windows",
        arch="AMD64",
        cpu_name="12th Gen Intel(R) Core(TM) i5-1235U",
        physical_cores=cores,
        logical_cores=cores + 2,
        ram_total_bytes=int(ram_gib * GIB),
        ram_available_bytes=int(ram_gib * GIB * 0.6),
        gpus=[Gpu("Intel(R) UHD Graphics", 2 * GIB, "integrated", shared_memory=True)],
    )


def workstation(vram_gib: float = 24) -> Hardware:
    return Hardware(
        os="Linux",
        arch="x86_64",
        cpu_name="big",
        physical_cores=16,
        logical_cores=32,
        ram_total_bytes=64 * GIB,
        ram_available_bytes=48 * GIB,
        gpus=[Gpu("NVIDIA RTX 4090", int(vram_gib * GIB), "cuda")],
    )


def small_gpu_box() -> Hardware:
    return Hardware(
        os="Windows",
        arch="AMD64",
        cpu_name="desktop",
        physical_cores=8,
        logical_cores=16,
        ram_total_bytes=32 * GIB,
        ram_available_bytes=20 * GIB,
        gpus=[Gpu("NVIDIA RTX 3060", 12 * GIB, "cuda")],
    )


def fake_install(settings: Settings, model: MeshModel | ToolModel) -> None:
    """Pretend a model is downloaded: stub files plus manifest entries. The size check is
    patched out by the autouse fixture below (the real files are gigabytes)."""
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = settings.manifest_file
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    files = model.files if isinstance(model, MeshModel) else (model.file,)
    for f in files:
        p = settings.models_dir / f.filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"stub")
        manifest[f.filename] = f.sha256
    manifest_path.write_text(json.dumps(manifest))


@pytest.fixture(autouse=True)
def _no_size_check(monkeypatch):
    monkeypatch.setattr("shapestudio.models._size_matches", lambda _p, _s: True)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings(data_dir=tmp_path / "data")
    s.ensure_dirs()
    return s


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    from PIL import Image, ImageDraw

    p = tmp_path / "teapot.png"
    img = Image.new("RGB", (96, 64), (250, 250, 250))
    ImageDraw.Draw(img).ellipse((20, 10, 76, 54), fill=(180, 90, 40))
    img.save(p, "PNG")
    return p


def make_studio(settings: Settings, hardware: Hardware, cuda_runtime: bool = False, **kw) -> Studio:
    from shapestudio.engine.trellis import TrellisEngine

    # TripoSR is faked; TRELLIS is the real stub, which refuses honestly without a GPU
    engines = {"triposr": FakeMeshEngine(**kw), "trellis": TrellisEngine()}
    return Studio(
        settings=settings,
        engines=engines,
        hardware=hardware,
        background_engine=FakeBackgroundEngine(),
        cuda_runtime=cuda_runtime,
    )


@pytest.fixture
def studio(settings: Settings):
    fake_install(settings, TRIPOSR)
    fake_install(settings, TOOL_MODELS["isnet-general-use"])
    st = make_studio(settings, laptop())
    yield st
    st.close()


__all__ = [
    "TRELLIS",
    "TRIPOSR",
    "fake_install",
    "laptop",
    "make_studio",
    "small_gpu_box",
    "workstation",
]
