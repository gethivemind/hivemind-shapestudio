"""The PyTorch runtime: what is importable, whether it sees a GPU, and the consented install
of the CUDA build.

Two situations:

* **pip / pipx install**: ``torch`` is a declared dependency, so the CPU build (or on Linux the
  default PyPI build, which carries CUDA libraries) is already present. ``install-runtime
  --cuda`` runs pip in the same interpreter against the pinned CUDA index.
* **packaged app** (PyInstaller): the CPU build is inside the app. ``install-runtime --cuda``
  downloads the pinned, hash-verified CUDA wheel for this platform (Windows only: the Linux
  CUDA wheel depends on separate nvidia-* wheels and is pip-only), unpacks it under
  ``<data>/runtime/cuda`` and a meta-path finder installed at start-up serves ``torch`` from
  there instead of the bundled copy. This path is written but **not yet verified on a GPU
  machine**; ``doctor`` says which torch actually loaded.
"""
# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.machinery
import json
import subprocess
import sys
import zipfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from shapestudio.config import Settings
from shapestudio.registry import (
    GIB,
    TORCH_CUDA_INDEX,
    TORCH_CUDA_LABEL,
    TORCH_VERSION,
    RuntimeWheel,
    find_runtime_wheel,
)

ConsentFn = Callable[[str, int], bool]
ProgressFn = Callable[[int, int], None]


@dataclass
class TorchStatus:
    installed: bool
    version: str | None = None
    cuda_available: bool = False
    cuda_version: str | None = None
    device_name: str | None = None
    device_vram_gib: float | None = None
    flavor: str = "none"  # cpu | cuda | none
    location: str | None = None
    error: str | None = None
    from_runtime_dir: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


_status: TorchStatus | None = None


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def torch_status(refresh: bool = False) -> TorchStatus:
    """Import torch once (slow: a second or two) and remember what it is."""
    global _status
    if _status is not None and not refresh:
        return _status
    try:
        import torch
    except Exception as e:  # ImportError, OSError for a broken runtime dir
        _status = TorchStatus(installed=False, error=f"{type(e).__name__}: {e}")
        return _status
    cuda = False
    cuda_version = None
    name = None
    vram = None
    try:
        cuda = bool(torch.cuda.is_available())
        cuda_version = torch.version.cuda
        if cuda:
            name = torch.cuda.get_device_name(0)
            vram = round(torch.cuda.get_device_properties(0).total_memory / GIB, 1)
    except Exception as e:  # a CUDA build without a driver, for example
        cuda = False
        name = f"cuda probe failed: {e}"
    location = str(Path(torch.__file__).parent)
    _status = TorchStatus(
        installed=True,
        version=torch.__version__,
        cuda_available=cuda,
        cuda_version=cuda_version,
        device_name=name,
        device_vram_gib=vram,
        flavor="cuda" if cuda else "cpu",
        location=location,
        from_runtime_dir=_RUNTIME_ROOT is not None and location.startswith(str(_RUNTIME_ROOT)),
    )
    return _status


# --- the unpacked CUDA wheel in the data dir -------------------------------------------------

_RUNTIME_ROOT: Path | None = None


class _RuntimeFinder(importlib.machinery.PathFinder):
    """Serves ``torch`` and its submodules from the unpacked wheel, ahead of PyInstaller's
    frozen importer (which would otherwise win by name)."""

    def __init__(self, root: Path) -> None:
        self.root = str(root)

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "torch" and not fullname.startswith("torch."):
            return None
        search = [self.root] if fullname == "torch" else path
        return importlib.machinery.PathFinder.find_spec(fullname, search, target)


def runtime_info(settings: Settings) -> dict | None:
    p = settings.runtime_dir / "runtime.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def activate_runtime(settings: Settings) -> Path | None:
    """Called before anything imports torch. Returns the runtime root in use, if any."""
    global _RUNTIME_ROOT
    info = runtime_info(settings)
    if not info or info.get("flavor") != "cuda":
        return None
    if settings.config().get("runtime", "cuda") != "cuda":
        return None  # the user switched back to the bundled CPU build
    root = settings.runtime_dir / "cuda" / "site-packages"
    if not (root / "torch" / "__init__.py").exists():
        return None
    _RUNTIME_ROOT = root
    sys.meta_path.insert(0, _RuntimeFinder(root))
    return root


# --- install --------------------------------------------------------------------------------


def cuda_wheel_for_this_machine() -> RuntimeWheel | None:
    import platform

    return find_runtime_wheel("cuda", sys.platform, platform.machine())


def cuda_install_plan(settings: Settings) -> dict:
    """What ``install-runtime --cuda`` would do here, before any consent: the method, the
    size in words, and whether it is possible at all."""
    wheel = cuda_wheel_for_this_machine()
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if is_frozen():
        if wheel is None or not wheel.self_contained:
            return {
                "possible": False,
                "method": "none",
                "reason": "no self-contained CUDA wheel is pinned for this platform; install "
                "with pipx instead (pipx install presidium-shapestudio, then shapestudio "
                "install-runtime --cuda).",
            }
        if wheel.python != py:
            return {
                "possible": False,
                "method": "none",
                "reason": f"the packaged app runs {py} but the pinned wheel is {wheel.python}",
            }
        return {
            "possible": True,
            "method": "wheel",
            "wheel": wheel.file.filename,
            "url": wheel.file.url,
            "sha256": wheel.file.sha256,
            "bytes": wheel.file.size,
            "gib": round(wheel.file.size / GIB, 2),
            "target": str(settings.runtime_dir / "cuda"),
            "note": "downloaded once, hash-verified, unpacked under the data folder; the app "
            "loads it instead of its bundled CPU build. Not yet verified on a GPU machine.",
        }
    return {
        "possible": True,
        "method": "pip",
        "command": [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            f"torch=={TORCH_VERSION}",
            "--index-url",
            TORCH_CUDA_INDEX,
        ],
        "bytes": (wheel.file.size if wheel else 0),
        "gib": round((wheel.file.size if wheel else 0) / GIB, 2),
        "note": (
            f"pip resolves the {TORCH_CUDA_LABEL} build of torch {TORCH_VERSION} for this "
            "interpreter. On Linux the wheel is about 0.5 GiB and pulls roughly 2.5 GiB of "
            "nvidia-* library wheels; on Windows the single wheel is about 1.9 GiB."
        ),
    }


def install_cuda_runtime(
    settings: Settings, consent: ConsentFn, progress: ProgressFn | None = None
) -> dict:
    plan = cuda_install_plan(settings)
    if not plan["possible"]:
        raise RuntimeError(plan["reason"])
    if plan["method"] == "pip":
        desc = f"PyTorch {TORCH_VERSION} CUDA build ({TORCH_CUDA_LABEL}) via pip: {plan['note']}"
        if not consent(desc, plan["bytes"]):
            return {"installed": False, "message": "not approved"}
        try:
            r = subprocess.run(plan["command"], check=False)
        except OSError as e:
            raise RuntimeError(f"could not run pip: {e}") from e
        if r.returncode != 0:
            raise RuntimeError(
                "pip failed. If this is a pipx install, run: pipx inject presidium-shapestudio "
                f"'torch=={TORCH_VERSION}' --pip-args='--index-url {TORCH_CUDA_INDEX}' "
                "(or --force to replace the CPU build)."
            )
        torch_status(refresh=True)
        return {"installed": True, "method": "pip", "status": torch_status().to_dict()}
    # frozen: download the wheel and unpack it
    from shapestudio import download as dl

    wheel = cuda_wheel_for_this_machine()
    assert wheel is not None
    desc = f"{wheel.file.filename} ({wheel.file.licence}) from {wheel.file.source}: {plan['note']}"
    if not consent(desc, wheel.file.size):
        return {"installed": False, "message": "not approved"}
    target_dir = settings.runtime_dir / "cuda"
    whl = settings.runtime_dir / wheel.file.filename
    dl.download(wheel.file.url, whl, wheel.file.sha256, wheel.file.size, progress=progress)
    site = target_dir / "site-packages"
    site.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(whl) as z:
        z.extractall(site)
    whl.unlink(missing_ok=True)
    (settings.runtime_dir / "runtime.json").write_text(
        json.dumps(
            {
                "flavor": "cuda",
                "wheel": wheel.file.filename,
                "torch": TORCH_VERSION,
                "label": TORCH_CUDA_LABEL,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    cfg = settings.config()
    cfg["runtime"] = "cuda"
    settings.write_config(cfg)
    return {
        "installed": True,
        "method": "wheel",
        "restart_required": True,
        "message": "the CUDA runtime is unpacked; restart the studio to load it",
    }


def remove_cuda_runtime(settings: Settings) -> int:
    """Delete the unpacked wheel. Returns bytes freed."""
    import shutil

    root = settings.runtime_dir / "cuda"
    freed = sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0
    shutil.rmtree(root, ignore_errors=True)
    (settings.runtime_dir / "runtime.json").unlink(missing_ok=True)
    cfg = settings.config()
    cfg.pop("runtime", None)
    settings.write_config(cfg)
    return freed


def report(settings: Settings) -> dict:
    """Everything ``doctor`` and ``capabilities()`` say about the runtime."""
    st = torch_status()
    plan = cuda_install_plan(settings)
    return {
        "torch": st.to_dict(),
        "pinned_version": TORCH_VERSION,
        "cuda_label": TORCH_CUDA_LABEL,
        "frozen_app": is_frozen(),
        "runtime_dir": str(settings.runtime_dir),
        "runtime_installed": runtime_info(settings),
        "cuda_install": plan,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    }
