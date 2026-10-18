"""Hardware detection: CPU, RAM, GPU. Best effort, never raises, no network."""
# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import psutil

GIB = 1024**3
PROBE_TIMEOUT = 25.0  # PowerShell/CIM probes can take many seconds on a loaded machine
CACHE_MAX_AGE = 24 * 3600


@dataclass
class Gpu:
    name: str
    vram_bytes: int | None  # None when unknown; integrated GPUs report shared memory
    kind: str  # "cuda" | "rocm" | "integrated" | "discrete" | "unknown"
    shared_memory: bool = False


@dataclass
class Hardware:
    os: str
    arch: str
    cpu_name: str
    physical_cores: int
    logical_cores: int
    ram_total_bytes: int
    ram_available_bytes: int
    gpus: list[Gpu] = field(default_factory=list)

    @property
    def has_cuda(self) -> bool:
        return any(g.kind == "cuda" for g in self.gpus)

    @property
    def usable_gpu(self) -> Gpu | None:
        """A GPU worth running the diffusion model on: discrete with known VRAM."""
        for g in self.gpus:
            if g.kind in ("cuda", "rocm", "discrete") and (g.vram_bytes or 0) >= 4 * GIB:
                return g
        return None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ram_total_gib"] = round(self.ram_total_bytes / GIB, 1)
        d["ram_available_gib"] = round(self.ram_available_bytes / GIB, 1)
        for g in d["gpus"]:
            g["vram_gib"] = round(g["vram_bytes"] / GIB, 1) if g["vram_bytes"] else None
        return d


def _run(cmd: list[str], timeout: float = PROBE_TIMEOUT) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _cpu_name() -> str:
    name = platform.processor() or ""
    if sys.platform == "linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif sys.platform == "darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
        if out:
            return out
    elif sys.platform == "win32":
        out = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name",
            ]
        ).strip()
        if out:
            return out
    return name or "unknown"


def _nvidia_gpus() -> list[Gpu]:
    if not shutil.which("nvidia-smi"):
        return []
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    gpus: list[Gpu] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[1].isdigit():
            gpus.append(Gpu(name=parts[0], vram_bytes=int(parts[1]) * 1024 * 1024, kind="cuda"))
    return gpus


def _windows_gpus() -> list[Gpu]:
    out = _run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | "
            "ForEach-Object { $_.Name + '|' + $_.AdapterRAM }",
        ]
    )
    gpus: list[Gpu] = []
    for line in out.splitlines():
        if "|" not in line:
            continue
        name, _, ram = line.strip().rpartition("|")
        vram = int(ram) if ram.isdigit() else None
        lname = name.lower()
        integrated = any(k in lname for k in ("intel", "uhd", "iris", "radeon graphics", "vega"))
        kind = "integrated" if integrated else "discrete"
        gpus.append(Gpu(name=name, vram_bytes=vram, kind=kind, shared_memory=integrated))
    return gpus


def _linux_gpus() -> list[Gpu]:
    out = _run(["lspci"])
    gpus: list[Gpu] = []
    for line in out.splitlines():
        if "VGA" in line or "3D controller" in line or "Display controller" in line:
            name = line.split(":", 2)[-1].strip()
            lname = name.lower()
            integrated = "intel" in lname
            kind = "integrated" if integrated else "unknown"
            gpus.append(Gpu(name=name, vram_bytes=None, kind=kind, shared_memory=integrated))
    return gpus


def _mac_gpus() -> list[Gpu]:
    out = _run(["system_profiler", "SPDisplaysDataType"], timeout=20)
    gpus: list[Gpu] = []
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Chipset Model:"):
            name = s.split(":", 1)[1].strip()
            apple = "apple" in name.lower()
            gpus.append(
                Gpu(
                    name=name,
                    vram_bytes=None,
                    kind="integrated" if apple else "discrete",
                    shared_memory=apple,
                )
            )
    return gpus


def detect(cache_file: Path | None = None) -> Hardware:
    """Detect hardware. Static facts (CPU name, cores, GPUs, total RAM) are cached in
    ``cache_file`` for a day because the probes are slow; available RAM is always live."""
    cached = _read_cache(cache_file)
    vm = psutil.virtual_memory()
    if cached is not None:
        cached.ram_available_bytes = vm.available
        return cached
    hw = _probe()
    _write_cache(cache_file, hw)
    return hw


def _read_cache(cache_file: Path | None) -> Hardware | None:
    if cache_file is None or not cache_file.exists():
        return None
    try:
        if time.time() - cache_file.stat().st_mtime > CACHE_MAX_AGE:
            return None
        d = json.loads(cache_file.read_text(encoding="utf-8"))
        gpus = [Gpu(**g) for g in d.pop("gpus", [])]
        d = {k: d[k] for k in Hardware.__dataclass_fields__ if k in d and k != "gpus"}
        return Hardware(gpus=gpus, **d)
    except (OSError, ValueError, TypeError):
        return None


def _write_cache(cache_file: Path | None, hw: Hardware) -> None:
    if cache_file is None:
        return
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(asdict(hw), indent=2), encoding="utf-8")
    except OSError:
        pass


def _probe() -> Hardware:
    gpus = _nvidia_gpus()
    if sys.platform == "win32":
        seen = {g.name for g in gpus}
        gpus += [g for g in _windows_gpus() if g.name not in seen]
    elif sys.platform == "linux":
        if not gpus:
            gpus = _linux_gpus()
    elif sys.platform == "darwin":
        gpus = _mac_gpus()
    vm = psutil.virtual_memory()
    return Hardware(
        os=platform.system(),
        arch=platform.machine(),
        cpu_name=_cpu_name(),
        physical_cores=psutil.cpu_count(logical=False) or os.cpu_count() or 1,
        logical_cores=psutil.cpu_count(logical=True) or os.cpu_count() or 1,
        ram_total_bytes=vm.total,
        ram_available_bytes=vm.available,
        gpus=gpus,
    )
