"""Per-machine timing calibration.

A reconstruction has three measurable parts: loading the weights, the network pass plus
surface extraction, and writing the files. All three are recorded per model, device and
grid resolution from real runs on this machine (exponential moving average) in
``calibration.json``. Until the first run completes the studio has nothing measured and
says so: the estimate is labelled ``not measured yet`` and comes from a stated heuristic.
No number here is a promise.
"""
# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from shapestudio.hardware import Hardware
from shapestudio.registry import MeshModel

# Seed heuristics, stated here so the "basis" text can name them. TripoSR's authors report
# about 0.5 s on an A100; nobody in this project has timed a CPU run yet, so the CPU seed is
# a deliberately rough "a minute or two on eight cores for the network pass, plus the grid".
_CPU_NETWORK_SECONDS_8_CORES = 60.0
_CPU_GRID_SECONDS_PER_MILLION_CELLS_8_CORES = 2.0  # NeRF MLP queries on the marching grid
_GPU_NETWORK_SECONDS = 3.0
_GPU_GRID_SECONDS_PER_MILLION_CELLS = 0.1
_DISK_BYTES_PER_SECOND = 200 * 1024 * 1024
_ALPHA = 0.5  # EMA weight of the newest observation


@dataclass
class Coefficients:
    load_seconds: float
    network_seconds: float
    grid_seconds_per_million_cells: float
    samples: int = 0

    @property
    def calibrated(self) -> bool:
        return self.samples > 0


@dataclass
class Estimate:
    seconds: float
    calibrated: bool
    basis: str

    def to_dict(self) -> dict:
        return asdict(self) | {"human": human_duration(self.seconds)}


def human_duration(seconds: float) -> str:
    s = round(seconds)
    if s < 60:
        return f"{s} s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m} min {s:02d} s"
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} min"


class Calibration:
    def __init__(self, path: Path, hardware: Hardware) -> None:
        self.path = path
        self.hardware = hardware
        self._data: dict[str, Coefficients] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self._data = {k: Coefficients(**v) for k, v in raw.items()}
            except (OSError, ValueError, TypeError):
                self._data = {}

    _save_lock = threading.Lock()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._save_lock:
            tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(
                json.dumps({k: asdict(v) for k, v in self._data.items()}, indent=2),
                encoding="utf-8",
            )
            for attempt in range(20):
                try:
                    tmp.replace(self.path)
                    return
                except PermissionError:
                    if attempt == 19:
                        tmp.unlink(missing_ok=True)
                        raise
                    time.sleep(0.05)

    @staticmethod
    def key(model: MeshModel, device: str) -> str:
        return f"{model.name}:{device}"

    def _seed(self, model: MeshModel, device: str) -> Coefficients:
        load = model.weights_bytes_in_memory / _DISK_BYTES_PER_SECOND
        if device == "cuda":
            return Coefficients(load, _GPU_NETWORK_SECONDS, _GPU_GRID_SECONDS_PER_MILLION_CELLS)
        cores = max(1, self.hardware.physical_cores)
        f = 8 / cores
        return Coefficients(
            load,
            _CPU_NETWORK_SECONDS_8_CORES * f,
            _CPU_GRID_SECONDS_PER_MILLION_CELLS_8_CORES * f,
        )

    def coefficients(self, model: MeshModel, device: str) -> Coefficients:
        return self._data.get(self.key(model, device)) or self._seed(model, device)

    def estimate(self, model: MeshModel, device: str, resolution: int) -> Estimate:
        c = self.coefficients(model, device)
        cells = resolution**3 / 1_000_000
        seconds = c.load_seconds + c.network_seconds + cells * c.grid_seconds_per_million_cells
        if c.calibrated:
            basis = (
                f"measured: {c.samples} run(s) of {model.display_name} on this machine's {device}"
            )
        else:
            basis = (
                "not measured yet: a heuristic from the core count (the first real run on this "
                "machine replaces it)"
                if device == "cpu"
                else "not measured yet: a rough GPU heuristic (the first real run replaces it)"
            )
        return Estimate(seconds=seconds, calibrated=c.calibrated, basis=basis)

    def record(
        self,
        model: MeshModel,
        device: str,
        resolution: int,
        load_seconds: float,
        network_seconds: float,
        grid_seconds: float,
    ) -> None:
        """Fold one finished job into the coefficients for its model and device."""
        cells = max(1e-6, resolution**3 / 1_000_000)
        per_m = grid_seconds / cells
        key = self.key(model, device)
        prev = self._data.get(key)
        if prev is None or prev.samples == 0:
            new = Coefficients(load_seconds, network_seconds, per_m, 1)
        else:
            a = _ALPHA
            new = Coefficients(
                load_seconds=prev.load_seconds * (1 - a) + load_seconds * a,
                network_seconds=prev.network_seconds * (1 - a) + network_seconds * a,
                grid_seconds_per_million_cells=prev.grid_seconds_per_million_cells * (1 - a)
                + per_m * a,
                samples=prev.samples + 1,
            )
        self._data[key] = new
        self._save()
