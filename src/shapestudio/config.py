"""Paths and settings. Everything the studio writes lives under one data directory.

Identifiers here are brand-neutral on purpose (``shapestudio``, ``SHAPESTUDIO_*``): a later
rename of the product touches display strings and artwork, not folders on users' disks."""
# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

APP_NAME = "shapestudio"  # the data folder name; brand-neutral
ENV_DATA_DIR = "SHAPESTUDIO_DATA_DIR"
ENV_THREADS = "SHAPESTUDIO_THREADS"
ENV_ALLOW_OVER_BUDGET = "SHAPESTUDIO_ALLOW_OVER_BUDGET"
ENV_DEVICE = "SHAPESTUDIO_DEVICE"  # cpu | cuda; overrides the planner's device choice


def default_data_dir() -> Path:
    env = os.environ.get(ENV_DATA_DIR)
    if env:
        return Path(env).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
        return Path(base) / APP_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    base = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    return Path(base) / APP_NAME


@dataclass(frozen=True)
class Settings:
    """Resolved locations. Create with ``Settings.load()`` or pass a ``data_dir`` in tests."""

    data_dir: Path = field(default_factory=default_data_dir)

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def runtime_dir(self) -> Path:
        """Optional GPU runtime (a PyTorch CUDA wheel unpacked by ``install-runtime``)."""
        return self.data_dir / "runtime"

    @property
    def out_dir(self) -> Path:
        return self.data_dir / "out"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def trash_dir(self) -> Path:
        return self.data_dir / "trash"

    @property
    def calibration_file(self) -> Path:
        return self.data_dir / "calibration.json"

    @property
    def config_file(self) -> Path:
        return self.data_dir / "config.json"

    @property
    def hardware_cache_file(self) -> Path:
        return self.data_dir / "hardware.json"

    @property
    def worker_lock_file(self) -> Path:
        return self.data_dir / "worker.lock"

    @property
    def profile_file(self) -> Path:
        return self.data_dir / "profile.json"

    @property
    def manifest_file(self) -> Path:
        return self.models_dir / "manifest.json"

    @property
    def projects_file(self) -> Path:
        return self.data_dir / "projects.json"

    def config(self) -> dict:
        """User settings from config.json in the data dir (missing file = {})."""
        try:
            return json.loads(self.config_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def write_config(self, cfg: dict) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.config_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        tmp.replace(self.config_file)

    @property
    def allow_over_budget(self) -> bool:
        """The unlock: run settings the memory budget says will not fit. Off by default;
        enabled only by the environment variable or an explicit config.json entry."""
        env = os.environ.get(ENV_ALLOW_OVER_BUDGET, "").lower()
        if env in ("1", "true", "yes"):
            return True
        return bool(self.config().get("allow_over_budget", False))

    @property
    def max_memory_gib(self) -> float | None:
        """Optional hard cap on the memory budget (Settings → Preferences). None = automatic."""
        v = self.config().get("max_memory_gib")
        try:
            return float(v) if v not in (None, "", 0) else None
        except (TypeError, ValueError):
            return None

    @property
    def device_preference(self) -> str:
        """``auto`` (default), ``cpu`` or ``cuda``. Environment first, then config.json."""
        env = os.environ.get(ENV_DEVICE, "").lower().strip()
        if env in ("cpu", "cuda"):
            return env
        v = str(self.config().get("device") or "").lower().strip()
        return v if v in ("cpu", "cuda") else "auto"

    @property
    def default_threads(self) -> int | None:
        env = os.environ.get(ENV_THREADS)
        if env and env.isdigit():
            return int(env)
        v = self.config().get("default_threads")
        return int(v) if isinstance(v, int) and v > 0 else None

    def ensure_dirs(self) -> None:
        for d in (self.models_dir, self.out_dir, self.uploads_dir, self.jobs_dir):
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls) -> Settings:
        return cls(data_dir=default_data_dir())
