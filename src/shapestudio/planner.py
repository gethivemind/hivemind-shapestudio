"""Turns a request (model or tier, preset) into a concrete plan: which weights, on which
device, at which grid resolution, how long it is expected to take on this machine, and why.

Policy (ADR-051): quality is a setting, never a consequence. The only things that decide
*which* model runs are what the caller asked for and what physically fits (memory, and for
GPU-only models the presence of a suitable GPU). When the planner's automatic choice is not
the strongest model in the catalogue, the plan says so in ``fit_note``. Nothing here lowers
the grid resolution on its own.
"""
# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

from shapestudio.calibration import Calibration, Estimate
from shapestudio.hardware import GIB, Gpu, Hardware
from shapestudio.models import ModelStore
from shapestudio.registry import MODELS, PRESET_NAMES, TIERS, MeshModel, Preset

HEADROOM_BYTES = 2 * GIB  # for the OS and the caller
# Working memory of a TripoSR-class reconstruction at a 256³ grid: the grid positions, the
# chunked NeRF queries, the density volume and marching cubes' own buffers. Scales with cells.
WORKING_BYTES_AT_256 = int(2.5 * GIB)
MIN_RESOLUTION = 64
MAX_RESOLUTION = 512
RESOLUTION_STEP = 32
MIN_FOREGROUND = 0.5
MAX_FOREGROUND = 1.0


class PlanError(ValueError):
    pass


UNLOCK_HINT = (
    "Over-budget runs are locked by default. To try anyway, set "
    'SHAPESTUDIO_ALLOW_OVER_BUDGET=1 or put {"allow_over_budget": true} in config.json '
    "in the data directory; the run may swap for a long time or fail to allocate."
)


@dataclass
class Load:
    """The performance meter: what the job needs against what the machine has."""

    required_bytes: int
    budget_bytes: int
    available_bytes: int
    device: str

    @property
    def score(self) -> int:
        if self.budget_bytes <= 0:
            return 999
        return round(100 * self.required_bytes / self.budget_bytes)

    @property
    def label(self) -> str:
        if self.score > 100:
            return "over budget"
        if self.required_bytes > self.available_bytes:
            return "will swap"
        if self.score >= 70:
            return "tight"
        return "comfortable"

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "label": self.label,
            "device": self.device,
            "required_gib": round(self.required_bytes / GIB, 1),
            "budget_gib": round(self.budget_bytes / GIB, 1),
            "available_gib": round(self.available_bytes / GIB, 1),
            "meaning": {
                "comfortable": "fits in free memory with room to spare",
                "tight": "fits the budget; close to the machine's limit",
                "will swap": "fits total memory but not what is free now; expect it to be "
                "several times slower",
                "over budget": "needs more memory than this machine has; locked unless "
                "allow_over_budget is set",
            }[self.label],
        }


@dataclass
class Fit:
    model: MeshModel
    device: str  # cpu | cuda | none
    required_bytes: int
    budget_bytes: int
    reason: str | None = None  # why it does not fit, in words

    @property
    def fits(self) -> bool:
        return self.reason is None and self.required_bytes <= self.budget_bytes


@dataclass
class Plan:
    model: MeshModel
    preset: Preset
    device: str
    resolution: int
    foreground_ratio: float
    estimate: Estimate
    fit_note: str | None = None
    installed: bool = True
    warnings: list[str] = field(default_factory=list)
    load: Load | None = None
    over_budget: bool = False
    remove_background: bool = True

    def to_dict(self) -> dict:
        return {
            "model": self.model.name,
            "display_name": self.model.display_name,
            "tier": self.model.tier,
            "preset": self.preset.name,
            "device": self.device,
            "resolution": self.resolution,
            "foreground_ratio": self.foreground_ratio,
            "remove_background": self.remove_background,
            "estimate": self.estimate.to_dict(),
            "fit_note": self.fit_note,
            "installed": self.installed,
            "warnings": list(self.warnings),
            "load": self.load.to_dict() if self.load else None,
            "over_budget": self.over_budget,
            "verified": self.model.verified,
        }


def cuda_gpu(hw: Hardware) -> Gpu | None:
    """The NVIDIA GPU the CUDA runtime would use, if any."""
    for g in hw.gpus:
        if g.kind == "cuda":
            return g
    return None


def memory_budget(hw: Hardware, device: str, cap_bytes: int | None = None) -> int:
    """Bytes available for weights plus working memory on ``device``. ``cap_bytes`` is the
    user's own limit (Settings → Memory limit) and only ever lowers the CPU budget."""
    if device == "cuda":
        g = cuda_gpu(hw)
        return int(g.vram_bytes or 0) if g else 0
    budget = hw.ram_total_bytes - HEADROOM_BYTES
    if cap_bytes:
        budget = min(budget, cap_bytes)
    return max(0, budget)


def working_bytes(resolution: int) -> int:
    return int(WORKING_BYTES_AT_256 * (resolution / 256) ** 3)


def required_bytes(model: MeshModel, device: str, resolution: int) -> int:
    weights = model.weights_bytes_in_memory
    if device == "cuda":
        weights = weights // 2 if model.engine == "triposr" else weights  # fp16 on the GPU
    return weights + working_bytes(resolution)


def fit_for(
    model: MeshModel,
    hw: Hardware,
    device: str,
    resolution: int,
    cap_bytes: int | None = None,
    cuda_runtime: bool = True,
) -> Fit:
    need = required_bytes(model, device, resolution)
    budget = memory_budget(hw, device, cap_bytes)
    reason = None
    if device == "cuda":
        g = cuda_gpu(hw)
        if g is None:
            reason = "needs an NVIDIA GPU and this machine has none"
        elif not cuda_runtime:
            reason = "the CUDA runtime is not installed (shapestudio install-runtime --cuda)"
        elif model.min_vram_bytes and (g.vram_bytes or 0) < model.min_vram_bytes:
            reason = (
                f"needs a GPU with at least {model.min_vram_bytes / GIB:.0f} GiB of memory; "
                f"{g.name} has {(g.vram_bytes or 0) / GIB:.0f} GiB"
            )
    elif not model.cpu_capable:
        reason = f"{model.display_name} has no CPU path"
    return Fit(model, device, need, budget, reason)


def snap_resolution(value: int) -> int:
    if not MIN_RESOLUTION <= value <= MAX_RESOLUTION:
        raise PlanError(
            f"resolution must be between {MIN_RESOLUTION} and {MAX_RESOLUTION}, got {value}"
        )
    return (value // RESOLUTION_STEP) * RESOLUTION_STEP


class Planner:
    def __init__(
        self,
        hw: Hardware,
        store: ModelStore,
        calibration: Calibration,
        allow_over_budget: bool = False,
        cap_bytes: int | None = None,
        device_preference: str = "auto",
        cuda_runtime: bool = False,
    ) -> None:
        self.hw = hw
        self.store = store
        self.calibration = calibration
        self.allow_over_budget = allow_over_budget
        self.cap_bytes = cap_bytes
        self.device_preference = device_preference
        self.cuda_runtime = cuda_runtime  # torch with CUDA is importable and sees a device

    # -- device ----------------------------------------------------------------------------

    def device_for(self, model: MeshModel, resolution: int) -> tuple[str, Fit]:
        """The device a model would run on here: CUDA when it fits and is wanted, else CPU.
        Returns the fit on that device (which may not fit either)."""
        pref = self.device_preference
        gpu = fit_for(model, self.hw, "cuda", resolution, self.cap_bytes, self.cuda_runtime)
        cpu = fit_for(model, self.hw, "cpu", resolution, self.cap_bytes, self.cuda_runtime)
        if pref == "cuda":
            return "cuda", gpu
        if pref == "cpu":
            return "cpu", cpu
        if gpu.fits:
            return "cuda", gpu
        if model.cuda_only:
            return "cuda", gpu
        return "cpu", cpu

    # -- model -----------------------------------------------------------------------------

    def resolve_model(self, name: str | None, resolution: int) -> tuple[MeshModel, str | None]:
        """``None``/``"auto"`` picks the strongest catalogue model that is installed and fits;
        failing that, the strongest installed one; failing that, the strongest that fits.
        A tier name (``best``, ``standard``) is accepted in place of a model name. The
        second value explains the choice when it is not the strongest model."""
        if name and name not in ("auto", "best-available"):
            if name in TIERS:
                return next(m for m in MODELS.values() if m.tier == name), None
            if name not in MODELS:
                raise PlanError(
                    f"unknown model {name!r}; known: {', '.join(MODELS)} (or a tier: "
                    f"{', '.join(TIERS)})"
                )
            return MODELS[name], None
        ranked = sorted(MODELS.values(), key=lambda m: m.quality_rank)
        top = ranked[0]
        for m in ranked:
            if self.store.model_installed(m) and self.device_for(m, resolution)[1].fits:
                return m, self._why_not_top(m, top, resolution)
        for m in ranked:
            if self.store.model_installed(m):
                return m, self._why_not_top(m, top, resolution)
        for m in ranked:
            if self.device_for(m, resolution)[1].fits:
                return m, self._why_not_top(m, top, resolution)
        return ranked[-1], self._why_not_top(ranked[-1], top, resolution)

    def _why_not_top(self, chosen: MeshModel, top: MeshModel, resolution: int) -> str | None:
        if chosen is top:
            return None
        _dev, fit = self.device_for(top, resolution)
        if fit.fits:
            return (
                f"Running {chosen.display_name} ({chosen.tier} tier) because "
                f"{top.display_name} ({top.tier} tier) is not downloaded. It would fit this "
                f"machine; download it to use it. Not a speed choice."
            )
        reason = fit.reason or (
            f"needs about {fit.required_bytes / GIB:.1f} GiB and this machine's budget is "
            f"{fit.budget_bytes / GIB:.1f} GiB"
        )
        return (
            f"Running {chosen.display_name} ({chosen.tier} tier) because {top.display_name} "
            f"({top.tier} tier) {reason}. This is a hardware fit, not a speed choice; the "
            f"{chosen.tier} tier runs at its full settings."
        )

    # -- plan ------------------------------------------------------------------------------

    def plan(
        self,
        model_name: str | None = None,
        preset_name: str = "best",
        resolution: int | None = None,
        foreground_ratio: float | None = None,
        remove_background: bool = True,
    ) -> Plan:
        if preset_name not in PRESET_NAMES:
            raise PlanError(f"unknown preset {preset_name!r}; one of {', '.join(PRESET_NAMES)}")
        if (
            foreground_ratio is not None
            and not MIN_FOREGROUND <= foreground_ratio <= MAX_FOREGROUND
        ):
            raise PlanError(
                f"foreground_ratio must be between {MIN_FOREGROUND} and {MAX_FOREGROUND}"
            )
        res0 = snap_resolution(resolution) if resolution else None
        model, fit_note = self.resolve_model(model_name, res0 or 256)
        preset = model.presets[preset_name]
        res = res0 or preset.resolution
        device, fit = self.device_for(model, res)
        warnings: list[str] = []
        over_budget = False
        if not fit.fits:
            if fit.reason:
                msg = f"{model.display_name} {fit.reason}."
                if model.cuda_only or (device == "cuda" and self.device_preference == "cuda"):
                    raise PlanError(
                        msg + " Choose the standard tier (TripoSR), which runs on a CPU."
                    )
                raise PlanError(msg)
            msg = (
                f"{model.display_name} at a {res}³ grid needs about {fit.required_bytes / GIB:.1f} "
                f"GiB on the {device}; this machine's budget is {fit.budget_bytes / GIB:.1f} GiB."
            )
            if not self.allow_over_budget:
                raise PlanError(f"{msg} Choose a lower resolution or preset. {UNLOCK_HINT}")
            over_budget = True
            warnings.append(f"{msg} Running over budget because allow_over_budget is set.")
        installed = self.store.model_installed(model)
        estimate = self.calibration.estimate(model, device, res)
        available = (
            self.hw.ram_available_bytes
            if device == "cpu"
            else int((cuda_gpu(self.hw).vram_bytes if cuda_gpu(self.hw) else 0) or 0)
        )
        if device == "cpu" and self.hw.ram_available_bytes < fit.required_bytes:
            warnings.append(
                f"Only {self.hw.ram_available_bytes / GIB:.1f} GiB of RAM is free right now; the "
                f"job needs about {fit.required_bytes / GIB:.1f} GiB. Close other programs or "
                "expect swapping to slow it down."
            )
        if device == "cpu" and cuda_gpu(self.hw) and not self.cuda_runtime:
            warnings.append(
                f"This machine has an NVIDIA GPU ({cuda_gpu(self.hw).name}); the CUDA runtime "
                "is not installed, so the reconstruction runs on the CPU. `shapestudio "
                "install-runtime --cuda` offers the download."
            )
        load = Load(fit.required_bytes, fit.budget_bytes, available, device)
        return Plan(
            model=model,
            preset=preset,
            device=device,
            resolution=res,
            foreground_ratio=0.85 if foreground_ratio is None else foreground_ratio,
            estimate=estimate,
            fit_note=fit_note,
            installed=installed,
            warnings=warnings,
            load=load,
            over_budget=over_budget,
            remove_background=remove_background,
        )

    def preset_table(self, model: MeshModel) -> list[dict]:
        """Presets with the expected time on this machine, for ``capabilities()``."""
        out = []
        for p in model.presets.values():
            try:
                plan = self.plan(model.name, p.name)
            except PlanError as e:
                out.append({"preset": p.name, "tradeoff": p.tradeoff, "error": str(e)})
                continue
            out.append(
                {
                    "preset": p.name,
                    "resolution": plan.resolution,
                    "device": plan.device,
                    "tradeoff": p.tradeoff,
                    "expected_time": plan.estimate.to_dict(),
                    "installed": plan.installed,
                    "load": plan.load.to_dict() if plan.load else None,
                }
            )
        return out
