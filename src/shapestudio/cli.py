"""Command line: doctor, install-runtime, download-model, reconstruct, plan, benchmark, jobs,
ui, serve."""
# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
import time
from pathlib import Path

from shapestudio import PRODUCT_NAME, __version__
from shapestudio.config import Settings
from shapestudio.registry import GIB, MODELS, PRESET_NAMES, TIERS, TOOL_MODELS


def _consent_prompt(yes: bool):
    def consent(description: str, size: int) -> bool:
        print(f"Download: {description}")
        if size:
            print(f"Size: {size / GIB:.2f} GiB. Nothing else will be fetched.")
        if yes:
            print("Approved with --yes.")
            return True
        if not sys.stdin.isatty():
            print("No terminal to ask on; re-run with --yes to approve.")
            return False
        answer = input("Proceed? [y/N] ").strip().lower()
        return answer in ("y", "yes")

    return consent


class _Bar:
    def __init__(self) -> None:
        self._last = 0.0
        self._name = ""

    def __call__(self, name: str, done: int, total: int) -> None:
        now = time.monotonic()
        if name != self._name:
            self._name = name
            print()
        if now - self._last < 0.5 and done < total:
            return
        self._last = now
        pct = (done / total * 100) if total else 0
        if total >= GIB:
            amount = f"{done / GIB:6.2f}/{total / GIB:6.2f} GiB"
        else:
            amount = f"{done / 2**20:6.1f}/{total / 2**20:6.1f} MiB"
        print(f"\r  {name}: {amount} ({pct:5.1f}%)", end="", flush=True)
        if done >= total:
            print()


def _studio(start_worker: bool):
    from shapestudio.service import Studio

    return Studio(start_worker=start_worker)


def cmd_doctor(args: argparse.Namespace) -> int:
    studio = _studio(start_worker=False)
    caps = studio.capabilities()
    if args.json:
        print(json.dumps(caps, indent=2))
        return 0
    hw = caps["hardware"]
    rt = caps["runtime"]
    t = rt["torch"]
    print(f"{PRODUCT_NAME} {__version__}")
    print(f"Data dir: {caps['paths']['data_dir']}")
    print()
    print("Hardware")
    print(f"  OS/arch: {hw['os']} {hw['arch']}")
    print(f"  CPU: {hw['cpu_name']} ({hw['physical_cores']} cores, {hw['logical_cores']} threads)")
    print(f"  RAM: {hw['ram_total_gib']} GiB total, {hw['ram_available_gib']} GiB available")
    if hw["gpus"]:
        for g in hw["gpus"]:
            vram = f"{g['vram_gib']} GiB" if g["vram_gib"] else "unknown VRAM"
            shared = " (shared with RAM)" if g["shared_memory"] else ""
            print(f"  GPU: {g['name']} [{g['kind']}, {vram}{shared}]")
    else:
        print("  GPU: none found")
    print()
    print("Runtime")
    if t["installed"]:
        where = " (from the studio's runtime folder)" if t["from_runtime_dir"] else ""
        print(f"  PyTorch {t['version']}: {t['flavor']}{where}")
        if t["cuda_available"]:
            print(f"  CUDA {t['cuda_version']}: {t['device_name']} ({t['device_vram_gib']} GiB)")
        elif t.get("device_name"):
            print(f"  CUDA: {t['device_name']}")
    else:
        print(f"  PyTorch: NOT importable ({t['error']})")
    ov = rt.get("override") or {}
    if ov.get("state") not in (None, "none"):
        print(
            f"  CUDA build override: {ov['state']}"
            + (f" ({ov['detail']})" if ov.get("detail") else "")
        )
    ci = rt["cuda_install"]
    if ci.get("possible"):
        size = f"about {ci['gib']} GiB" if ci.get("gib") else "size: see note"
        print(f"  CUDA runtime install: {ci['method']} ({size}). {ci.get('note', '')}")
    else:
        print(f"  CUDA runtime install: not possible here: {ci.get('reason')}")
    print(f"  Python {rt['python']}{' (packaged app)' if rt['frozen_app'] else ''}")
    print()
    print("Background removal")
    bg = caps["background_engine"]
    print(
        f"  {bg['name']}: {'available' if bg['available'] else 'NOT available'} "
        f"{bg['version'] or ''}"
    )
    for m in bg["models"]:
        mark = "*" if m["installed"] else " "
        print(
            f"  {mark} {m['name']} ({m['licence']}, {m['size_mb']} MB)"
            f"{' default' if m['default'] else ''}"
        )
    print()
    print("Models (tiers)")
    for m in caps["models"]:
        inst = "installed" if m["installed"] else f"not downloaded ({m['download_gib']} GiB)"
        fit = "fits" if m["fits_this_machine"] else f"does NOT fit: {m['fit_note']}"
        print(
            f"  {m['display_name']} [{m['tier']}] ({m['name']}, {m['licence']}): "
            f"{inst}; {fit}; device {m['device']}"
        )
        print(f"    status: {m['verified']}")
        if m["engine_status"] and not m["engine_available"]:
            print(f"    engine: {m['engine_status'].get('reason')}")
        for p in m["presets"]:
            if "error" in p:
                print(f"    preset {p['preset']}: {p['error']}")
                continue
            et = p["expected_time"]
            tag = "" if et["calibrated"] else " (not measured yet)"
            print(
                f"    preset {p['preset']}: {p['resolution']}³ grid on {p['device']}: "
                f"about {et['human']}{tag}"
            )
    print()
    q = caps["queue"]
    worker = (
        f"pid {q['worker_pid']} (v{q['worker_version']})" if q["worker_pid"] else "none running"
    )
    print(
        f"Queue: {q['queued']} queued, running {q['running'] or '-'}, worker {worker}"
        f"{' PAUSED' if q['paused'] else ''}"
    )
    print("Policy: quality is a setting, never a consequence. Default preset: best.")
    print(
        "Licences: " + "; ".join(f"{li['component']}: {li['licence']}" for li in caps["licences"])
    )
    ex = "; ".join(f"{e['model']} ({e['reason']})" for e in caps["excluded_by_policy"])
    print(f"Excluded by policy: {ex}")
    if args.verify:
        print()
        print("Verifying installed file hashes (slow)...")
        for m in MODELS.values():
            for f in m.files:
                if studio.models.path_for(f).exists():
                    ok = studio.models.verify_file(f)
                    print(f"  {f.filename}: {'ok' if ok else 'MISMATCH'}")
        for t in TOOL_MODELS.values():
            if studio.models.path_for(t.file).exists():
                print(
                    f"  {t.file.filename}: "
                    f"{'ok' if studio.models.verify_file(t.file) else 'MISMATCH'}"
                )
    return 0


def cmd_install_runtime(args: argparse.Namespace) -> int:
    from shapestudio import runtime

    settings = Settings.load()
    settings.ensure_dirs()
    if args.remove:
        freed = runtime.remove_cuda_runtime(settings)
        print(
            f"CUDA runtime removed ({freed / GIB:.2f} GiB freed). "
            "The bundled CPU build is used again."
        )
        return 0
    st = runtime.torch_status()
    if not args.cuda:
        if st.installed:
            print(f"PyTorch {st.version} ({st.flavor}) is installed at {st.location}.")
            if st.cuda_available:
                print(f"CUDA device: {st.device_name} ({st.device_vram_gib} GiB)")
            else:
                print(
                    "No CUDA device is visible to it. For an NVIDIA GPU: "
                    "shapestudio install-runtime --cuda"
                )
            return 0
        print(f"PyTorch is not importable: {st.error}")
        print("pip/pipx installs carry it as a dependency; reinstall the package, or for a GPU run")
        print("  shapestudio install-runtime --cuda")
        return 1
    plan = runtime.cuda_install_plan(settings)
    if not plan["possible"]:
        print(f"Not possible here: {plan['reason']}")
        return 1
    from shapestudio.hardware import detect
    from shapestudio.planner import cuda_gpu

    if cuda_gpu(detect(settings.hardware_cache_file)) is None:
        print(
            "Note: no NVIDIA GPU was detected on this machine (nvidia-smi not found or no CUDA "
            "device). The CUDA build will run on the CPU only; continue if you know better."
        )
    print(f"Method: {plan['method']}. {plan.get('note', '')}")
    bar = _Bar()
    try:
        r = runtime.install_cuda_runtime(
            settings, _consent_prompt(args.yes), progress=lambda d, t: bar("torch (CUDA)", d, t)
        )
    except RuntimeError as e:
        print(f"error: {e}")
        return 1
    if not r.get("installed"):
        print("Not installed.")
        return 1
    print(r.get("message") or "Installed.")
    if r.get("status"):
        s = r["status"]
        print(
            f"PyTorch {s['version']}: cuda_available={s['cuda_available']} "
            f"device={s['device_name']}"
        )
    return 0


def cmd_download_model(args: argparse.Namespace) -> int:
    from shapestudio.service import StudioError

    studio = _studio(start_worker=False)
    try:
        info = studio.download_info(args.name)
    except StudioError as e:
        print(f"error: {e}")
        return 2
    if info["already_installed"]:
        print(f"{info['display_name']} is already installed.")
        return 0
    print(f"{info['display_name']} ({info['licence']}): {info.get('note', '')}")
    if info.get("verified"):
        print(f"Status: {info['verified']}")
    for f in info["files"]:
        print(
            f"  need  {f['filename']}  {f['size_gib']:.3f} GiB  "
            f"sha256 {f['sha256'][:16]}…  from {f['source']}"
        )
    bar = _Bar()
    try:
        ok = studio.install(args.name, _consent_prompt(args.yes), progress=bar)
    except Exception as e:
        print(f"\nerror: {e}")
        return 1
    print(f"{info['display_name']} {'installed.' if ok else 'not downloaded.'}")
    return 0 if ok else 1


def _follow(studio, job_id: str) -> dict:
    last = ""
    try:
        while True:
            st = studio.job_status(job_id)
            line = f"{st['status']}/{st['phase']} {st['progress'] * 100:5.1f}%  {st['summary']}"
            if line != last:
                print(f"\r  {line:<78}", end="", flush=True)
                last = line
            if st["status"] in ("done", "failed", "cancelled"):
                print()
                return st
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nCancelling...")
        studio.cancel(job_id)
        time.sleep(1)
        return studio.job_status(job_id)


def cmd_reconstruct(args: argparse.Namespace) -> int:
    from shapestudio.service import StudioError

    studio = _studio(start_worker=not args.no_wait)
    try:
        job = studio.reconstruct(
            args.image,
            model=args.tier or args.model,
            preset=args.preset,
            resolution=args.resolution,
            foreground_ratio=args.foreground,
            remove_background=not args.no_remove_bg,
            threads=args.threads,
        )
    except StudioError as e:
        print(f"error: {e}")
        studio.close()
        return 1
    print(
        f"Job {job.id}: {job.model} [{job.tier}] on {job.device}, preset {job.preset}, "
        f"{job.resolution}³ grid"
    )
    from shapestudio.calibration import human_duration

    print(f"Estimate: {human_duration(job.estimate_seconds)} ({job.estimate_basis})")
    if job.fit_note:
        print(f"Fit: {job.fit_note}")
    for w in job.warnings:
        print(f"Warning: {w}")
    if args.no_wait:
        print(
            "Queued; a running studio (ui or serve) will pick it up. `shapestudio jobs` to follow."
        )
        studio.close()
        return 0
    st = _follow(studio, job.id)
    code = 0
    if st["status"] == "done":
        print(f"GLB: {st['glb_path']}")
        print(f"OBJ: {st['obj_path']}")
        if st["preview_path"]:
            print(f"Turntable: {st['preview_path']}")
        print(
            f"{st['vertices']} vertices, {st['faces']} faces; "
            f"took {human_duration(st['total_seconds'] or 0)} "
            f"(load {human_duration(st['load_seconds'] or 0)}, "
            f"network {human_duration(st['network_seconds'] or 0)}, "
            f"grid {human_duration(st['grid_seconds'] or 0)})"
        )
        if args.out:
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            for key in ("glb_path", "obj_path", "preview_path"):
                if st.get(key):
                    shutil.copy2(st[key], out / Path(st[key]).name)
            print(f"Copied to {out}")
    else:
        print(f"{st['status']}: {st['error'] or st['message']}")
        code = 1
    studio.close()
    return code


def cmd_plan(args: argparse.Namespace) -> int:
    from shapestudio.service import StudioError

    studio = _studio(start_worker=False)
    try:
        p = studio.plan(args.tier or args.model, args.preset, args.resolution)
    except StudioError as e:
        print(f"refused: {e}")
        return 1
    d = p.to_dict()
    if args.json:
        print(json.dumps(d, indent=2))
        return 0
    load = d["load"]
    bar = "#" * min(40, load["score"] * 40 // 100) + "-" * max(0, 40 - load["score"] * 40 // 100)
    print(
        f"{d['display_name']} [{d['tier']}] on {d['device']}, preset {d['preset']}: "
        f"{d['resolution']}³ grid"
    )
    print(f"Expected time: {d['estimate']['human']} ({d['estimate']['basis']})")
    print(
        f"Load [{bar}] {load['score']}% of the {load['device']} budget: {load['label']} "
        f"(needs {load['required_gib']} GiB, budget {load['budget_gib']} GiB, "
        f"free now {load['available_gib']} GiB)"
    )
    if d["fit_note"]:
        print(f"Fit: {d['fit_note']}")
    for w in d["warnings"]:
        print(f"Warning: {w}")
    if not d["installed"]:
        print("Model files not downloaded yet.")
    print(f"Status: {d['verified']}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from shapestudio.service import StudioError

    studio = _studio(start_worker=True)
    last = ""

    def show(st: dict) -> None:
        nonlocal last
        line = f"{st['status']}/{st['phase']} {st['summary']}"
        if line != last:
            print(f"\r  {line:<78}", end="", flush=True)
            last = line

    print(f"Benchmark: one {args.preset} reconstruction of the built-in test image.")
    try:
        p = studio.benchmark(args.preset, wait=1.0, on_status=show)
    except StudioError as e:
        print(f"\nerror: {e}")
        studio.close()
        return 1
    studio.close()
    print()
    if args.json:
        print(json.dumps(p, indent=2))
        return 0
    from shapestudio.calibration import human_duration

    print(
        f"Machine: {p['cpu']}, {p['physical_cores']} cores, {p['ram_total_gib']} GiB RAM, "
        f"GPU {p['gpu'] or 'none usable'}"
    )
    print(
        f"Measured: {p['model']} on {p['device']}, {p['resolution']}³ grid: "
        f"wall {human_duration(p['wall_seconds'])} "
        f"(load {human_duration(p['load_seconds'])}, "
        f"network {human_duration(p['network_seconds'])}, "
        f"grid {human_duration(p['grid_seconds'])}), peak RSS {p['peak_rss_gib']} GiB, "
        f"{p['vertices']} vertices / {p['faces']} faces"
    )
    print("Expected per preset now (measured on this machine):")
    for name, secs in p["expected_seconds_per_preset"].items():
        print(f"  {name:<9} {human_duration(secs) if secs is not None else 'n/a'}")
    print(f"Profile written to {studio.settings.profile_file}")
    return 0


def cmd_jobs(args: argparse.Namespace) -> int:
    studio = _studio(start_worker=False)
    rows = studio.list_jobs()
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        print("No jobs.")
    for r in rows:
        print(
            f"{r['id']}  {r['status']:<9} {r['tier']:<8} {r['preset']:<8} "
            f"{r['resolution']}³  {r['source_name']}"
        )
        if r["glb_path"]:
            print(f"    {r['glb_path']}")
        if r["error"]:
            print(f"    error: {r['error']}")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    from shapestudio.ui import serve_ui

    serve_ui(args.host, args.port, start_worker=not args.no_worker)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from shapestudio.server import serve

    serve(args.transport, args.host, args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="shapestudio",
        description=f"{PRODUCT_NAME}: free, local, permissively licensed image-to-3D.",
    )
    p.add_argument("--version", action="version", version=f"shapestudio {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="show hardware, runtime, models, tiers and expected times")
    d.add_argument("--json", action="store_true")
    d.add_argument("--verify", action="store_true", help="re-hash installed model files")
    d.set_defaults(fn=cmd_doctor)

    ir = sub.add_parser(
        "install-runtime", help="report PyTorch, or install the CUDA build (consent)"
    )
    ir.add_argument("--cuda", action="store_true", help="install the CUDA build of PyTorch")
    ir.add_argument(
        "--remove", action="store_true", help="remove the unpacked CUDA runtime (packaged app)"
    )
    ir.add_argument("--yes", "-y", action="store_true", help="approve the download")
    ir.set_defaults(fn=cmd_install_runtime)

    dm = sub.add_parser("download-model", help="download a model's weights (asks for consent)")
    dm.add_argument(
        "name",
        help="a model ("
        + ", ".join(MODELS)
        + ") or background model ("
        + ", ".join(TOOL_MODELS)
        + ")",
    )
    dm.add_argument("--yes", "-y", action="store_true", help="approve the download")
    dm.set_defaults(fn=cmd_download_model)

    r = sub.add_parser("reconstruct", help="turn one picture into a mesh and wait for it")
    r.add_argument("image", help="a PNG, JPEG or WebP path")
    r.add_argument("--tier", choices=list(TIERS), help="best (TRELLIS, GPU) or standard (TripoSR)")
    r.add_argument("--model", choices=list(MODELS))
    r.add_argument("--preset", choices=PRESET_NAMES, default="best")
    r.add_argument(
        "--resolution", type=int, help="surface grid 64-512 (multiple of 32); overrides the preset"
    )
    r.add_argument(
        "--foreground", type=float, help="object size in the frame 0.5-1.0 (default 0.85)"
    )
    r.add_argument(
        "--no-remove-bg",
        action="store_true",
        help="the picture already has a transparent background",
    )
    r.add_argument("--threads", type=int, help="CPU threads (default: PyTorch's choice)")
    r.add_argument("--out", help="also copy the GLB, OBJ and turntable to this folder")
    r.add_argument(
        "--no-wait", action="store_true", help="queue and exit (needs ui or serve running)"
    )
    r.set_defaults(fn=cmd_reconstruct)

    pl = sub.add_parser("plan", help="dry run: expected time and load meter for settings")
    pl.add_argument("--tier", choices=list(TIERS))
    pl.add_argument("--model", choices=list(MODELS))
    pl.add_argument("--preset", choices=PRESET_NAMES, default="best")
    pl.add_argument("--resolution", type=int)
    pl.add_argument("--json", action="store_true")
    pl.set_defaults(fn=cmd_plan)

    b = sub.add_parser(
        "benchmark", help="run one real reconstruction and write the machine profile"
    )
    b.add_argument("--preset", choices=PRESET_NAMES, default="fast")
    b.add_argument("--json", action="store_true")
    b.set_defaults(fn=cmd_benchmark)

    j = sub.add_parser("jobs", help="list jobs")
    j.add_argument("--json", action="store_true")
    j.set_defaults(fn=cmd_jobs)

    u = sub.add_parser("ui", help="run the local web UI")
    u.add_argument("--host", default="127.0.0.1")
    u.add_argument("--port", type=int, default=8776)
    u.add_argument(
        "--no-worker",
        action="store_true",
        help="only submit and view jobs; another process runs them",
    )
    u.set_defaults(fn=cmd_ui)

    s = sub.add_parser("serve", help="run the MCP server")
    s.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8775)
    s.set_defaults(fn=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    from shapestudio import runtime

    # legacy Windows consoles are cp1252: keep ≈ · ³ from crashing a print
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.reconfigure(errors="replace")

    runtime.activate_runtime(Settings.load())
    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
