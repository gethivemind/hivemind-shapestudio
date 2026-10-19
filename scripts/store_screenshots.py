"""Store screenshots (PNG 1440x900) of the running app, captured with Playwright.

The studio is started in-process against a temporary data folder with a demonstration engine:
the mesh pipeline (marching cubes + GLB/OBJ writer + turntable) runs for real on a synthetic
density field, so the gallery shows objects the studio actually produced, while no model
weights are needed. The pictures are labelled "demonstration objects" in the listing.

    .venv/Scripts/python scripts/store_screenshots.py [--out packaging/store-screenshots]
"""
# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import contextlib
import socket
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
PORT = 8798


class DemoEngine:
    """Real surface extraction on a synthetic field: a blobby object per input picture."""

    name = "demo"

    def __init__(self) -> None:
        self._cancel = False

    def available(self) -> bool:
        return True

    def version(self) -> str | None:
        return "demo"

    def cancel(self) -> None:
        self._cancel = True

    def reconstruct(self, spec, on_progress=None):
        from skimage import measure

        from shapestudio.engine.base import EngineResult, Progress
        from shapestudio.engine.meshio import write_glb, write_obj
        from shapestudio.engine.preview import write_previews

        t0 = time.monotonic()
        if on_progress:
            on_progress(Progress("encoding", 0.2, "demo field"))
        seed = sum(spec.input.read_bytes()[:64]) % 7
        r = 96
        ax = np.linspace(-1, 1, r)
        x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
        vol = 0.75 - np.sqrt(x**2 + y**2 + z**2)
        for k in range(3 + seed % 3):
            cx, cy, cz = np.cos(k * 2.1 + seed), np.sin(k * 1.7 + seed), np.cos(k * 0.9)
            vol += 0.22 * np.exp(
                -((x - 0.55 * cx) ** 2 + (y - 0.55 * cy) ** 2 + (z - 0.4 * cz) ** 2) * 14
            )
        verts, faces, _n, _v = measure.marching_cubes(vol, 0.0)
        verts = verts / (r - 1) * 1.74 - 0.87
        faces = np.ascontiguousarray(faces[:, ::-1])
        with Image.open(spec.input) as im:
            rgb = np.asarray(im.convert("RGB").resize((8, 8))).reshape(-1, 3).mean(axis=0) / 255
        cols = np.clip(0.35 + 0.65 * (verts + 0.87) / 1.74, 0, 1) * rgb
        spec.out_dir.mkdir(parents=True, exist_ok=True)
        glb = write_glb(spec.out_dir / f"{spec.stem}.glb", verts, faces, cols, name=spec.stem)
        obj = write_obj(spec.out_dir / f"{spec.stem}.obj", verts, faces, cols)
        preview = spec.out_dir / f"{spec.stem}.preview.png"
        write_previews(verts, faces, cols, preview, spec.out_dir / f"{spec.stem}.thumb.png")
        if on_progress:
            on_progress(Progress("saving", 1.0))
        return EngineResult(
            glb, obj, preview, None, len(verts), len(faces), 0.4, 1.2, 0.8, time.monotonic() - t0
        )


def sample_pictures(folder: Path) -> list[Path]:
    out = []
    for name, colour in (
        ("mug", (205, 92, 48)),
        ("lamp", (60, 110, 200)),
        ("toy-car", (70, 160, 90)),
    ):
        img = Image.new("RGB", (640, 480), (246, 245, 241))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((200, 120, 440, 400), radius=60, fill=colour)
        d.ellipse((150, 60, 490, 250), fill=tuple(min(255, c + 40) for c in colour))
        p = folder / f"{name}.png"
        img.save(p)
        out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "packaging" / "store-screenshots"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import uvicorn
    from playwright.sync_api import sync_playwright

    from shapestudio.config import Settings
    from shapestudio.engine.fake import FakeBackgroundEngine
    from shapestudio.hardware import Gpu, Hardware
    from shapestudio.registry import TOOL_MODELS, TRIPOSR
    from shapestudio.service import Studio
    from shapestudio.ui import build_app

    tmp = Path(tempfile.mkdtemp(prefix="shapestudio-shots-"))
    settings = Settings(data_dir=tmp / "data")
    settings.ensure_dirs()
    # pretend TripoSR and IS-Net are installed so the screens show the ready state
    import json

    manifest = {}
    for f in [*TRIPOSR.files, TOOL_MODELS["isnet-general-use"].file]:
        p = settings.models_dir / f.filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"demo")
        manifest[f.filename] = f.sha256
    settings.manifest_file.write_text(json.dumps(manifest))
    import shapestudio.models as models_mod

    models_mod._size_matches = lambda _p, _s: True  # type: ignore[attr-defined]
    gib = 1024**3
    hw = Hardware(
        "Windows",
        "AMD64",
        "12th Gen Intel(R) Core(TM) i5-1235U",
        10,
        12,
        16 * gib,
        9 * gib,
        [Gpu("Intel(R) UHD Graphics", 2 * gib, "integrated", True)],
    )
    studio = Studio(
        settings=settings,
        engines={"triposr": DemoEngine(), "trellis": DemoEngine()},
        hardware=hw,
        background_engine=FakeBackgroundEngine(),
        cuda_runtime=False,
    )
    studio.add_project("Product shots", "demonstration objects")
    pics = sample_pictures(tmp)
    jobs = [
        studio.reconstruct(str(p), preset=preset, note=note)
        for p, preset, note in zip(
            pics, ("best", "balanced", "fast"), ("hero shot", "", "turntable test"), strict=True
        )
    ]
    for j in jobs:
        for _ in range(600):
            if studio.job_status(j.id)["status"] in ("done", "failed"):
                break
            time.sleep(0.05)
    studio.set_favorite(jobs[0].id, True)

    server = uvicorn.Server(
        uvicorn.Config(build_app(studio), host="127.0.0.1", port=PORT, log_level="warning")
    )
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        with socket.socket() as s:
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", PORT)) == 0:
                break
        time.sleep(0.1)

    url = f"http://127.0.0.1:{PORT}/"
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900}, color_scheme="dark")
        page.goto(url)
        page.wait_for_selector("body[data-booted]")
        page.evaluate("document.documentElement.dataset.theme='dark'")

        def shot(name: str) -> None:
            page.wait_for_timeout(600)
            page.screenshot(path=str(out / name), full_page=False)
            print("wrote", out / name)

        page.evaluate("showTab('home')")
        page.wait_for_timeout(400)
        shot("01-home.png")

        page.evaluate("showTab('create')")
        page.set_input_files("#file", str(pics[0]))
        page.wait_for_selector("#planline:not(:empty)")
        shot("02-create.png")

        page.evaluate("showTab('gallery')")
        page.wait_for_selector(".gcard")
        shot("03-gallery.png")

        page.evaluate(f"openViewer('{jobs[0].id}')")
        page.wait_for_selector("#viewer[open]")
        page.wait_for_timeout(2500)  # model-viewer loads from the CDN and renders
        shot("04-viewer.png")
        page.evaluate("document.getElementById('viewer').close()")

        page.evaluate("showTab('settings')")
        page.wait_for_selector("#comps tbody tr")
        shot("05-settings.png")
        browser.close()
    server.should_exit = True
    studio.close()
    with contextlib.suppress(Exception):
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
