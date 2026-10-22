"""The 0.1.6 round, from the first run on a GPU laptop: the background remover's result type
(every real run failed on it), the CUDA override the launcher never activated, the planner
preferring a model that fits over one that can never run here, requirements, samples, mesh
tools, the problem report and batch uploads."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from starlette.testclient import TestClient

from conftest import TRELLIS, TRIPOSR, fake_install, laptop, make_studio, small_gpu_box
from shapestudio import runtime
from shapestudio.engine import meshtools
from shapestudio.engine.background import BackgroundEngine, BackgroundSpec
from shapestudio.engine.base import BackgroundResult
from shapestudio.registry import TOOL_MODELS
from shapestudio.ui import build_app

ROOT = Path(__file__).resolve().parents[1]


class _StubSession:
    """Stands in for an onnxruntime InferenceSession: a matte that keeps the centre."""

    class _In:
        name = "input"

    def get_inputs(self):
        return [self._In()]

    def run(self, _outs, feeds):
        x = next(iter(feeds.values()))
        n = x.shape[-1]
        yy, xx = np.mgrid[0:n, 0:n]
        matte = (((yy - n / 2) ** 2 + (xx - n / 2) ** 2) < (n / 3) ** 2).astype(np.float32)
        return [matte[None, None]]


def test_real_background_engine_returns_its_own_result_type(tmp_path, sample_image, monkeypatch):
    """0.1.5 returned Image Studio's EngineResult(output=...) here, so TripoSR died with
    "unexpected keyword argument 'output'" on every run with background removal on."""
    eng = BackgroundEngine()
    monkeypatch.setattr(eng, "_session", lambda _p: _StubSession())
    out = tmp_path / "cut.png"
    r = eng.remove_background(BackgroundSpec(sample_image, out, tmp_path / "m.onnx", "isnet"))
    assert isinstance(r, BackgroundResult) and r.output == out and out.exists()
    with Image.open(out) as im:
        assert im.mode == "RGBA"
        a = np.asarray(im)[..., 3]
        assert a.max() > 200 and a.min() < 50  # the matte was applied


def test_activate_runtime_serves_torch_from_the_unpacked_wheel(settings, monkeypatch):
    site = settings.runtime_dir / "cuda" / "site-packages"
    (site / "torch").mkdir(parents=True)
    (site / "torch" / "__init__.py").write_text("FAKE = True\n", encoding="utf-8")
    (settings.runtime_dir / "runtime.json").write_text(
        json.dumps({"flavor": "cuda", "wheel": "torch-x.whl"}), encoding="utf-8"
    )
    settings.write_config({"runtime": "cuda"})
    assert runtime.override_state(settings)["state"] == "pending_restart"
    saved = sys.meta_path[:]
    try:
        monkeypatch.setattr(runtime, "_RUNTIME_ROOT", None)
        # torch is already imported in this process, which is exactly when the override must
        # refuse (it would be a lie to say the CUDA build is active)
        assert runtime.activate_runtime(settings) is None
        monkeypatch.delitem(sys.modules, "torch", raising=False)
        root = runtime.activate_runtime(settings)
        assert root == site
        spec = sys.meta_path[0].find_spec("torch")
        assert spec is not None and Path(spec.origin).parent == site / "torch"
        assert sys.meta_path[0].find_spec("numpy") is None
    finally:
        sys.meta_path[:] = saved
        monkeypatch.setattr(runtime, "_RUNTIME_ROOT", None)
    settings.write_config({"runtime": "cpu"})
    assert runtime.override_state(settings)["state"] == "disabled"
    assert runtime.remove_cuda_runtime(settings) > 0
    assert runtime.override_state(settings)["state"] == "none"


def test_launcher_activates_the_override_before_the_server_and_can_restart(tmp_path, monkeypatch):
    src = (ROOT / "packaging" / "launcher.py").read_text(encoding="utf-8")
    # order matters: the override must be installed before Server.start imports torch
    assert src.index("runtime.activate_runtime(settings)") < src.index("server.start()")
    assert "studio.on_restart = app.restart" in src
    spec = importlib.util.spec_from_file_location(
        "launcher_0_1_6", ROOT / "packaging" / "launcher.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert "restart" in {n for n in dir(mod.JsApi) if not n.startswith("_")}
    calls = []
    monkeypatch.setattr(
        mod.subprocess if hasattr(mod, "subprocess") else __import__("subprocess"),
        "Popen",
        lambda *a, **k: calls.append(a),
    )
    mod.relaunch()
    assert calls and calls[0][0][0] == sys.executable


def test_auto_prefers_a_model_that_fits_over_one_that_can_never_run_here(settings):
    """The GPU laptop had downloaded TRELLIS (6 GiB GPU: never fits) and not TripoSR; auto
    then refused. Now it plans TripoSR, says it is not downloaded, and hints the download."""
    fake_install(settings, TRELLIS)
    st = make_studio(settings, small_gpu_box(), cuda_runtime=False)
    try:
        p = st.plan()
        assert p.model is TRIPOSR and not p.installed
        assert p.hints == {"install_model": "triposr", "install_cuda_runtime": True} or (
            p.hints.get("install_model") == "triposr" and p.hints.get("install_cuda_runtime")
        )
        d = p.to_dict()
        # the hardware verdict, not the missing runtime: installing CUDA would not change it
        assert "16 GiB" in d["fit_note"] and "12 GiB" in d["fit_note"]
        assert "install-runtime" not in d["fit_note"]
        assert any("CUDA build of PyTorch" in w and "Settings" in w for w in d["warnings"])
        assert st.recommended_model() is TRIPOSR
        caps = st.capabilities()
        assert caps["recommended_model"] == "triposr" and caps["setup"]["ready"] is False
        assert [m["name"] for m in caps["setup"]["missing"]] == ["triposr", "isnet-general-use"]
        req = next(m for m in caps["models"] if m["name"] == "trellis-image-large")
        assert req["requirements"]["min_vram_gib"] == 16
        assert req["requirements_met"]["overall"] == "below"
        tri = next(m for m in caps["models"] if m["name"] == "triposr")
        assert tri["requirements_met"]["overall"] in ("minimum", "recommended")
        assert tri["max_resolution_fits"] >= 256
        comps = st.components()
        trellis = next(c for c in comps if c["name"] == "trellis-image-large")
        assert trellis["fits_this_machine"] is False and "16 GiB" in trellis["fit_note"]
        rt = next(c for c in comps if c["kind"] == "runtime")
        assert rt["override"]["state"] == "none" and rt["gpu_name"] == "NVIDIA RTX 3060"
    finally:
        st.close()


def test_laptop_first_run_writes_the_profile_and_keeps_the_source_name(studio, sample_image):
    client = TestClient(build_app(studio))
    assert studio.profile() is None
    with sample_image.open("rb") as fh:
        r = client.post(
            "/api/reconstruct",
            files={"file": ("my chair.png", fh, "image/png")},
            data={"preset": "fast"},
        )
    assert r.status_code == 200
    jid = r.json()["id"]
    for _ in range(300):
        st = client.get(f"/api/jobs/{jid}").json()
        if st["status"] in ("done", "failed"):
            break
        time.sleep(0.02)
    assert st["status"] == "done" and st["source_name"] == "my chair.png"
    assert st["title"].startswith("my chair · fast 128³")
    for _ in range(200):  # the finish hook writes the profile just after the status flips
        prof = studio.profile()
        if prof:
            break
        time.sleep(0.02)
    assert prof and prof["source"] == "first reconstruction" and prof["job_id"] == jid
    assert studio.setup_state()["measured"] is True and studio.setup_state()["first_run_done"]
    # mesh tools on the finished job
    for op in ("stl", "flip", "smooth", "decimate", "scale"):
        r = client.post(
            f"/api/jobs/{jid}/mesh", json={"op": op, "params": {"size_mm": 50, "target_faces": 6}}
        )
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["faces"] > 0 and d["name"].startswith(f"{jid}.derived-{op}-1")
    st = client.get(f"/api/jobs/{jid}").json()
    assert len(st["derived"]) == 5 and {d["format"] for d in st["derived"]} == {"stl", "glb"}
    stl = client.get(f"/api/derived/{jid}/{st['derived'][0]['name']}")
    assert stl.status_code == 200 and len(stl.content) > 84
    assert client.get(f"/api/derived/{jid}/../secret").status_code in (400, 404)
    assert client.post(f"/api/jobs/{jid}/mesh", json={"op": "explode"}).status_code == 400
    # the report bundle
    rep = client.get("/api/report").json()
    assert "Presidium Shape Studio" in rep["body"] and rep["issues_url"].endswith("/issues/new")
    assert "Models installed: triposr" in rep["body"]


def test_batch_upload_queues_one_job_per_picture_and_samples_work(studio, sample_image):
    client = TestClient(build_app(studio))
    data = sample_image.read_bytes()
    r = client.post(
        "/api/reconstruct",
        files=[("file", ("a.png", data, "image/png")), ("file", ("b.png", data, "image/png"))],
        data={"preset": "fast"},
    )
    assert r.status_code == 200 and len(r.json()["jobs"]) == 2
    names = {j["source_name"] for j in r.json()["jobs"]}
    assert names == {"a.png", "b.png"}
    for name in ("sphere", "box", "mug"):
        s = client.get(f"/api/sample/{name}")
        assert s.status_code == 200 and s.content[:4] == b"\x89PNG"
    assert client.get("/api/sample/teapot").status_code == 404
    r = client.post("/api/reconstruct", json={"image": "sample:mug", "preset": "fast"})
    assert r.status_code == 200 and r.json()["source_name"] == "sample-mug.png"
    assert client.post("/api/restart").json()["restarted"] is False
    for _ in range(400):
        if all(j["status"] in ("done", "failed") for j in client.get("/api/jobs").json()["jobs"]):
            break
        time.sleep(0.02)


def test_mesh_tools_on_a_cube(tmp_path):
    from shapestudio.engine.meshio import cube, write_obj

    v, f, c = cube()
    obj = write_obj(tmp_path / "c.obj", v, f, c)
    v2, f2, c2 = meshtools.read_obj(obj)
    assert v2.shape == (8, 3) and f2.shape == (12, 3) and c2.shape == (8, 3)
    assert np.array_equal(meshtools.flip(f2)[:, 0], f2[:, 2])
    sm = meshtools.smooth(v2, f2, iterations=1)
    assert np.abs(sm).max() < np.abs(v2).max()  # moved inwards
    sv, sf, sc = meshtools.decimate(v2, f2, c2, target_faces=4)
    assert sc is not None and len(sc) == len(sv)
    assert len(sf) <= 12 and sf.max() < len(sv)
    scaled = meshtools.scale_to(v2, 10.0)
    assert np.isclose((scaled.max(axis=0) - scaled.min(axis=0)).max(), 10.0)
    p = meshtools.write_stl(tmp_path / "c.stl", v2, f2)
    assert p.stat().st_size == 84 + 50 * 12
    r = meshtools.apply("stl", obj, tmp_path / "d", {"size_mm": 20})
    assert r["format"] == "stl" and "20 mm" in r["note"]
    with pytest.raises(ValueError):
        meshtools.apply("nope", obj, tmp_path / "e")


def test_internal_errors_read_as_a_bug_report_not_a_traceback(settings, sample_image):
    class Boom:
        name = "fake"

        def available(self):
            return True

        def version(self):
            return "x"

        def cancel(self):
            pass

        def reconstruct(self, spec, on_progress=None):
            raise TypeError("EngineResult.__init__() got an unexpected keyword argument 'output'")

    from shapestudio.engine.fake import FakeBackgroundEngine
    from shapestudio.engine.trellis import TrellisEngine
    from shapestudio.service import Studio

    fake_install(settings, TRIPOSR)
    fake_install(settings, TOOL_MODELS["isnet-general-use"])
    st = Studio(
        settings=settings,
        engines={"triposr": Boom(), "trellis": TrellisEngine()},
        hardware=laptop(),
        background_engine=FakeBackgroundEngine(),
        cuda_runtime=False,
    )
    try:
        job = st.reconstruct(str(sample_image), preset="fast")
        for _ in range(300):
            s = st.job_status(job.id)
            if s["status"] in ("done", "failed"):
                break
            time.sleep(0.02)
        assert s["status"] == "failed"
        assert s["error"].startswith("The engine hit an internal error (TypeError")
        assert "report it" in s["error"]
        rep = st.report_bundle()
        assert job.id in rep["body"] and "TypeError" in rep["title"]
    finally:
        st.close()
