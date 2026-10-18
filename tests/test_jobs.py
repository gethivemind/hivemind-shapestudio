"""Queue behaviour with the fake engine: results, persistence, cancel, calibration,
gallery management, projects and exports."""

from __future__ import annotations

import base64
import io
import json
import time
import zipfile

import pytest

from conftest import TOOL_MODELS, TRIPOSR, fake_install, laptop, make_studio
from shapestudio.engine.meshio import read_glb_json
from shapestudio.service import Studio, StudioError


def wait_done(studio: Studio, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        st = studio.job_status(job_id)
        if st["status"] in ("done", "failed", "cancelled"):
            return st
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_reconstruct_returns_at_once_and_produces_glb_obj_and_preview(studio, sample_image):
    job = studio.reconstruct(str(sample_image))
    assert job.status == "queued" and job.estimate_seconds > 0
    assert job.preset == "best" and job.resolution == 256 and job.device == "cpu"
    assert "not measured yet" in job.estimate_basis
    st = wait_done(studio, job.id)
    assert st["status"] == "done" and st["progress"] == 1.0
    assert st["vertices"] == 8 and st["faces"] == 12
    r = studio.job_result(job.id)
    assert (
        r["ready"] and r["path"].endswith(f"{job.id}.glb") and r["mime_type"] == "model/gltf-binary"
    )
    assert read_glb_json(r["path"])["asset"]["version"] == "2.0"
    o = studio.job_result(job.id, fmt="obj")
    assert o["path"].endswith(".obj")
    b = studio.job_result(job.id, as_base64=True)
    assert base64.b64decode(b["base64"])[:4] == b"glTF"
    assert st["preview_path"] and st["cutout_path"]
    # the source picture was copied into the uploads folder, the job record persisted
    assert studio.settings.uploads_dir in __import__("pathlib").Path(job.input_path).parents
    assert (studio.settings.jobs_dir / f"{job.id}.json").exists()
    # the first real run calibrates the estimate: later plans are "measured"
    p = studio.plan()
    assert p.estimate.calibrated and "measured" in p.estimate.basis


def test_result_before_done_does_not_block_and_eta_is_reported(settings, sample_image):
    fake_install(settings, TRIPOSR)
    fake_install(settings, TOOL_MODELS["isnet-general-use"])
    studio = make_studio(settings, laptop(), phase_delay=0.15)
    try:
        job = studio.reconstruct(str(sample_image))
        r = studio.job_result(job.id)
        assert r["ready"] is False and r["status"] in ("queued", "running")
        st = studio.job_status(job.id)
        assert st["eta_seconds"] is not None and st["summary"]
        wait_done(studio, job.id)
    finally:
        studio.close()


def test_jobs_run_in_order_and_cancel_works(settings, sample_image):
    fake_install(settings, TRIPOSR)
    fake_install(settings, TOOL_MODELS["isnet-general-use"])
    studio = make_studio(settings, laptop(), phase_delay=0.1)
    try:
        a = studio.reconstruct(str(sample_image))
        b = studio.reconstruct(str(sample_image))
        c = studio.reconstruct(str(sample_image))
        assert studio.job_status(c.id)["queue_position"] in (2, 3)
        assert studio.cancel(c.id) is True
        assert studio.job_status(c.id)["status"] == "cancelled"
        wait_done(studio, a.id)
        wait_done(studio, b.id)
        # cancelling the running one stops it within a phase
        d = studio.reconstruct(str(sample_image))
        for _ in range(100):
            if studio.job_status(d.id)["status"] == "running":
                break
            time.sleep(0.02)
        studio.cancel(d.id)
        st = wait_done(studio, d.id)
        assert st["status"] == "cancelled"
        assert studio.cancel(d.id) is False
    finally:
        studio.close()


def test_missing_model_or_background_model_is_explained(settings, sample_image):
    studio = make_studio(settings, laptop())
    try:
        with pytest.raises(StudioError, match="download-model triposr"):
            studio.reconstruct(str(sample_image))
        fake_install(settings, TRIPOSR)
        with pytest.raises(StudioError, match="isnet-general-use"):
            studio.reconstruct(str(sample_image))
        # a picture with a transparent background needs no segmentation model
        from PIL import Image

        rgba = settings.data_dir / "cut.png"
        im = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        im.putpixel((16, 16), (200, 10, 10, 255))
        im.save(rgba)
        job = studio.reconstruct(str(rgba))
        assert wait_done(studio, job.id)["status"] == "done"
        with pytest.raises(StudioError, match="not found"):
            studio.reconstruct("/nowhere/x.png")
    finally:
        studio.close()


def test_engine_failure_is_recorded_not_raised(settings, sample_image):
    fake_install(settings, TRIPOSR)
    fake_install(settings, TOOL_MODELS["isnet-general-use"])
    studio = make_studio(settings, laptop(), fail=True)
    try:
        job = studio.reconstruct(str(sample_image))
        st = wait_done(studio, job.id)
        assert st["status"] == "failed" and "fake engine failure" in st["error"]
    finally:
        studio.close()


def test_trash_restore_favourite_note_and_projects(studio, sample_image):
    job = studio.reconstruct(str(sample_image))
    wait_done(studio, job.id)
    assert studio.set_favorite(job.id, True)["favorite"] is True
    assert studio.set_note(job.id, "a teapot")["note"] == "a teapot"
    glb = studio.job_status(job.id)["glb_path"]
    studio.delete_job(job.id)
    assert job.id not in {r["id"] for r in studio.list_jobs()}
    assert studio.trash()[0]["id"] == job.id
    assert not __import__("pathlib").Path(glb).exists()
    studio.restore_job(job.id)
    assert __import__("pathlib").Path(studio.job_status(job.id)["glb_path"]).exists()
    prj = studio.add_project("Chairs", "test series")
    assert studio.current_project() == prj["id"]
    studio.set_job_project(job.id, prj["id"])
    assert studio.projects()[0]["count"] == 1
    name, data = studio.export_project(prj["id"])
    assert name == "Chairs.zip"
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = z.namelist()
        assert any(n.endswith(".glb") for n in names) and "project.json" in names
        meta = json.loads(z.read("project.json"))
        assert meta["meshes"][0]["job"] == job.id and meta["meshes"][0]["note"] == "a teapot"
    name, data = studio.export_jobs([job.id], "picked")
    assert name == "picked.zip"
    studio.remove_project(prj["id"])
    assert studio.job_status(job.id)["project"] is None
    studio.delete_job(job.id)
    assert studio.empty_trash() == 1
    assert studio.jobs.get(job.id) is None


def test_second_process_submits_and_the_worker_picks_it_up(settings, sample_image):
    fake_install(settings, TRIPOSR)
    fake_install(settings, TOOL_MODELS["isnet-general-use"])
    worker = make_studio(settings, laptop())
    worker.worker.poll_seconds = 0.1
    submitter = make_studio(settings, laptop())
    try:
        assert worker.worker_active and not submitter.worker_active
        job = submitter.reconstruct(str(sample_image))
        st = wait_done(submitter, job.id)
        assert st["status"] == "done"
        caps = submitter.capabilities()
        assert caps["queue"]["worker_in_this_process"] is False
        assert caps["queue"]["worker_pid"] == __import__("os").getpid()
    finally:
        submitter.close()
        worker.close()


def test_pause_holds_the_queue(settings, sample_image):
    fake_install(settings, TRIPOSR)
    fake_install(settings, TOOL_MODELS["isnet-general-use"])
    studio = make_studio(settings, laptop())
    studio.worker.poll_seconds = 0.1
    try:
        studio.set_config({"paused": True})
        job = studio.reconstruct(str(sample_image))
        time.sleep(0.5)
        st = studio.job_status(job.id)
        assert st["status"] == "queued" and "paused" in st["summary"]
        studio.set_config({"paused": False})
        assert wait_done(studio, job.id)["status"] == "done"
    finally:
        studio.close()


def test_benchmark_writes_a_measured_profile(studio):
    p = studio.benchmark("fast", wait=0.05)
    assert p["preset"] == "fast" and p["resolution"] == 128 and p["wall_seconds"] is not None
    assert set(p["expected_seconds_per_preset"]) == {"best", "balanced", "fast"}
    assert studio.settings.profile_file.exists()
    assert studio.capabilities()["profile"]["job_id"] == p["job_id"]
