"""The local web UI's JSON API, through Starlette's test client."""

from __future__ import annotations

import time

from starlette.testclient import TestClient

from shapestudio.service import Studio
from shapestudio.ui import build_app


def _wait(client: TestClient, job_id: str) -> dict:
    for _ in range(300):
        st = client.get(f"/api/jobs/{job_id}").json()
        if st["status"] in ("done", "failed", "cancelled"):
            return st
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_page_plan_upload_reconstruct_files_and_jobs(studio: Studio, sample_image):
    client = TestClient(build_app(studio))
    html = client.get("/").text
    assert "Shape Studio" in html and "model-viewer" in html
    assert client.get("/manifest.webmanifest").json()["name"] == "Shape Studio by Hivemind"
    assert client.get("/icon-64.png").content.startswith(b"\x89PNG")
    assert client.get("/api/about").json()["credit"].startswith("made with")
    caps = client.get("/api/capabilities").json()
    assert caps["policy"]["default_preset"] == "best"
    plan = client.get("/api/plan", params={"preset": "fast"}).json()
    assert plan["resolution"] == 128 and plan["load"]["label"]
    refused = client.get("/api/plan", params={"tier": "best"})
    assert refused.status_code == 400 and "NVIDIA" in refused.json()["error"]

    with sample_image.open("rb") as fh:
        r = client.post(
            "/api/reconstruct",
            files={"file": ("teapot.png", fh, "image/png")},
            data={"preset": "fast", "note": "first"},
        )
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    st = _wait(client, job_id)
    assert st["status"] == "done" and st["resolution"] == 128 and st["note"] == "first"
    glb = client.get(f"/api/files/{job_id}/glb")
    assert glb.status_code == 200 and glb.content[:4] == b"glTF"
    dl = client.get(f"/api/files/{job_id}/obj", params={"download": "1"})
    assert "attachment" in dl.headers["content-disposition"] and dl.text.startswith(
        "# Shape Studio"
    )
    assert client.get(f"/api/files/{job_id}/preview").status_code == 200
    assert client.get(f"/api/files/{job_id}/nope").status_code == 404
    listing = client.get("/api/jobs").json()
    assert listing["jobs"][0]["id"] == job_id and listing["worker_in_this_process"] is True
    # JSON body with a path works too
    r = client.post("/api/reconstruct", json={"image": str(sample_image), "preset": "balanced"})
    assert r.status_code == 200 and _wait(client, r.json()["id"])["resolution"] == 192
    assert client.post("/api/reconstruct", json={"image": "/nowhere.png"}).status_code == 400


def test_gallery_management_projects_export_and_trash(studio: Studio, sample_image):
    client = TestClient(build_app(studio))
    job_id = client.post(
        "/api/reconstruct", json={"image": str(sample_image), "preset": "fast"}
    ).json()["id"]
    _wait(client, job_id)
    assert client.post(f"/api/jobs/{job_id}/favorite", json={"favorite": True}).json()["favorite"]
    assert client.post(f"/api/jobs/{job_id}/note", json={"note": "keep"}).json()["note"] == "keep"
    prj = client.post("/api/projects", json={"name": "Chairs"}).json()
    assert client.get("/api/projects").json()["current"] == prj["id"]
    client.post(f"/api/jobs/{job_id}/project", json={"project": prj["id"]})
    z = client.get(f"/api/projects/{prj['id']}/export")
    assert z.status_code == 200 and z.content[:2] == b"PK"
    z = client.post("/api/export", json={"ids": [job_id], "title": "picked"})
    assert z.status_code == 200 and "picked.zip" in z.headers["content-disposition"]
    assert client.delete(f"/api/jobs/{job_id}").status_code == 200
    assert client.get("/api/trash").json()["jobs"][0]["id"] == job_id
    assert client.post(f"/api/jobs/{job_id}/restore").status_code == 200
    assert client.get("/api/jobs").json()["jobs"][0]["id"] == job_id
    client.delete(f"/api/jobs/{job_id}")
    assert client.post("/api/trash/empty").json()["emptied"] == 1


def test_components_downloads_runtime_benchmark_tasks_and_config(studio: Studio, monkeypatch):
    client = TestClient(build_app(studio))
    comps = client.get("/api/components").json()["components"]
    kinds = {c["kind"] for c in comps}
    assert kinds == {"model", "tool", "runtime"}
    trellis = next(c for c in comps if c["name"] == "trellis-image-large")
    assert not trellis["installed"] and trellis["size"] > 3 * 2**30
    info = client.get("/api/models/download", params={"name": "trellis-image-large"}).json()
    assert info["total_gib"] > 3 and not info["already_installed"]
    assert (
        client.post("/api/models/download", json={"name": "u2net"}).status_code == 400
    )  # no consent

    def fake_install(name, consent, progress=None):
        from shapestudio.registry import TOOL_MODELS

        t = TOOL_MODELS[name]
        if progress:
            progress(t.file.filename, t.file.size, t.file.size)
        return True

    monkeypatch.setattr(studio, "install", fake_install)
    t = client.post("/api/models/download", json={"name": "u2net", "consent": True}).json()
    assert t["kind"] == "download"
    for _ in range(100):
        tasks = client.get("/api/tasks").json()
        if tasks["download:u2net"]["status"] != "running":
            break
        time.sleep(0.02)
    assert tasks["download:u2net"]["status"] == "done"

    rt = client.get("/api/runtime").json()
    assert "torch" in rt and "cuda_install" in rt
    assert client.post("/api/runtime/install", json={}).status_code == 400

    b = client.post("/api/benchmark", json={"preset": "fast"}).json()
    assert b["kind"] == "benchmark"
    for _ in range(300):
        tasks = client.get("/api/tasks").json()
        if tasks["benchmark"]["status"] != "running":
            break
        time.sleep(0.05)
    assert tasks["benchmark"]["status"] == "done"
    assert tasks["benchmark"]["result"]["profile"]["resolution"] == 128

    cfg = client.get("/api/config").json()
    assert cfg["allow_over_budget"] is False and cfg["device"] == "auto"
    cfg = client.post(
        "/api/config", json={"device": "cpu", "default_threads": 4, "max_memory_gib": 8}
    ).json()
    assert (
        cfg["device"] == "cpu" and cfg["default_threads"] == 4 and cfg["memory_budget_gib"] == 8.0
    )
    assert client.post("/api/config", json={"device": "tpu"}).status_code == 400
    m = client.get("/api/memory").json()
    assert m["total_gib"] > 0 and isinstance(m["top"], list)
    assert client.post("/api/reveal", json={"path": "/etc"}).status_code == 400
