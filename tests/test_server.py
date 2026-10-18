"""The MCP surface: tool names, annotations, descriptions written for an AI caller, and
round trips through the server's own call_tool."""

from __future__ import annotations

import json
import time

from shapestudio.server import build_server
from shapestudio.service import Studio

EXPECTED_TOOLS = {
    "capabilities",
    "plan",
    "reconstruct",
    "job_status",
    "job_result",
    "jobs",
    "cancel_job",
    "download_model",
}


def _payload(result):
    if isinstance(result, tuple):
        content, structured = result
        if isinstance(structured, dict):
            return structured.get("result", structured)
        result = content
    if isinstance(result, dict):
        return result.get("result", result)
    for attr in ("structured_content", "structuredContent"):
        sc = getattr(result, attr, None)
        if sc:
            return sc.get("result", sc)
    content = getattr(result, "content", result)
    if isinstance(content, list) and content and hasattr(content[0], "text"):
        return json.loads(content[0].text)
    if isinstance(result, list) and result and hasattr(result[0], "text"):
        return json.loads(result[0].text)
    raise AssertionError(f"unexpected call_tool result: {result!r}")


async def test_tools_are_listed_with_annotations_and_ai_facing_descriptions(studio: Studio):
    server = build_server(studio)
    tools = await server.list_tools()
    by_name = {t.name: t for t in tools}
    assert set(by_name) == EXPECTED_TOOLS
    for t in tools:
        assert t.annotations is not None, t.name

    def hint(tool, name):  # mcp 1.x exposes camelCase, mcp 2.x snake_case attributes
        a = tool.annotations
        snake = "".join("_" + c.lower() if c.isupper() else c for c in name)
        return getattr(a, snake, None) if hasattr(a, snake) else getattr(a, name)

    assert hint(by_name["download_model"], "openWorldHint") is True
    assert hint(by_name["capabilities"], "readOnlyHint") is True
    assert hint(by_name["cancel_job"], "destructiveHint") is True
    rec = by_name["reconstruct"].description
    assert "best" in rec and "fast" in rec and "job id" in rec.lower() and "poll" in rec.lower()
    assert "consent" in by_name["download_model"].description
    assert "hardware" in by_name["capabilities"].description.lower()
    schema = (
        getattr(by_name["reconstruct"], "input_schema", None) or by_name["reconstruct"].inputSchema
    )
    assert schema["properties"]["preset"]["default"] == "best"


async def test_reconstruct_status_result_round_trip(studio: Studio, sample_image):
    server = build_server(studio)
    r = _payload(await server.call_tool("reconstruct", {"image": str(sample_image)}))
    assert (
        r["status"] in ("queued", "running") and r["preset"] == "best" and r["tier"] == "standard"
    )
    assert r["estimate_seconds"] > 0 and "not measured yet" in r["estimate_basis"]
    job_id = r["job_id"]
    for _ in range(300):
        st = _payload(await server.call_tool("job_status", {"job_id": job_id}))
        if st["status"] == "done":
            break
        time.sleep(0.02)
    assert st["status"] == "done" and st["glb_path"].endswith(".glb")
    res = _payload(await server.call_tool("job_result", {"job_id": job_id, "as_base64": True}))
    assert res["ready"] and res["mime_type"] == "model/gltf-binary" and res["base64"]
    res = _payload(await server.call_tool("job_result", {"job_id": job_id, "format": "obj"}))
    assert res["path"].endswith(".obj")
    listing = _payload(await server.call_tool("jobs", {}))
    assert listing["jobs"][0]["id"] == job_id
    assert _payload(await server.call_tool("cancel_job", {"job_id": job_id}))["cancelled"] is False


async def test_errors_come_back_as_error_field_not_exceptions(studio: Studio, sample_image):
    server = build_server(studio)
    r = _payload(
        await server.call_tool("reconstruct", {"image": str(sample_image), "preset": "ultra"})
    )
    assert "error" in r and "preset" in r["error"]
    r = _payload(
        await server.call_tool("reconstruct", {"image": str(sample_image), "tier": "best"})
    )
    assert "error" in r and "NVIDIA GPU" in r["error"]
    r = _payload(await server.call_tool("job_status", {"job_id": "job_nope"}))
    assert "error" in r


async def test_capabilities_state_policy_hardware_tiers_runtime_and_licences(studio: Studio):
    server = build_server(studio)
    caps = _payload(await server.call_tool("capabilities", {}))
    assert (
        caps["policy"]["default_preset"] == "best" and caps["policy"]["default_model"] == "triposr"
    )
    assert caps["hardware"]["physical_cores"] == 10
    assert "torch" in caps["runtime"] and "cuda_install" in caps["runtime"]
    tiers = {m["tier"]: m for m in caps["models"]}
    assert tiers["standard"]["installed"] and tiers["standard"]["fits_this_machine"]
    assert not tiers["best"]["fits_this_machine"] and "NVIDIA GPU" in tiers["best"]["fit_note"]
    assert "not run yet" in tiers["best"]["verified"]
    assert tiers["best"]["engine_status"]["integrated"] is False
    assert any(li["licence"] == "MIT" for li in caps["licences"])
    assert any("Hunyuan3D" in e["model"] for e in caps["excluded_by_policy"])
    best = next(p for p in tiers["standard"]["presets"] if p["preset"] == "best")
    assert best["resolution"] == 256 and best["expected_time"]["human"]


async def test_download_model_requires_consent_and_states_sizes(studio: Studio):
    server = build_server(studio)
    r = _payload(await server.call_tool("download_model", {"name": "trellis-image-large"}))
    assert r["downloaded"] is False and r["total_gib"] > 3 and "consent" in r["message"]
    assert all(len(f["sha256"]) == 64 for f in r["files"])
    r = _payload(await server.call_tool("download_model", {"name": "nope"}))
    assert "error" in r
    r = _payload(await server.call_tool("download_model", {"name": "triposr"}))
    assert r["already_installed"] is True
    r = _payload(await server.call_tool("download_model", {"name": "u2net"}))
    assert r["kind"] == "tool" and r["downloaded"] is False


async def test_plan_is_a_dry_run(studio: Studio):
    server = build_server(studio)
    before = len(studio.list_jobs())
    p = _payload(await server.call_tool("plan", {"preset": "fast"}))
    assert p["resolution"] == 128 and p["device"] == "cpu" and p["load"]["label"]
    assert len(studio.list_jobs()) == before
