"""The update check: consent first, one cached request a day, honest about failures, and
the right advice per installation kind."""

from __future__ import annotations

import json

import httpx
from starlette.testclient import TestClient

from shapestudio import __version__, updates
from shapestudio.ui import build_app

RELEASE = {
    "tag_name": "v9.9.9",
    "html_url": "https://github.com/presidiumdev/presidium-shapestudio/releases/tag/v9.9.9",
    "published_at": "2026-12-01T00:00:00Z",
    "body": "notes",
    "assets": [
        {"name": "ShapeStudio-9.9.9-Setup.exe", "browser_download_url": "https://x/Setup.exe"},
        {
            "name": "ShapeStudio-9.9.9-Setup.exe.sha256",
            "browser_download_url": "https://x/s.sha256",
        },
    ],
}


def _client(payload, status=200):
    def handler(request):
        return httpx.Response(status, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_version_compare_and_install_kind():
    assert updates.parse_version("v0.1.6") == (0, 1, 6)
    assert updates.is_newer("0.1.7", "0.1.6") and not updates.is_newer("0.1.6", "0.1.6")
    assert not updates.is_newer("garbage", "0.1.6")
    assert updates.install_kind() == "python"  # the test process is not frozen


def test_check_caches_for_a_day_and_reports_failures(settings):
    assert updates.enabled(settings) is None
    info = updates.check(settings, client=_client(RELEASE))
    assert info["available"] and info["latest"] == "9.9.9" and info["current"] == __version__
    assert info["setup_name"] == "ShapeStudio-9.9.9-Setup.exe" and info["setup_sha256_url"]
    again = updates.check(settings, client=_client({}, 500))  # would fail: served from cache
    assert again["from_cache"] and again["latest"] == "9.9.9"
    failed = updates.check(settings, force=True, client=_client({}, 500))
    assert "error" in failed and failed["latest"] == "9.9.9"  # keeps the last good answer
    how = updates.how_to_update("python", info)
    assert not how["can_install"] and "pipx upgrade" in how["text"]
    assert updates.how_to_update("installer", info)["can_install"]
    assert "Store" in updates.how_to_update("store", info)["text"]
    updates.set_enabled(settings, True)
    assert updates.enabled(settings) is True
    assert json.loads(settings.config_file.read_text())["check_updates"] is True


def test_api_asks_before_any_network_and_installs_only_where_it_can(studio, monkeypatch):
    client = TestClient(build_app(studio))
    calls = []

    def fake_check(settings, force=False, client=None):
        calls.append(force)
        return {**updates._summarise(RELEASE), "checked_at": 1.0}

    monkeypatch.setattr(updates, "check", fake_check)
    st = client.get("/api/update").json()
    assert st["enabled"] is None and st["latest"] is None and calls == []  # no consent, no call
    st = client.post("/api/update/consent", json={"enabled": True}).json()
    assert st["enabled"] is True and st["available"] and st["latest"] == "9.9.9"
    assert st["how"]["can_install"] is False  # a Python install: pipx advice, no installer
    r = client.post("/api/update/install", json={"consent": True})
    assert r.status_code == 400 and "pipx" in r.json()["error"]
    assert client.post("/api/update/install", json={}).status_code == 400
    cfg = client.get("/api/config").json()
    assert cfg["check_updates"] is True
    st = client.post("/api/update/consent", json={"enabled": False}).json()
    assert st["enabled"] is False
    html = client.get("/").text
    assert "Check for updates once a day" in html and "/api/update" in html
