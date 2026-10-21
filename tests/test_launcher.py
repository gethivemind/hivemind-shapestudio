"""The desktop launcher's in-process server: it must build the web app and listen, which is
exactly what the packaged app does first. Run against a temporary data dir and a free port."""

from __future__ import annotations

import importlib.util
import socket
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def _load_launcher(monkeypatch, data_dir: Path, port: int):
    monkeypatch.setenv("SHAPESTUDIO_DATA_DIR", str(data_dir))
    monkeypatch.setenv("SHAPESTUDIO_PORT", str(port))
    spec = importlib.util.spec_from_file_location(
        "shapestudio_launcher", ROOT / "packaging" / "launcher.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_launcher_server_starts_and_serves_the_studio(tmp_path, monkeypatch):
    port = _free_port()
    launcher = _load_launcher(monkeypatch, tmp_path / "data", port)
    server = launcher.Server()
    try:
        server.start()  # the line the packaged app crashed on in 0.1.x
        assert launcher.port_open(port)
        caps = httpx.get(f"http://127.0.0.1:{port}/api/capabilities", timeout=10).json()
        assert caps["policy"]["default_preset"] == "best"
        assert "Shape Studio" in httpx.get(f"http://127.0.0.1:{port}/", timeout=10).text
        assert server.rendering() is False
        assert launcher.TITLE in server.progress_text()
    finally:
        server.stop()
