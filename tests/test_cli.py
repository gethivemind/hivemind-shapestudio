"""The command line, run in-process against a temporary data dir with the fake engine."""

from __future__ import annotations

import json

import pytest

from conftest import TOOL_MODELS, TRIPOSR, fake_install, laptop, make_studio
from shapestudio import cli


@pytest.fixture
def cli_env(settings, monkeypatch):
    monkeypatch.setenv("SHAPESTUDIO_DATA_DIR", str(settings.data_dir))
    fake_install(settings, TRIPOSR)
    fake_install(settings, TOOL_MODELS["isnet-general-use"])
    made = []

    def fake_studio(start_worker: bool):
        st = make_studio(settings, laptop())
        if not start_worker:
            st.close()
        made.append(st)
        return st

    monkeypatch.setattr(cli, "_studio", fake_studio)
    yield settings
    for st in made:
        st.close()


def test_doctor_prints_hardware_runtime_tiers_and_policy(cli_env, capsys):
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "Hivemind Shape Studio" in out and "i5-1235U" in out
    assert "PyTorch" in out and "TripoSR [standard]" in out and "TRELLIS image-large [best]" in out
    assert "does NOT fit" in out and "not measured yet" in out
    assert "Excluded by policy" in out and "Hunyuan3D" in out
    assert cli.main(["doctor", "--json"]) == 0
    caps = json.loads(capsys.readouterr().out)
    assert caps["policy"]["default_preset"] == "best"


def test_plan_and_reconstruct_and_jobs(cli_env, sample_image, capsys, tmp_path):
    assert cli.main(["plan", "--preset", "fast"]) == 0
    out = capsys.readouterr().out
    assert "128³ grid" in out and "Load [" in out and "not measured yet" in out
    assert cli.main(["plan", "--tier", "best"]) == 1
    assert "refused" in capsys.readouterr().out
    out_dir = tmp_path / "exported"
    assert (
        cli.main(["reconstruct", str(sample_image), "--preset", "fast", "--out", str(out_dir)]) == 0
    )
    out = capsys.readouterr().out
    assert "GLB:" in out and "8 vertices, 12 faces" in out and "Copied to" in out
    assert any(p.suffix == ".glb" for p in out_dir.iterdir())
    assert cli.main(["jobs"]) == 0
    assert "done" in capsys.readouterr().out
    assert cli.main(["jobs", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["status"] == "done"


def test_download_model_without_tty_needs_yes(cli_env, capsys):
    assert cli.main(["download-model", "nope"]) == 2
    assert cli.main(["download-model", "triposr"]) == 0
    assert "already installed" in capsys.readouterr().out
    # u2net is not installed; without a terminal and without --yes nothing is fetched
    assert cli.main(["download-model", "u2net"]) == 1
    out = capsys.readouterr().out
    assert "U²-Net" in out and "sha256" in out and "not downloaded" in out


def test_install_runtime_reports(cli_env, capsys):
    assert cli.main(["install-runtime"]) == 0
    assert "PyTorch" in capsys.readouterr().out
