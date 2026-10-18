"""The catalogue's pins and the planner's tier choices on three machine classes."""

from __future__ import annotations

import re

import pytest

from conftest import TRELLIS, TRIPOSR, fake_install, laptop, make_studio, small_gpu_box, workstation
from shapestudio import registry
from shapestudio.planner import PlanError
from shapestudio.service import StudioError


def test_every_pinned_file_has_a_sha256_size_licence_and_source():
    files = [f for m in registry.MODELS.values() for f in m.files]
    files += [t.file for t in registry.TOOL_MODELS.values()]
    files += [w.file for w in registry.RUNTIME_WHEELS]
    assert len(files) > 20
    for f in files:
        assert re.fullmatch(r"[0-9a-f]{64}", f.sha256), f.filename
        assert f.size > 0 and f.licence and f.source and f.url.startswith("https://"), f.filename
    assert TRIPOSR.download_bytes == 1677246742 + 987
    assert round(TRELLIS.download_bytes / registry.GIB, 2) == pytest.approx(3.07, abs=0.02)


def test_catalogue_is_permissive_only_and_names_the_exclusions():
    for m in registry.MODELS.values():
        assert m.licence in ("MIT", "Apache-2.0", "BSD-3-Clause")
    names = " ".join(m for m, _ in registry.EXCLUDED_BY_POLICY)
    assert "Hunyuan3D" in names and "Stable Fast 3D" in names
    assert all(reason for _, reason in registry.EXCLUDED_BY_POLICY)
    assert registry.model_for_tier("best") is TRELLIS
    assert registry.model_for_tier("standard") is TRIPOSR


def test_runtime_wheels_cover_the_packaged_platforms():
    assert registry.find_runtime_wheel("cpu", "win32", "AMD64").file.size == 123996008
    assert registry.find_runtime_wheel("cuda", "win32", "AMD64").self_contained is True
    assert registry.find_runtime_wheel("cuda", "linux", "x86_64").self_contained is False
    assert registry.find_runtime_wheel("cuda", "darwin", "arm64") is None


def test_laptop_without_gpu_gets_triposr_on_cpu_and_an_honest_fit_note(settings):
    fake_install(settings, TRIPOSR)
    st = make_studio(settings, laptop())
    try:
        p = st.plan()
        assert p.model is TRIPOSR and p.device == "cpu" and p.resolution == 256
        assert p.fit_note and "TRELLIS" in p.fit_note and "NVIDIA GPU" in p.fit_note
        assert "not a speed choice" in p.fit_note
        assert not p.estimate.calibrated and "not measured yet" in p.estimate.basis
        assert p.load.label in ("comfortable", "tight")
        with pytest.raises(StudioError, match="NVIDIA GPU"):
            st.plan("best")
        with pytest.raises(StudioError, match="NVIDIA GPU"):
            st.plan("trellis-image-large")
    finally:
        st.close()


def test_presets_change_only_the_grid_and_fast_is_explicit(settings):
    fake_install(settings, TRIPOSR)
    st = make_studio(settings, laptop())
    try:
        assert st.plan(preset="fast").resolution == 128
        assert st.plan(preset="balanced").resolution == 192
        assert st.plan(resolution=320).resolution == 320
        with pytest.raises(StudioError, match="resolution"):
            st.plan(resolution=1024)
        with pytest.raises(StudioError, match="preset"):
            st.plan(preset="ultra")
        table = st.planner.preset_table(TRIPOSR)
        assert [p["preset"] for p in table] == ["best", "balanced", "fast"]
        assert all("not measured yet" in p["expected_time"]["basis"] for p in table)
    finally:
        st.close()


def test_workstation_with_big_gpu_offers_best_tier_when_downloaded(settings):
    fake_install(settings, TRIPOSR)
    st = make_studio(settings, workstation(), cuda_runtime=True)
    try:
        p = st.plan()
        # TRELLIS fits but is not downloaded: the plan says so instead of silently choosing
        assert p.model is TRIPOSR and p.device == "cuda"
        assert p.fit_note and "not downloaded" in p.fit_note
        fake_install(settings, TRELLIS)
        p = st.plan()
        assert p.model is TRELLIS and p.device == "cuda" and p.fit_note is None
    finally:
        st.close()


def test_small_gpu_refuses_trellis_with_numbers_and_runs_triposr_on_the_gpu(settings):
    fake_install(settings, TRIPOSR)
    fake_install(settings, TRELLIS)
    st = make_studio(settings, small_gpu_box(), cuda_runtime=True)
    try:
        p = st.plan()
        assert p.model is TRIPOSR and p.device == "cuda"
        assert "16 GiB" in p.fit_note and "12 GiB" in p.fit_note
        with pytest.raises(StudioError, match="at least 16 GiB"):
            st.plan("best")
    finally:
        st.close()


def test_gpu_present_but_cuda_runtime_missing_runs_cpu_and_says_how_to_fix(settings):
    fake_install(settings, TRIPOSR)
    st = make_studio(settings, small_gpu_box(), cuda_runtime=False)
    try:
        p = st.plan()
        assert p.device == "cpu"
        assert any("install-runtime --cuda" in w for w in p.warnings)
    finally:
        st.close()


def test_over_budget_is_refused_unless_unlocked(settings):
    fake_install(settings, TRIPOSR)
    tiny = laptop(ram_gib=4)
    st = make_studio(settings, tiny)
    try:
        with pytest.raises(StudioError, match="locked"):
            st.plan()
        st.set_config({"allow_over_budget": True})
        p = st.plan()
        assert p.over_budget and any("over budget" in w for w in p.warnings)
    finally:
        st.close()


def test_plan_error_lists_known_names(settings):
    fake_install(settings, TRIPOSR)
    st = make_studio(settings, laptop())
    try:
        with pytest.raises((StudioError, PlanError), match="known"):
            st.plan("nope")
    finally:
        st.close()
