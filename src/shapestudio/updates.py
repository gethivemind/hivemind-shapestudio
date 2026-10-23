"""Update check: is there a newer release than the one running?

Consent first: the check is off until the user says yes (Home asks once; Settings has the
switch). When on, at most one request a day goes to GitHub's releases API for the latest
tag; nothing about the machine is sent beyond what any HTTPS request carries. The answer is
cached in ``<data>/update.json``. On the Windows installer build the matching ``Setup.exe``
can be downloaded, checked against its published SHA-256 and handed to Inno Setup, which
closes the running studio, installs and starts the new one. Store installs update through
the Store; pip installs are told the pipx command; macOS and Linux get the release link.
"""
# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from shapestudio import __version__
from shapestudio.config import Settings

REPO = "presidiumdev/presidium-shapestudio"
LATEST_URL = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_URL = f"https://github.com/{REPO}/releases"
CHECK_INTERVAL = 24 * 3600
TIMEOUT = 10.0
ProgressFn = Callable[[int, int], None]


def parse_version(v: str) -> tuple[int, ...]:
    """``v0.1.6`` → ``(0, 1, 6)``; anything unparsable sorts lowest."""
    m = re.match(r"v?(\d+)\.(\d+)\.(\d+)", v.strip())
    return tuple(int(x) for x in m.groups()) if m else (-1,)


def is_newer(latest: str, current: str = __version__) -> bool:
    return parse_version(latest) > parse_version(current)


def install_kind() -> str:
    """``store`` (MSIX), ``installer`` (Inno Setup on Windows), ``portable`` (zip or the
    macOS/Linux bundles) or ``python`` (pip/pipx)."""
    frozen = bool(getattr(sys, "frozen", False))
    if not frozen:
        return "python"
    exe = Path(sys.executable).resolve()
    if "WindowsApps" in exe.parts:
        return "store"
    if sys.platform == "win32" and (exe.parent / "unins000.exe").exists():
        return "installer"
    return "portable"


def enabled(settings: Settings) -> bool | None:
    """True / False once the user decided; None until asked."""
    v = settings.config().get("check_updates")
    return None if v is None else bool(v)


def set_enabled(settings: Settings, value: bool) -> None:
    cfg = settings.config()
    cfg["check_updates"] = bool(value)
    settings.write_config(cfg)


def _cache_path(settings: Settings) -> Path:
    return settings.data_dir / "update.json"


def cached(settings: Settings) -> dict | None:
    try:
        return json.loads(_cache_path(settings).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _summarise(release: dict) -> dict:
    tag = str(release.get("tag_name") or "")
    assets = {
        a.get("name", ""): a.get("browser_download_url", "") for a in release.get("assets", [])
    }
    setup = next((n for n in assets if n.endswith("-Setup.exe")), None)
    return {
        "latest": tag.lstrip("v"),
        "current": __version__,
        "available": is_newer(tag),
        "url": release.get("html_url") or RELEASES_URL,
        "published_at": release.get("published_at"),
        "notes": (release.get("body") or "")[:1500],
        "setup_exe": assets.get(setup) if setup else None,
        "setup_sha256_url": assets.get(f"{setup}.sha256") if setup else None,
        "setup_name": setup,
    }


def check(settings: Settings, force: bool = False, client=None) -> dict:
    """Return the latest release summary, from the cache when it is fresh. ``client`` is an
    httpx.Client (tests pass a mock transport)."""
    now = time.time()
    c = cached(settings)
    if c and not force and now - float(c.get("checked_at", 0)) < CHECK_INTERVAL:
        c["from_cache"] = True
        c["available"] = is_newer(c.get("latest", "0.0.0"))
        return c
    import httpx

    own = client is None
    client = client or httpx.Client(timeout=TIMEOUT, follow_redirects=True)
    try:
        r = client.get(LATEST_URL, headers={"Accept": "application/vnd.github+json"})
        r.raise_for_status()
        out = _summarise(r.json())
    except Exception as e:  # network down, rate limit, bad JSON: say so, keep the old answer
        out = dict(c or {"latest": None, "current": __version__, "available": False})
        out["error"] = f"{type(e).__name__}: {e}"
    finally:
        if own:
            client.close()
    out["checked_at"] = now
    out["install_kind"] = install_kind()
    out["from_cache"] = False
    with contextlib.suppress(OSError):
        _cache_path(settings).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def how_to_update(kind: str, info: dict) -> dict:
    """What this installation can do about a newer version, in words and as a flag."""
    url = info.get("url") or RELEASES_URL
    if kind == "store":
        return {
            "can_install": False,
            "text": "This copy came from the Microsoft Store, which updates it by itself "
            "(Store → Library → Get updates to hurry it along).",
            "url": "ms-windows-store://downloadsandupdates",
        }
    if kind == "installer" and info.get("setup_exe") and info.get("setup_sha256_url"):
        return {
            "can_install": True,
            "text": f"Download {info['setup_name']} (about 170 MB), verify its SHA-256 and run "
            "the installer: it closes the studio, installs over this copy and starts the new "
            "one. Models, meshes and settings stay where they are.",
            "url": url,
        }
    if kind == "python":
        return {
            "can_install": False,
            "text": "Installed with pip or pipx: run `pipx upgrade presidium-shapestudio` "
            "(or `pip install -U presidium-shapestudio`), then start the studio again.",
            "url": url,
        }
    return {
        "can_install": False,
        "text": "Download the new build from the release page and replace this one; the data "
        "folder is separate, so nothing is lost.",
        "url": url,
    }


def download_installer(settings: Settings, info: dict, progress: ProgressFn | None = None) -> Path:
    """Fetch the Setup.exe named in ``info`` into ``<data>/updates`` and verify it against the
    published ``.sha256`` file. Raises on any mismatch."""
    import httpx

    from shapestudio import download as dl

    if not info.get("setup_exe") or not info.get("setup_sha256_url"):
        raise RuntimeError("this release has no Windows installer asset")
    with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as c:
        r = c.get(info["setup_sha256_url"])
        r.raise_for_status()
        sha = r.text.split()[0].strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise RuntimeError("the published .sha256 file is not a SHA-256")
    folder = settings.data_dir / "updates"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / info["setup_name"]
    if target.exists() and dl.sha256_of(target) == sha:
        return target
    return dl.download(info["setup_exe"], target, sha, None, progress=progress)


def launch_installer(path: Path) -> None:
    """Start Inno Setup silently; it closes the running studio (CloseApplications=yes) and
    starts the new one. The caller quits right after."""
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    subprocess.Popen(
        [str(path), "/SILENT", "/CLOSEAPPLICATIONS", "/RESTARTAPPLICATIONS", "/NORESTART"],
        close_fds=True,
        creationflags=flags,
    )
