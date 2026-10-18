"""Resumable, hash-verified downloads. The only module that talks to the network.

A download is written to ``<target>.part`` and resumed with an HTTP Range request when a
partial file exists. On completion the SHA-256 is checked against the pinned value before the
file is moved into place; a mismatch deletes the file and raises. Nothing here runs unless the
caller has already obtained consent (``--yes`` on the CLI or ``consent=True`` on the tool).
"""
# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import httpx

try:  # Trust the operating system's certificate store (corporate TLS proxies, Windows roots).
    import truststore

    truststore.inject_into_ssl()
except ImportError:  # pragma: no cover
    pass

CHUNK = 1024 * 1024
MAX_ATTEMPTS = 8  # transient network errors resume from the .part file
RETRY_WAIT_SECONDS = 5.0
ProgressFn = Callable[[int, int], None]  # (bytes_done, bytes_total)


class DownloadError(RuntimeError):
    pass


class HashMismatchError(DownloadError):
    pass


def sha256_of(path: Path, progress: ProgressFn | None = None) -> str:
    h = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    with path.open("rb") as f:
        while chunk := f.read(CHUNK * 8):
            h.update(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)
    return h.hexdigest()


def download(
    url: str,
    target: Path,
    sha256: str,
    size: int | None = None,
    progress: ProgressFn | None = None,
    client: httpx.Client | None = None,
    attempts: int = MAX_ATTEMPTS,
) -> Path:
    """Fetch ``url`` into ``target``, resuming a ``.part`` file if present, verifying SHA-256.

    Transient network errors (timeouts, dropped connections) are retried up to ``attempts``
    times, each time resuming from the bytes already on disk. Returns ``target``. If
    ``target`` already exists with the right hash it is returned untouched (the hash is
    recomputed, which is slow for 10 GB files; callers cache the result in the manifest).
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_of(target, progress) == sha256:
            return target
        target.unlink()

    part = target.with_suffix(target.suffix + ".part")
    own_client = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0))
    try:
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                _fetch(client, url, part, size, progress)
                break
            except (httpx.TransportError, httpx.RemoteProtocolError) as e:
                last_error = e
                if attempt == attempts:
                    raise DownloadError(
                        f"{target.name}: gave up after {attempts} attempts: {e}. The partial "
                        "file is kept; running the download again resumes it."
                    ) from e
                time.sleep(RETRY_WAIT_SECONDS)
        else:  # pragma: no cover
            raise DownloadError(str(last_error))
    finally:
        if own_client:
            client.close()

    actual = sha256_of(part)
    if actual != sha256:
        part.unlink(missing_ok=True)
        raise HashMismatchError(
            f"{target.name}: SHA-256 {actual} does not match pinned {sha256}; "
            "the partial file was deleted. Retry, and if it repeats, report it: the upstream "
            "file may have changed."
        )
    shutil.move(str(part), str(target))
    return target


def _fetch(
    client: httpx.Client, url: str, part: Path, size: int | None, progress: ProgressFn | None
) -> None:
    """One attempt: resume ``part`` with a Range request and stream to its end."""
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with client.stream("GET", url, headers=headers) as r:
        if have and r.status_code == 200:
            have = 0  # server ignored the range: start over
        elif have and r.status_code == 416:
            return  # already complete; the hash check decides
        elif have and r.status_code != 206:
            raise DownloadError(f"resume refused: HTTP {r.status_code} for {url}")
        elif not have and r.status_code != 200:
            raise DownloadError(f"HTTP {r.status_code} for {url}")
        total = size
        if total is None:
            if r.status_code == 206 and "content-range" in r.headers:
                total = int(r.headers["content-range"].rsplit("/", 1)[-1])
            elif "content-length" in r.headers:
                total = have + int(r.headers["content-length"])
        with part.open("ab" if have else "wb") as f:
            done = have
            for chunk in r.iter_bytes(CHUNK):
                f.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total or done)
