from __future__ import annotations

import hashlib

import httpx
import pytest

from shapestudio import download as dl

PAYLOAD = bytes(range(256)) * 40  # 10 KiB
SHA = hashlib.sha256(PAYLOAD).hexdigest()


def make_client(payload: bytes, honour_range: bool = True) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        rng = request.headers.get("range")
        if rng and honour_range:
            start = int(rng.split("=")[1].rstrip("-"))
            body = payload[start:]
            return httpx.Response(
                206,
                content=body,
                headers={"content-range": f"bytes {start}-{len(payload) - 1}/{len(payload)}"},
            )
        return httpx.Response(200, content=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_download_verifies_hash(tmp_path):
    target = tmp_path / "f.bin"
    seen = []
    out = dl.download(
        "https://x/f",
        target,
        SHA,
        len(PAYLOAD),
        progress=lambda d, t: seen.append((d, t)),
        client=make_client(PAYLOAD),
    )
    assert out == target and target.read_bytes() == PAYLOAD
    assert seen[-1] == (len(PAYLOAD), len(PAYLOAD))
    assert not target.with_suffix(".bin.part").exists()


def test_download_resumes_partial_file(tmp_path):
    target = tmp_path / "f.bin"
    part = tmp_path / "f.bin.part"
    part.write_bytes(PAYLOAD[:4000])
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.headers.get("range"))
        start = 4000
        return httpx.Response(
            206,
            content=PAYLOAD[start:],
            headers={"content-range": f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}"},
        )

    dl.download(
        "https://x/f", target, SHA, client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    assert requests == ["bytes=4000-"]
    assert target.read_bytes() == PAYLOAD


def test_download_restarts_when_server_ignores_range(tmp_path):
    target = tmp_path / "f.bin"
    (tmp_path / "f.bin.part").write_bytes(b"garbage")
    dl.download("https://x/f", target, SHA, client=make_client(PAYLOAD, honour_range=False))
    assert target.read_bytes() == PAYLOAD


def test_hash_mismatch_deletes_and_raises(tmp_path):
    target = tmp_path / "f.bin"
    with pytest.raises(dl.HashMismatchError):
        dl.download("https://x/f", target, "0" * 64, client=make_client(PAYLOAD))
    assert not target.exists() and not target.with_suffix(".bin.part").exists()


def test_existing_good_file_is_kept(tmp_path):
    target = tmp_path / "f.bin"
    target.write_bytes(PAYLOAD)

    def boom(request: httpx.Request) -> httpx.Response:
        raise AssertionError("network must not be used")

    dl.download(
        "https://x/f", target, SHA, client=httpx.Client(transport=httpx.MockTransport(boom))
    )


def test_http_error_raises(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with pytest.raises(dl.DownloadError):
        dl.download(
            "https://x/f",
            tmp_path / "f",
            SHA,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )


def test_transient_error_is_retried_with_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "RETRY_WAIT_SECONDS", 0.0)
    monkeypatch.setattr(dl, "CHUNK", 1000)  # so bytes reach disk before the drop
    target = tmp_path / "f.bin"
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.headers.get("range"))
        if len(calls) == 1:
            # first attempt: half the file, then the connection drops
            return httpx.Response(200, content=_dropping_stream(PAYLOAD[:5000]))
        start = int(calls[-1].split("=")[1].rstrip("-"))
        return httpx.Response(
            206,
            content=PAYLOAD[start:],
            headers={"content-range": f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}"},
        )

    dl.download(
        "https://x/f",
        target,
        SHA,
        len(PAYLOAD),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert target.read_bytes() == PAYLOAD
    assert calls[0] is None and calls[1] == "bytes=5000-"


def _dropping_stream(data: bytes):
    yield data
    raise httpx.ReadTimeout("stalled")


def test_gives_up_after_max_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "RETRY_WAIT_SECONDS", 0.0)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    with pytest.raises(dl.DownloadError, match="gave up after 3 attempts"):
        dl.download(
            "https://x/f",
            tmp_path / "f",
            SHA,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
            attempts=3,
        )
