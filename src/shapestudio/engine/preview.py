"""A turntable sheet of a mesh rendered without a GPU: numpy for the maths, Pillow to fill
the triangles (painter's algorithm, back to front), 2× supersampled. Lambert shading with a
headlight plus ambient, vertex colours averaged per face. Good enough to judge a
reconstruction at a glance and fast enough for 300k triangles on a laptop CPU."""
# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BG_DARK = (24, 24, 27)
BG_LIGHT = (244, 243, 239)


def _rotation(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """World (x right, y forward, z up as TripoSR emits) → camera looking at the origin."""
    a = math.radians(azimuth_deg)
    e = math.radians(elevation_deg)
    rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, math.cos(e), -math.sin(e)], [0, math.sin(e), math.cos(e)]])
    # z-up world to y-up camera: swap axes so the screen's up is world z
    to_cam = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    return to_cam @ rx @ rz


def render_view(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray | None,
    size: int = 512,
    azimuth_deg: float = 30.0,
    elevation_deg: float = 20.0,
    background: tuple[int, int, int] = BG_DARK,
    supersample: int = 2,
    max_faces: int = 250_000,
) -> Image.Image:
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if len(f) == 0 or len(v) == 0:
        return Image.new("RGB", (size, size), background)
    if len(f) > max_faces:  # keep the preview bounded; a random subset still reads as the shape
        rng = np.random.default_rng(0)
        f = f[np.sort(rng.choice(len(f), max_faces, replace=False))]
    c = None
    if colors is not None:
        c = np.asarray(colors, dtype=np.float64)
        if c.max() > 1.0:
            c = c / 255.0
    # normalise to the unit cube around the centroid of the bounding box
    lo, hi = v.min(axis=0), v.max(axis=0)
    centre = (lo + hi) / 2
    scale = (hi - lo).max() or 1.0
    vn = (v - centre) / scale
    r = _rotation(azimuth_deg, elevation_deg)
    cam = vn @ r.T  # x right, y up, z towards the viewer (negative = away)
    s = size * supersample
    margin = 0.08
    px = (cam[:, 0] * (1 - 2 * margin) + 0.5) * s
    py = (0.5 - cam[:, 1] * (1 - 2 * margin)) * s
    depth = cam[:, 2]
    tri = f
    p0, p1, p2 = cam[tri[:, 0]], cam[tri[:, 1]], cam[tri[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    nl = np.linalg.norm(n, axis=1)
    keep = nl > 1e-12
    tri, n, nl = tri[keep], n[keep], nl[keep]
    n = n / nl[:, None]
    light = np.array([0.3, 0.5, 0.81])
    light = light / np.linalg.norm(light)
    lambert = np.abs(n @ light)  # double-sided
    shade = 0.35 + 0.65 * lambert
    if c is not None:
        base = (c[tri[:, 0]] + c[tri[:, 1]] + c[tri[:, 2]]) / 3.0
    else:
        base = np.full((len(tri), 3), 0.72)
    rgb = np.clip(base * shade[:, None] * 255.0, 0, 255).astype(np.uint8)
    order = np.argsort(depth[tri[:, 0]] + depth[tri[:, 1]] + depth[tri[:, 2]])  # far first
    img = Image.new("RGB", (s, s), background)
    draw = ImageDraw.Draw(img)
    xs, ys = px, py
    for i in order:
        a, b, d = tri[i]
        col = (int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2]))
        draw.polygon([(xs[a], ys[a]), (xs[b], ys[b]), (xs[d], ys[d])], fill=col, outline=col)
    if supersample > 1:
        img = img.resize((size, size), Image.LANCZOS)
    return img


def turntable_sheet(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray | None,
    frames: int = 6,
    tile: int = 512,
    elevation_deg: float = 20.0,
    background: tuple[int, int, int] = BG_DARK,
) -> Image.Image:
    """``frames`` views around the object in a grid of 3 columns (1536 px wide at 512)."""
    cols = min(3, frames)
    rows = math.ceil(frames / cols)
    sheet = Image.new("RGB", (cols * tile, rows * tile), background)
    for i in range(frames):
        az = 30.0 + 360.0 * i / frames
        im = render_view(vertices, faces, colors, tile, az, elevation_deg, background)
        sheet.paste(im, ((i % cols) * tile, (i // cols) * tile))
    return sheet


def write_previews(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray | None,
    sheet_path: Path,
    thumb_path: Path,
    frames: int = 6,
    tile: int = 512,
) -> None:
    sheet = turntable_sheet(vertices, faces, colors, frames, tile)
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(sheet_path, "PNG", optimize=True)
    sheet.crop((0, 0, tile, tile)).save(thumb_path, "PNG", optimize=True)
