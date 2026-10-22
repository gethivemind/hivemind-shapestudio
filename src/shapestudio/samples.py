"""Sample pictures drawn with Pillow: a shaded sphere (the benchmark object), a box and a
mug. They let a first-time user try the studio before finding a photo, they need no
download, and every machine measures the same input."""
# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

from PIL import Image, ImageDraw, ImageFilter

SIZE = 512
BG = (255, 255, 255)


def _shade(base: tuple[int, int, int], k: float) -> tuple[int, int, int]:
    k = max(0.0, min(1.0, k))
    return tuple(int(c * (0.35 + 0.65 * k)) for c in base)  # type: ignore[return-value]


def sphere() -> Image.Image:
    img = Image.new("RGB", (SIZE, SIZE), BG)
    d = ImageDraw.Draw(img)
    cx, cy, r = SIZE // 2, SIZE // 2, SIZE // 3
    for i in range(r, 0, -2):
        t = i / r
        shade = int(60 + 180 * (1 - t) ** 0.7)
        d.ellipse(
            (
                cx - i - int(20 * t),
                cy - i - int(20 * t),
                cx + i - int(20 * t),
                cy + i - int(20 * t),
            ),
            fill=(shade, int(shade * 0.55), int(shade * 0.25)),
        )
    return img


def box() -> Image.Image:
    """An isometric box with three visible faces, lit from the top left."""
    img = Image.new("RGB", (SIZE, SIZE), BG)
    d = ImageDraw.Draw(img)
    cx, cy = SIZE / 2, SIZE / 2 + 20
    w, h, dep = 150, 150, 120
    ang = math.radians(30)
    dx, dy = dep * math.cos(ang), dep * math.sin(ang)
    base = (70, 120, 200)
    front = [
        (cx - w / 2, cy - h / 2),
        (cx + w / 2, cy - h / 2),
        (cx + w / 2, cy + h / 2),
        (cx - w / 2, cy + h / 2),
    ]
    top = [
        front[0],
        (front[0][0] + dx, front[0][1] - dy),
        (front[1][0] + dx, front[1][1] - dy),
        front[1],
    ]
    side = [
        front[1],
        (front[1][0] + dx, front[1][1] - dy),
        (front[2][0] + dx, front[2][1] - dy),
        front[2],
    ]
    d.polygon(top, fill=_shade(base, 1.0))
    d.polygon(side, fill=_shade(base, 0.45))
    d.polygon(front, fill=_shade(base, 0.75))
    for poly in (top, side, front):
        d.line([*poly, poly[0]], fill=_shade(base, 0.2), width=2)
    return img


def mug() -> Image.Image:
    """A cylinder with a handle: a hole in the silhouette and a thin part."""
    img = Image.new("RGB", (SIZE, SIZE), BG)
    d = ImageDraw.Draw(img)
    base = (200, 80, 60)
    cx, cy = 230, 290
    bw, bh, ell = 170, 220, 46
    # body: vertical shading in strips
    for x in range(int(cx - bw / 2), int(cx + bw / 2)):
        t = (x - (cx - bw / 2)) / bw
        k = 0.45 + 0.55 * math.sin(math.pi * t) ** 0.8
        d.line([(x, cy - bh / 2), (x, cy + bh / 2)], fill=_shade(base, k))
    d.ellipse(
        (cx - bw / 2, cy + bh / 2 - ell / 2, cx + bw / 2, cy + bh / 2 + ell / 2),
        fill=_shade(base, 0.5),
    )
    d.ellipse(
        (cx - bw / 2, cy - bh / 2 - ell / 2, cx + bw / 2, cy - bh / 2 + ell / 2),
        fill=_shade(base, 1.0),
    )
    d.ellipse(
        (cx - bw / 2 + 12, cy - bh / 2 - ell / 2 + 7, cx + bw / 2 - 12, cy - bh / 2 + ell / 2 - 7),
        fill=(60, 30, 25),
    )
    # handle: a thick ring, outer minus inner, on the right
    hx, hy = cx + bw / 2 - 6, cy - 10
    d.pieslice((hx - 60, hy - 85, hx + 80, hy + 85), -80, 80, fill=_shade(base, 0.7))
    d.pieslice((hx - 30, hy - 55, hx + 50, hy + 55), -90, 90, fill=BG)
    # keep the handle behind the body silhouette
    for x in range(int(cx - bw / 2), int(cx + bw / 2)):
        t = (x - (cx - bw / 2)) / bw
        k = 0.45 + 0.55 * math.sin(math.pi * t) ** 0.8
        d.line([(x, cy - bh / 2), (x, cy + bh / 2)], fill=_shade(base, k))
    d.ellipse(
        (cx - bw / 2, cy - bh / 2 - ell / 2, cx + bw / 2, cy - bh / 2 + ell / 2),
        fill=_shade(base, 1.0),
    )
    d.ellipse(
        (cx - bw / 2 + 12, cy - bh / 2 - ell / 2 + 7, cx + bw / 2 - 12, cy - bh / 2 + ell / 2 - 7),
        fill=(60, 30, 25),
    )
    return img.filter(ImageFilter.SMOOTH)


def draw_sample(name: str) -> Image.Image:
    return {"sphere": sphere, "box": box, "mug": mug}[name]()
