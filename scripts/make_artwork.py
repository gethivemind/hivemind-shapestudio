"""Draw the studio mark at every size the packaging needs: the Windows .ico, the MSIX tile
set, the Store display logo and the website icon. Pure Pillow, deterministic, no fonts."""
# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
HONEY = "#f0b429"
DARK = "#0f1012"
INK = "#1b1b19"
GREY = "#3a3a37"
PAPER = "#f7f6f2"


def mark(size: int, background: str | None = DARK, rounded: bool = True) -> Image.Image:
    s = size * 4
    u = s / 64
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if background:
        if rounded:
            d.rounded_rectangle((0, 0, s - 1, s - 1), radius=int(14 * u), fill=background)
        else:
            d.rectangle((0, 0, s - 1, s - 1), fill=background)
    hexagon = [(32, 6), (54, 19), (54, 45), (32, 58), (10, 45), (10, 19)]
    d.polygon([(x * u, y * u) for x, y in hexagon], fill=HONEY)
    d.polygon([(x * u, y * u) for x, y in [(32, 20), (44, 27), (32, 34), (20, 27)]], fill=PAPER)
    d.polygon([(x * u, y * u) for x, y in [(20, 27), (32, 34), (32, 48), (20, 41)]], fill=INK)
    d.polygon([(x * u, y * u) for x, y in [(32, 34), (44, 27), (44, 41), (32, 48)]], fill=GREY)
    return img.resize((size, size), Image.LANCZOS)


def tile(w: int, h: int, pad: float = 0.18) -> Image.Image:
    """Transparent tile with the mark centred (Windows paints the tile colour behind)."""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    side = int(min(w, h) * (1 - 2 * pad))
    m = mark(side, background=None)
    img.alpha_composite(m, ((w - side) // 2, (h - side) // 2))
    return img


def main() -> None:
    pk = ROOT / "packaging"
    assets = pk / "msix" / "Assets"
    assets.mkdir(parents=True, exist_ok=True)
    (pk / "store-assets").mkdir(exist_ok=True)
    (ROOT / "site" / "assets").mkdir(parents=True, exist_ok=True)
    mark(256).save(
        pk / "ShapeStudio.ico",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    for size in (71, 150, 300):
        mark(size).save(pk / "store-assets" / f"display-1x1-{size}.png")
    scales = {"": 1.0, ".scale-125": 1.25, ".scale-150": 1.5, ".scale-200": 2.0, ".scale-400": 4.0}
    for name, (w, h) in {
        "Square150x150Logo": (150, 150),
        "Square44x44Logo": (44, 44),
        "Wide310x150Logo": (310, 150),
        "StoreLogo": (50, 50),
    }.items():
        for suffix, f in scales.items():
            tile(round(w * f), round(h * f)).save(assets / f"{name}{suffix}.png")
    tile(71, 71).save(assets / "Square71x71Logo.png")
    tile(310, 310).save(assets / "Square310x310Logo.png")
    tile(620, 300).save(assets / "SplashScreen.png")
    mark(44, background=None).save(assets / "Square44x44Logo.targetsize-44_altform-unplated.png")
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
        "<path d='M32 6 54 19v26L32 58 10 45V19z' fill='#f0b429'/>"
        "<path d='M32 20l12 7-12 7-12-7z' fill='#f7f6f2'/>"
        "<path d='M20 27l12 7v14l-12-7z' fill='#1b1b19'/>"
        "<path d='M44 27l-12 7v14l12-7z' fill='#3a3a37'/></svg>\n"
    )
    (ROOT / "site" / "assets" / "icon.svg").write_text(svg, encoding="utf-8")
    mark(512).save(ROOT / "site" / "assets" / "icon-512.png")
    print("artwork written")


if __name__ == "__main__":
    main()
