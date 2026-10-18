"""Draw the Presidium Shape Studio mark at every size the packaging needs: the Windows .ico,
the MSIX tile set, the Store display logo and the website icons.

The mark (ADR-065): six seats around one table, the chair at the head; Shape Studio puts an
isometric cube on the table. Geometry is copied from the brand kit's shape-*.svg (source of
truth: packages/web/src/lib/brand-mark.ts in the app repository); only this file's
rasterisation is ours. Colours: Shape Studio coral #D5582F, light mark #F4F4F5, mono #2C2C2A.
"""
# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
BRAND_KIT = Path(r"C:\Users\ratul.roy\Desktop\Temp\Hivemind\app\brand")
CORAL = "#D5582F"
LIGHT = "#F4F4F5"
MONO = "#2C2C2A"
SEATS = [(53.1, 18.2), (58.3, 41.0), (43.7, 59.3), (20.3, 59.3), (5.7, 41.0), (10.9, 18.2)]
CHAIR = (32.0, 8.0)
CUBE_OUTLINE = [(32, 24), (42, 29.5), (42, 40.5), (32, 46), (22, 40.5), (22, 29.5)]
CUBE_INNER = [[(22, 29.5), (32, 35), (42, 29.5)], [(32, 35), (32, 46)]]


def mark(
    size: int, background: str | None = CORAL, ink: str = LIGHT, rounded: bool = True
) -> Image.Image:
    s = size * 8
    u = s / 64
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if background:
        if rounded:
            d.rounded_rectangle((0, 0, s - 1, s - 1), radius=int(14 * u), fill=background)
        else:
            d.rectangle((0, 0, s - 1, s - 1), fill=background)
    for cx, cy in SEATS:
        d.ellipse(((cx - 3.6) * u, (cy - 3.6) * u, (cx + 3.6) * u, (cy + 3.6) * u), fill=ink)
    cx, cy = CHAIR
    d.ellipse(((cx - 5.5) * u, (cy - 5.5) * u, (cx + 5.5) * u, (cy + 5.5) * u), fill=ink)
    pts = [(x * u, y * u) for x, y in CUBE_OUTLINE]
    d.line([*pts, pts[0]], fill=ink, width=max(1, round(3 * u)), joint="curve")
    for seg in CUBE_INNER:
        d.line(
            [(x * u, y * u) for x, y in seg], fill=ink, width=max(1, round(2.6 * u)), joint="curve"
        )
    # round the stroke ends (ImageDraw.line leaves them square)
    r = 1.5 * u
    for x, y in [*CUBE_OUTLINE, (22, 29.5), (42, 29.5), (32, 46)]:
        d.ellipse((x * u - r, y * u - r, x * u + r, y * u + r), fill=ink)
    return img.resize((size, size), Image.LANCZOS)


def tile(w: int, h: int, pad: float = 0.18) -> Image.Image:
    """Transparent tile with the light mark centred (Windows paints the tile colour behind)."""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    side = int(min(w, h) * (1 - 2 * pad))
    img.alpha_composite(mark(side, background=None, ink=LIGHT), ((w - side) // 2, (h - side) // 2))
    return img


def main() -> None:
    pk = ROOT / "packaging"
    assets = pk / "msix" / "Assets"
    assets.mkdir(parents=True, exist_ok=True)
    (pk / "store-assets").mkdir(exist_ok=True)
    site_assets = ROOT / "site" / "assets"
    site_assets.mkdir(parents=True, exist_ok=True)
    brand_static = ROOT / "src" / "shapestudio" / "static" / "brand"
    brand_static.mkdir(parents=True, exist_ok=True)
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
    mark(512).save(site_assets / "icon-512.png")
    # the brand kit's SVGs, copied verbatim (the kit is the source of truth)
    for name in ("shape-tile", "shape-mark", "shape-mono", "shape-mono-light", "shape-spinner"):
        src = BRAND_KIT / f"{name}.svg"
        if src.exists():
            shutil.copy2(src, brand_static / f"{name}.svg")
            shutil.copy2(src, site_assets / f"{name}.svg")
    shutil.copy2(brand_static / "shape-tile.svg", site_assets / "icon.svg")
    print("artwork written")


if __name__ == "__main__":
    main()
