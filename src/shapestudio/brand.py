"""The Presidium Shape Studio mark, drawn with Pillow from the family's SVG geometry
(ADR-065): six seats around one table with the chair at the head; Shape Studio puts an
isometric cube on the table. The poster template (dark ground, glow, lockup, tagline) is the
family's (ADR-066), copied from Presidium Image Studio's brand module so the two products
share one layout."""

# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
from pathlib import Path

FAMILY = "Presidium"
PRODUCT = "Shape Studio"
FULL_NAME = f"{FAMILY} {PRODUCT}"
VIOLET = "#5B4FD1"  # Presidium
CORAL = "#D5582F"  # Shape Studio
LIGHT = "#F4F4F5"
GROUND = "#09090b"
TAGLINE = ("Your own 3D studio.", "Free, local, honest.")

SEATS = ((53.1, 18.2), (58.3, 41.0), (43.7, 59.3), (20.3, 59.3), (5.7, 41.0), (10.9, 18.2))
CHAIR = (32.0, 8.0, 5.5)
SEAT_R = 3.6
CUBE_OUTLINE = ((32, 24), (42, 29.5), (42, 40.5), (32, 46), (22, 40.5), (22, 29.5))
CUBE_OUTLINE_STROKE = 3.0
CUBE_INNER = (((22, 29.5), (32, 35), (42, 29.5)), ((32, 35), (32, 46)))
CUBE_INNER_STROKE = 2.6


def draw_mark(size: int, colour: str = CORAL, tile: str | None = None, scale: float = 1.0):
    """The mark as an RGBA Pillow image of ``size`` px, drawn at 4x and downsampled.
    ``tile`` fills a rounded square behind a light mark (the app icon); ``scale`` shrinks the
    mark inside the canvas."""
    from PIL import Image, ImageDraw

    s = size * 4
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    fg = LIGHT if tile else colour
    if tile:
        d.rounded_rectangle((0, 0, s - 1, s - 1), radius=int(s * 14 / 64), fill=tile)
    u = s / 64 * scale
    off = (s - 64 * u) / 2

    def pt(x: float, y: float) -> tuple[float, float]:
        return (off + x * u, off + y * u)

    def circle(x: float, y: float, r: float) -> None:
        cx, cy = pt(x, y)
        d.ellipse((cx - r * u, cy - r * u, cx + r * u, cy + r * u), fill=fg)

    def stroke(points, width_units: float) -> None:
        pts = [pt(x, y) for x, y in points]
        w = max(1, round(width_units * u))
        d.line(pts, fill=fg, width=w, joint="curve")
        for x, y in pts:  # round caps and joins
            d.ellipse((x - w / 2, y - w / 2, x + w / 2, y + w / 2), fill=fg)

    for x, y in SEATS:
        circle(x, y, SEAT_R)
    circle(*CHAIR)
    stroke([*CUBE_OUTLINE, CUBE_OUTLINE[0]], CUBE_OUTLINE_STROKE)
    for seg in CUBE_INNER:
        stroke(seg, CUBE_INNER_STROKE)
    return img.resize((size, size), Image.LANCZOS)


def mark_png(size: int, colour: str = CORAL, tile: str | None = None) -> bytes:
    out = io.BytesIO()
    draw_mark(size, colour, tile).save(out, "PNG", optimize=True)
    return out.getvalue()


def _font(size: int, bold: bool):
    from PIL import ImageFont

    candidates = (
        ["seguisb.ttf", "segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]
    )
    for name in candidates:
        for folder in (Path("C:/Windows/Fonts"), Path("/usr/share/fonts"), Path("/Library/Fonts")):
            p = folder / name
            if p.exists():
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def _glow(img, centre: tuple[int, int], radius: int, colour: str, peak: float = 0.38) -> None:
    """A soft radial glow of ``colour`` behind the mark: the only gradient allowed."""
    from PIL import Image, ImageDraw, ImageFilter

    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    cx, cy = centre
    d.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=int(255 * peak))
    mask = mask.filter(ImageFilter.GaussianBlur(radius * 0.55))
    layer = Image.new("RGBA", (w, h), colour)
    img.paste(layer, (0, 0), mask)


def _fit_font(draw, text: str, size: int, bold: bool, max_width: float):
    """The largest font at or below ``size`` whose rendering of ``text`` fits ``max_width``."""
    while size > 8:
        f = _font(size, bold)
        if draw.textlength(text, font=f) <= max_width:
            return f
        size = int(size * 0.94)
    return _font(size, bold)


def poster(
    width: int,
    height: int,
    colour: str = CORAL,
    family: str = FAMILY,
    product: str = PRODUCT,
    tagline: tuple[str, str] = TAGLINE,
):
    """Store artwork on the dark ground (ADR-066): the mark in colour over a soft glow,
    "Presidium" in medium weight at about 55% of the product size, the product name large and
    regular, then a two-line tagline with the accent colour on the second line. Nothing sits
    within 8% of an edge. Tall (9:16), wide (16:9) and square (1:1) layouts."""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (width, height), GROUND)
    d = ImageDraw.Draw(img)
    margin = 0.08
    if height > width:  # 9:16 poster
        mark_size = int(width * 0.42)
        mark_c = (width // 2, int(height / 3))
        text_w = width * (1 - 2 * margin)
        product_px = int(width * 0.105)
        tag_px = int(width * 0.046)
        text_top = mark_c[1] + mark_size // 2 + int(height * 0.06)
        centre_x = width // 2
    elif width > height:  # 16:9 hero
        mark_size = int(height * 0.5)
        mark_c = (int(width * 0.3), height // 2)
        text_w = width * 0.44
        product_px = int(height * 0.115)
        tag_px = int(height * 0.052)
        text_top = None  # vertically centred block on the right
        centre_x = None
    else:  # 1:1 box art
        mark_size = int(width * 0.36)
        mark_c = (width // 2, int(height * 0.36))
        text_w = width * (1 - 2 * margin)
        product_px = int(width * 0.095)
        tag_px = int(width * 0.042)
        text_top = mark_c[1] + mark_size // 2 + int(height * 0.05)
        centre_x = width // 2
    _glow(img, mark_c, int(mark_size * 0.9), colour)
    mark = draw_mark(mark_size, colour)
    img.alpha_composite(mark, (mark_c[0] - mark_size // 2, mark_c[1] - mark_size // 2))

    f_prod = _fit_font(d, product, product_px, False, text_w)
    f_fam = _fit_font(d, family, int(f_prod.size * 0.55), True, text_w)
    f_tag = _fit_font(d, max(tagline, key=len), tag_px, False, text_w)
    gap = int(f_prod.size * 0.18)
    lines = [
        (family, f_fam, LIGHT),
        (product, f_prod, LIGHT),
        (None, None, int(f_prod.size * 0.45)),  # spacer
        (tagline[0], f_tag, LIGHT),
        (tagline[1], f_tag, colour),
    ]
    heights = [(ln[2] if ln[0] is None else ln[1].size) for ln in lines]
    block_h = sum(heights) + gap * (len(lines) - 1)
    if text_top is None:
        text_top = (height - block_h) // 2
        left = int(width * 0.52)
    else:
        left = None
    y = text_top
    for (text, font, fill), hgt in zip(lines, heights, strict=True):
        if text is not None:
            tw = d.textlength(text, font=font)
            x = left if left is not None else int(centre_x - tw / 2)
            d.text((x, y), text, font=font, fill=fill)
        y += hgt + gap
    return img


STORE_ARTWORK = {
    "box-art-1x1-1080.png": (1080, 1080),
    "box-art-1x1-2160.png": (2160, 2160),
    "poster-9x16-720x1080.png": (720, 1080),
    "poster-9x16-1440x2160.png": (1440, 2160),
    "hero-16x9-1920x1080.png": (1920, 1080),
    "hero-16x9-3840x2160.png": (3840, 2160),
}


def write_store_artwork(root: Path) -> list[Path]:
    """The Store posters, hero images and box art under packaging/store-assets. The display
    icons and MSIX tiles come from scripts/make_artwork.py and are left alone."""
    store = root / "packaging" / "store-assets"
    store.mkdir(parents=True, exist_ok=True)
    written = []
    for name, (w, h) in STORE_ARTWORK.items():
        p = store / name
        poster(w, h).save(p, "PNG", optimize=True)
        written.append(p)
    return written


if __name__ == "__main__":
    import sys

    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
    for path in write_store_artwork(repo):
        print(path.relative_to(repo))
