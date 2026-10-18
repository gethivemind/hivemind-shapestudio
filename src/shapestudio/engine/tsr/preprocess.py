"""Input preparation as TripoSR's run.py does it: a cutout (RGBA) is cropped to its content,
padded to a square so the object fills ``foreground_ratio`` of the frame, and composited
over mid-grey, which is what the model saw in training. MIT upstream; rewritten here."""

from __future__ import annotations

import numpy as np
from PIL import Image


def resize_foreground(image: Image.Image, ratio: float) -> Image.Image:
    """RGBA in → RGBA out, square, object centred and scaled to ``ratio`` of the side."""
    arr = np.asarray(image.convert("RGBA"))
    ys, xs = np.where(arr[..., 3] > 0)
    if len(ys) == 0:
        raise ValueError("the cutout is fully transparent: no object was found in the image")
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    fg = arr[y1:y2, x1:x2]
    size = max(fg.shape[0], fg.shape[1])
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
    sq = np.pad(fg, ((ph0, ph1), (pw0, pw1), (0, 0)))
    new_size = int(size / ratio)
    p0 = (new_size - size) // 2
    p1 = new_size - size - p0
    out = np.pad(sq, ((p0, p1), (p0, p1), (0, 0)))
    return Image.fromarray(out)


def composite_grey(image: Image.Image, grey: float = 0.5) -> np.ndarray:
    """RGBA → RGB float32 in 0-1 over a flat grey background."""
    arr = np.asarray(image.convert("RGBA")).astype(np.float32) / 255.0
    rgb, a = arr[..., :3], arr[..., 3:4]
    return rgb * a + (1.0 - a) * grey


def has_alpha_cutout(image: Image.Image) -> bool:
    """True when the picture already carries a transparent background (as run.py checks)."""
    if image.mode != "RGBA":
        return False
    extrema = image.getextrema()
    return extrema[3][0] < 255
