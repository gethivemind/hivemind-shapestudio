"""Small mesh operations on a finished reconstruction, with numpy only: binary STL for
printing, flipped winding, Laplacian smoothing, vertex-clustering decimation and scaling to a
real-world size. Each writes a new file next to the original; nothing is changed in place."""
# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from shapestudio.engine.meshio import vertex_normals, write_glb, write_obj

OPS = ("stl", "flip", "smooth", "decimate", "scale")


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Reads the OBJ this studio writes (``v x y z [r g b]``, ``f a//a b//b c//c``)."""
    verts: list[list[float]] = []
    cols: list[list[float]] = []
    faces: list[list[int]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(x) for x in parts[1:4]])
                if len(parts) >= 7:
                    cols.append([float(x) for x in parts[4:7]])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]]
                faces.append(idx)
    v = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    c = np.asarray(cols, dtype=np.float64) if len(cols) == len(verts) and verts else None
    return v, f, c


def write_stl(path: Path, vertices: np.ndarray, faces: np.ndarray, name: str = "mesh") -> Path:
    """Binary STL: 80-byte header, triangle count, then normal + 3 vertices + attribute."""
    v = np.asarray(vertices, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)
    tri = v[f]  # (n, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lens = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(lens > 0, n / np.maximum(lens, 1e-12), 0).astype(np.float32)
    rec = np.zeros(
        len(f),
        dtype=np.dtype(
            [("n", np.float32, 3), ("v", np.float32, (3, 3)), ("attr", np.uint16)], align=False
        ),
    )
    rec["n"] = n
    rec["v"] = tri
    header = f"Presidium Shape Studio {name}".encode()[:80].ljust(80, b"\x00")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(header)
        fh.write(struct.pack("<I", len(f)))
        fh.write(rec.tobytes())
    return path


def flip(faces: np.ndarray) -> np.ndarray:
    return np.asarray(faces)[:, ::-1].copy()


def smooth(vertices: np.ndarray, faces: np.ndarray, iterations: int = 2, lam: float = 0.5):
    """Uniform Laplacian smoothing: each vertex moves towards the mean of its neighbours."""
    v = np.asarray(vertices, dtype=np.float64).copy()
    f = np.asarray(faces, dtype=np.int64)
    if len(v) == 0 or len(f) == 0:
        return v
    i = np.concatenate([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    j = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    deg = np.bincount(i, minlength=len(v)).astype(np.float64)
    deg[deg == 0] = 1
    for _ in range(max(1, int(iterations))):
        acc = np.zeros_like(v)
        np.add.at(acc, i, v[j])
        mean = acc / deg[:, None]
        v = v + lam * (mean - v)
    return v


def decimate(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray | None,
    target_faces: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Vertex clustering on a uniform grid, sized so the result lands near ``target_faces``.
    Simple and robust; it does not preserve sharp edges the way quadric decimation does."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if len(f) <= target_faces or len(v) == 0:
        return v, f, colors
    lo, hi = v.min(axis=0), v.max(axis=0)
    span = max(float((hi - lo).max()), 1e-9)
    # faces scale roughly with cells^(2/3) for a surface; solve for the grid size
    cells = max(2, round((target_faces / max(len(f), 1)) ** 0.5 * (len(v) ** (1 / 3)) * 2))
    for _ in range(12):
        step = span / cells
        key = np.floor((v - lo) / step).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        nf = f.copy()
        nf = inv[nf]
        keep = (nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])
        nf = nf[keep]
        if len(nf) <= target_faces or cells > 4096:
            break
        cells = int(cells * 1.3) + 1
    n = int(inv.max()) + 1 if len(inv) else 0
    nv = np.zeros((n, 3))
    cnt = np.bincount(inv, minlength=n).astype(np.float64)
    cnt[cnt == 0] = 1
    for k in range(3):
        nv[:, k] = np.bincount(inv, weights=v[:, k], minlength=n) / cnt
    nc = None
    if colors is not None:
        c = np.asarray(colors, dtype=np.float64)
        if c.dtype == np.uint8 or c.max() > 1.0:
            c = c / 255.0
        nc = np.zeros((n, 3))
        for k in range(3):
            nc[:, k] = np.bincount(inv, weights=c[:, k], minlength=n) / cnt
    used = np.unique(nf)
    remap = -np.ones(n, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return nv[used], remap[nf], (nc[used] if nc is not None else None)


def scale_to(vertices: np.ndarray, size: float, axis: str = "max") -> np.ndarray:
    """Uniform scale so the mesh's longest side (or the chosen axis) spans ``size`` units."""
    v = np.asarray(vertices, dtype=np.float64)
    if len(v) == 0:
        return v
    ext = v.max(axis=0) - v.min(axis=0)
    cur = float(ext.max()) if axis == "max" else float(ext["xyz".index(axis)])
    if cur <= 0:
        return v
    return (v - v.mean(axis=0)) * (size / cur)


def apply(op: str, obj_path: Path, out_stem: Path, params: dict | None = None) -> dict:
    """Run one operation on the OBJ at ``obj_path`` and write the result as
    ``<out_stem>.<ext>``. Returns what was written and the new counts."""
    params = params or {}
    if op not in OPS:
        raise ValueError(f"unknown mesh operation {op!r}; one of {', '.join(OPS)}")
    out_stem = Path(out_stem)

    def out_path(ext: str) -> Path:  # not with_suffix: the stem itself contains dots
        return out_stem.parent / f"{out_stem.name}.{ext}"

    v, f, c = read_obj(obj_path)
    note = ""
    if op == "stl":
        unit_mm = float(params.get("size_mm") or 0)
        if unit_mm > 0:
            v = scale_to(v, unit_mm)
            note = f"longest side {unit_mm:g} mm"
        p = write_stl(out_path("stl"), v, f, out_stem.name)
        return {"path": str(p), "format": "stl", "vertices": len(v), "faces": len(f), "note": note}
    if op == "flip":
        f = flip(f)
        note = "winding reversed (normals point the other way)"
    elif op == "smooth":
        it = int(params.get("iterations") or 2)
        v = smooth(v, f, iterations=it, lam=float(params.get("strength") or 0.5))
        note = f"{it} smoothing pass(es)"
    elif op == "decimate":
        target = int(params.get("target_faces") or max(1000, len(f) // 4))
        v, f, c = decimate(v, f, c, target)
        note = f"reduced to {len(f)} faces"
    elif op == "scale":
        size = float(params.get("size") or 1.0)
        v = scale_to(v, size, str(params.get("axis") or "max"))
        note = f"longest side {size:g} units"
    normals = vertex_normals(v, f) if len(f) else None
    fmt = str(params.get("format") or "glb").lower()
    if fmt == "obj":
        p = write_obj(out_path("obj"), v, f, c, normals)
    elif fmt == "stl":
        p = write_stl(out_path("stl"), v, f, out_stem.name)
    else:
        fmt = "glb"
        p = write_glb(out_path("glb"), v, f, c, name=out_stem.name)
    return {"path": str(p), "format": fmt, "vertices": len(v), "faces": len(f), "note": note}
