"""Mesh files without a mesh library: a binary glTF 2.0 (GLB) writer with positions, normals,
per-vertex colours and 32-bit indices, and a Wavefront OBJ writer with vertex colours.

Both are plain numpy. The GLB follows the specification closely enough for Blender,
three.js, model-viewer and Windows' 3D Viewer: one buffer, four buffer views, four
accessors, one mesh primitive (TRIANGLES), one node, one scene.
"""
# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

GLB_MAGIC = 0x46546C67
GLB_JSON = 0x4E4F534A
GLB_BIN = 0x004E4942
FLOAT = 5126
UNSIGNED_INT = 5125
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals, unit length (zero for isolated vertices)."""
    v = vertices.astype(np.float64)
    f = faces.astype(np.int64)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    length[length == 0] = 1.0
    return (n / length).astype(np.float32)


def _pad4(b: bytes, fill: bytes = b"\x00") -> bytes:
    return b + fill * ((4 - len(b) % 4) % 4)


def write_glb(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray | None = None,
    normals: np.ndarray | None = None,
    name: str = "mesh",
    generator: str = "Presidium Shape Studio",
) -> Path:
    """``vertices`` (N,3) float, ``faces`` (M,3) int, ``colors`` (N,3) float 0-1 or uint8."""
    v = np.ascontiguousarray(vertices, dtype=np.float32)
    f = np.ascontiguousarray(faces, dtype=np.uint32)
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("vertices must be (N,3) and faces (M,3)")
    if normals is None:
        normals = vertex_normals(v, f)
    n = np.ascontiguousarray(normals, dtype=np.float32)
    c = None
    if colors is not None:
        c = np.asarray(colors)
        if c.dtype == np.uint8:
            c = c.astype(np.float32) / 255.0
        c = np.ascontiguousarray(np.clip(c[:, :3], 0.0, 1.0), dtype=np.float32)

    blobs = [v.tobytes(), n.tobytes()]
    if c is not None:
        blobs.append(c.tobytes())
    blobs.append(f.tobytes())
    views = []
    offset = 0
    bin_parts = []
    for i, b in enumerate(blobs):
        b = _pad4(b)
        target = ELEMENT_ARRAY_BUFFER if i == len(blobs) - 1 else ARRAY_BUFFER
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(b), "target": target})
        bin_parts.append(b)
        offset += len(b)
    binary = b"".join(bin_parts)

    accessors = [
        {
            "bufferView": 0,
            "componentType": FLOAT,
            "count": int(v.shape[0]),
            "type": "VEC3",
            "min": [float(x) for x in v.min(axis=0)] if len(v) else [0, 0, 0],
            "max": [float(x) for x in v.max(axis=0)] if len(v) else [0, 0, 0],
        },
        {"bufferView": 1, "componentType": FLOAT, "count": int(n.shape[0]), "type": "VEC3"},
    ]
    attributes = {"POSITION": 0, "NORMAL": 1}
    if c is not None:
        accessors.append(
            {"bufferView": 2, "componentType": FLOAT, "count": int(c.shape[0]), "type": "VEC3"}
        )
        attributes["COLOR_0"] = 2
    idx_view = len(views) - 1
    accessors.append(
        {
            "bufferView": idx_view,
            "componentType": UNSIGNED_INT,
            "count": int(f.size),
            "type": "SCALAR",
        }
    )
    gltf = {
        "asset": {"version": "2.0", "generator": generator},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [
            {
                "name": name,
                "primitives": [
                    {
                        "attributes": attributes,
                        "indices": len(accessors) - 1,
                        "material": 0,
                        "mode": 4,
                    }
                ],
            }
        ],
        "materials": [
            {
                "name": "vertex-colours",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.9,
                },
                "doubleSided": True,
            }
        ],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": views,
        "accessors": accessors,
    }
    json_bytes = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    total = 12 + 8 + len(json_bytes) + 8 + len(binary)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(struct.pack("<III", GLB_MAGIC, 2, total))
        fh.write(struct.pack("<II", len(json_bytes), GLB_JSON))
        fh.write(json_bytes)
        fh.write(struct.pack("<II", len(binary), GLB_BIN))
        fh.write(binary)
    return path


def read_glb_json(path: Path) -> dict:
    """The JSON chunk of a GLB (for tests and the gallery's vertex/face counts)."""
    data = Path(path).read_bytes()
    magic, version, _length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC or version != 2:
        raise ValueError("not a GLB 2.0 file")
    chunk_len, chunk_type = struct.unpack_from("<II", data, 12)
    if chunk_type != GLB_JSON:
        raise ValueError("first GLB chunk is not JSON")
    return json.loads(data[20 : 20 + chunk_len].decode("utf-8"))


def write_obj(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray | None = None,
    normals: np.ndarray | None = None,
    comment: str = "Presidium Shape Studio",
) -> Path:
    """OBJ with the common ``v x y z r g b`` vertex-colour extension (read by Blender,
    MeshLab and most viewers) and per-vertex normals."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64) + 1
    c = None
    if colors is not None:
        c = np.asarray(colors)
        if c.dtype == np.uint8:
            c = c.astype(np.float64) / 255.0
        c = np.clip(c[:, :3], 0.0, 1.0)
    if normals is None:
        normals = vertex_normals(vertices, faces)
    n = np.asarray(normals, dtype=np.float64)
    lines = [f"# {comment}", f"# {len(v)} vertices, {len(f)} faces"]
    if c is not None:
        lines += [
            f"v {x:.6f} {y:.6f} {z:.6f} {r:.4f} {g:.4f} {b:.4f}"
            for (x, y, z), (r, g, b) in zip(v, c, strict=True)
        ]
    else:
        lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in v]
    lines += [f"vn {x:.4f} {y:.4f} {z:.4f}" for x, y, z in n]
    lines += [f"f {a}//{a} {b}//{b} {d}//{d}" for a, b, d in f]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def cube(size: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A coloured cube (8 vertices, 12 triangles): the fake engine's output and a test mesh."""
    s = size
    v = np.array(
        [
            [-s, -s, -s],
            [s, -s, -s],
            [s, s, -s],
            [-s, s, -s],
            [-s, -s, s],
            [s, -s, s],
            [s, s, s],
            [-s, s, s],
        ],
        dtype=np.float32,
    )
    f = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [1, 2, 6],
            [1, 6, 5],
            [0, 4, 7],
            [0, 7, 3],
        ],
        dtype=np.uint32,
    )
    c = (v + s) / (2 * s)
    return v, f, c.astype(np.float32)
