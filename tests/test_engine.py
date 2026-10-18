"""The real engine pieces that run without weights: the vendored TripoSR on random weights
(shapes, key names, marching cubes), the GLB/OBJ writer, the CPU turntable, preprocessing,
the runtime report, and the ViT cross-check against transformers when it is installed."""

from __future__ import annotations

import re

import numpy as np
import pytest
from PIL import Image

from shapestudio import runtime
from shapestudio.engine import meshio
from shapestudio.engine.preview import render_view, turntable_sheet
from shapestudio.engine.tsr.preprocess import composite_grey, has_alpha_cutout, resize_foreground


def test_glb_writer_produces_a_valid_gltf_2_with_colours_and_normals(tmp_path):
    v, f, c = meshio.cube()
    p = meshio.write_glb(tmp_path / "cube.glb", v, f, c)
    data = p.read_bytes()
    assert data[:4] == b"glTF" and len(data) % 4 == 0
    j = meshio.read_glb_json(p)
    prim = j["meshes"][0]["primitives"][0]
    assert prim["attributes"] == {"POSITION": 0, "NORMAL": 1, "COLOR_0": 2}
    assert j["accessors"][0]["count"] == 8 and j["accessors"][prim["indices"]]["count"] == 36
    assert j["accessors"][0]["min"] == [-0.5, -0.5, -0.5]
    n = meshio.vertex_normals(v, f)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0)
    obj = meshio.write_obj(tmp_path / "cube.obj", v, f, c).read_text()
    assert obj.count("\nv ") == 8 and obj.count("\nf ") == 12 and "vn " in obj
    with pytest.raises(ValueError):
        meshio.read_glb_json(tmp_path / "cube.obj")


def test_turntable_renders_on_the_cpu(tmp_path):
    v, f, c = meshio.cube()
    im = render_view(v, f, c, size=64)
    arr = np.asarray(im)
    assert im.size == (64, 64) and arr.std() > 10  # something was drawn
    sheet = turntable_sheet(v, f, None, frames=6, tile=32)
    assert sheet.size == (96, 64)
    empty = render_view(np.zeros((0, 3)), np.zeros((0, 3), dtype=int), None, size=16)
    assert empty.size == (16, 16)


def test_preprocessing_matches_triposr_recipe():
    im = Image.new("RGBA", (100, 60), (0, 0, 0, 0))
    for x in range(40, 60):
        for y in range(20, 30):
            im.putpixel((x, y), (200, 100, 50, 255))
    assert has_alpha_cutout(im)
    sq = resize_foreground(im, 0.5)
    assert sq.size[0] == sq.size[1] == 40  # 20 px object / 0.5
    rgb = composite_grey(sq)
    assert rgb.shape == (40, 40, 3) and rgb[0, 0, 0] == pytest.approx(0.5)
    assert not has_alpha_cutout(Image.new("RGB", (4, 4)))
    with pytest.raises(ValueError):
        resize_foreground(Image.new("RGBA", (8, 8), (0, 0, 0, 0)), 0.85)


def test_isosurface_extracts_a_sphere_with_outward_normals():
    import torch

    from shapestudio.engine.tsr.isosurface import MarchingCubeHelper

    h = MarchingCubeHelper(32)
    g = h.grid_vertices
    assert g.shape == (32**3, 3)
    level = 0.4 - torch.linalg.norm(g - 0.5, dim=-1)
    v, f = h(level)
    assert len(v) > 100 and f.shape[1] == 3
    assert torch.allclose(torch.linalg.norm(v - 0.5, dim=-1).mean(), torch.tensor(0.4), atol=0.03)
    n = meshio.vertex_normals(v.numpy(), f.numpy())
    outward = ((v.numpy() - 0.5) * n).sum(axis=1)
    assert (outward > 0).mean() > 0.95
    with pytest.raises(ValueError, match="never crosses"):
        h(torch.ones(32**3))


def test_vendored_triposr_has_the_published_parameter_layout_and_runs_small():
    import torch

    from shapestudio.engine.tsr import TSR

    torch.manual_seed(0)
    small = {
        "cond_image_size": 64,
        "tokenizer": {"plane_size": 4, "num_channels": 64},
        "backbone": {
            "in_channels": 64,
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "num_layers": 1,
            "cross_attention_dim": 768,
        },
        "post_processor": {"in_channels": 64, "out_channels": 8},
        "decoder": {"in_channels": 24, "n_neurons": 16, "n_hidden_layers": 2, "activation": "silu"},
        "renderer": {
            "radius": 0.87,
            "density_activation": "exp",
            "density_bias": -1.0,
            "num_samples_per_ray": 8,
        },
    }
    m = TSR(small).eval()
    keys = set(m.state_dict())
    for k in (
        "image_tokenizer.model.embeddings.cls_token",
        "image_tokenizer.model.encoder.layer.0.attention.attention.query.weight",
        "image_tokenizer.model.pooler.dense.weight",
        "tokenizer.embeddings",
        "backbone.transformer_blocks.0.attn2.to_k.weight",
        "backbone.transformer_blocks.0.ff.net.0.proj.weight",
        "post_processor.upsample.weight",
        "decoder.layers.4.weight",
    ):
        assert k in keys, k
    assert not any("image_mean" in k or "image_std" in k for k in keys)  # buffers are not persisted
    full = TSR()
    assert (
        sum(p.numel() for p in full.parameters()) == 419275628
    )  # 1.677 GB in fp32, like model.ckpt
    img = (np.random.rand(40, 50, 3) * 255).astype(np.uint8)
    codes = m.scene_codes(img, "cpu")
    assert codes.shape == (1, 3, 8, 8, 8)
    m.renderer.set_chunk_size(500)
    d = m.renderer.query_triplane(m.decoder, torch.zeros(1200, 3), codes[0])["density_act"]
    assert d.shape == (1200, 1)
    thr = float(d.median())
    v, f, c = m.extract_mesh(codes[0], resolution=16, threshold=thr)
    assert (
        v.shape[1] == 3 and f.shape[1] == 3 and c.shape == v.shape and 0 <= c.min() <= c.max() <= 1
    )
    renders = m.render(codes[0], n_views=2, size=8)
    assert len(renders) == 2 and renders[0].shape == (8, 8, 3)
    calls = []
    with pytest.raises(RuntimeError, match="cancelled"):

        def check():
            calls.append(1)
            raise RuntimeError("cancelled")

        m.extract_mesh(codes[0], resolution=16, check=check)
    assert calls


def test_checkpoint_mismatch_is_an_error_not_a_partial_load(tmp_path):
    import torch

    from shapestudio.engine.tsr import TSR

    torch.save({"tokenizer.embeddings": torch.zeros(1)}, tmp_path / "bad.ckpt")
    with pytest.raises(RuntimeError, match="does not match"):
        TSR.from_checkpoint(tmp_path / "bad.ckpt")


def test_vit_matches_transformers_reference_when_available():
    tf = pytest.importorskip("transformers")
    import torch

    from shapestudio.engine.tsr.vit import DINO_VITB16, ViTModel

    torch.manual_seed(0)
    hf = tf.ViTModel(
        tf.ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            image_size=224,
            patch_size=16,
            qkv_bias=True,
            layer_norm_eps=1e-12,
            hidden_act="gelu",
        ),
        add_pooling_layer=True,
    ).eval()
    mine = ViTModel(DINO_VITB16).eval()

    def map_key(
        k: str,
    ) -> str:  # transformers >= 5 renamed the layers; the checkpoint uses the old names
        k = re.sub(
            r"^layers\.(\d+)\.attention\.q_proj", r"encoder.layer.\1.attention.attention.query", k
        )
        k = re.sub(
            r"^layers\.(\d+)\.attention\.k_proj", r"encoder.layer.\1.attention.attention.key", k
        )
        k = re.sub(
            r"^layers\.(\d+)\.attention\.v_proj", r"encoder.layer.\1.attention.attention.value", k
        )
        k = re.sub(
            r"^layers\.(\d+)\.attention\.o_proj", r"encoder.layer.\1.attention.output.dense", k
        )
        k = re.sub(r"^layers\.(\d+)\.mlp\.fc1", r"encoder.layer.\1.intermediate.dense", k)
        k = re.sub(r"^layers\.(\d+)\.mlp\.fc2", r"encoder.layer.\1.output.dense", k)
        return re.sub(r"^layers\.(\d+)\.", r"encoder.layer.\1.", k)

    missing, unexpected = mine.load_state_dict(
        {map_key(k): v for k, v in hf.state_dict().items()}, strict=False
    )
    assert not missing and not unexpected
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        ref = hf(x)
        out, pooled = mine(x)
    assert torch.allclose(ref.last_hidden_state, out, atol=1e-5)
    assert torch.allclose(ref.pooler_output, pooled, atol=1e-5)
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        ref = hf(x, interpolate_pos_encoding=True).last_hidden_state
        out, _ = mine(x, interpolate_pos_encoding=True)
    # the interpolation recipe differs between transformers 4.35 (TripoSR's pin, used here)
    # and 5.x (size= instead of scale_factor + 0.1); the outputs agree to about 1e-2
    assert ref.shape == out.shape == (1, 1025, 768)
    assert (ref - out).abs().max() < 0.05


def test_runtime_report_is_honest_about_this_machine(settings):
    rep = runtime.report(settings)
    assert rep["pinned_version"] == "2.14.1" and rep["cuda_label"] == "cu130"
    assert rep["torch"]["installed"] is True and rep["torch"]["flavor"] in ("cpu", "cuda")
    plan = rep["cuda_install"]
    assert plan["possible"] is True and plan["method"] == "pip" and "pip" in plan["command"][2]
    assert runtime.activate_runtime(settings) is None  # nothing unpacked
    assert runtime.remove_cuda_runtime(settings) == 0


def test_fake_engine_writes_everything_the_worker_expects(tmp_path):
    from shapestudio.engine.base import MeshSpec
    from shapestudio.engine.fake import FakeMeshEngine

    e = FakeMeshEngine()
    phases = []
    r = e.reconstruct(
        MeshSpec(tmp_path / "in.png", tmp_path / "out", "job_x", {}),
        lambda p: phases.append(p.phase),
    )
    assert r.glb.exists() and r.obj.exists() and r.preview.exists() and r.cutout.exists()
    assert phases[0] == "loading" and phases[-1] == "saving" and "extracting" in phases
