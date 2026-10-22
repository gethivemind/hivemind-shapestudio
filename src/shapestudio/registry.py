"""The catalogue of what the studio may download and run.

Every file is pinned by SHA-256 (the Hugging Face LFS object id, or the hash of the file as
fetched on 2026-10-18 for the small non-LFS files) and by size, and every model carries its
licence. Only permissive licences are listed; weights under non-commercial or revenue-gated
terms are excluded by policy (Hivemind ADR-051) and named in ``EXCLUDED_BY_POLICY`` with the
reason.
"""
# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

GIB = 1024**3
MIB = 1024**2

HF = "https://huggingface.co/{repo}/resolve/main/{path}"


@dataclass(frozen=True)
class FileSpec:
    """One downloadable file. ``filename`` is relative to the models folder and may contain
    a sub-folder (``trellis-image-large/ckpts/…``)."""

    filename: str
    url: str
    sha256: str
    size: int
    licence: str
    source: str  # human-readable origin


@dataclass(frozen=True)
class Preset:
    """A quality/time trade-off chosen explicitly by the caller.

    For a feed-forward reconstructor the knob is the marching-cubes grid: ``resolution``
    cells per axis. Higher means finer surface detail and more triangles; the network pass
    itself is the same at every preset."""

    name: str
    resolution: int
    tradeoff: str  # stated in words


@dataclass(frozen=True)
class Requirements:
    """What a model needs, stated as minimum and recommended so the UI can show both next
    to the model and mark which this machine meets. Memory figures are for the ``best``
    preset; ``min`` is what the ``fast`` preset gets by without swapping."""

    min_ram_gib: float  # CPU path, or system RAM alongside a GPU run
    rec_ram_gib: float
    min_vram_gib: float  # 0 = the GPU path is not offered
    rec_vram_gib: float
    cpu_ok: bool
    gpu_required: bool
    summary: str  # one sentence, for the cards

    def to_dict(self) -> dict:
        return {
            "min_ram_gib": self.min_ram_gib,
            "rec_ram_gib": self.rec_ram_gib,
            "min_vram_gib": self.min_vram_gib,
            "rec_vram_gib": self.rec_vram_gib,
            "cpu_ok": self.cpu_ok,
            "gpu_required": self.gpu_required,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class MeshModel:
    """An image-to-3D model the studio can run.

    ``tier`` names the quality tier the planner offers: ``standard`` runs on a CPU, ``best``
    needs a discrete NVIDIA GPU. ``quality_rank`` 0 is the strongest model in the catalogue.
    ``verified`` says whether the Hivemind project has run this model end to end; the
    catalogue never claims a timing that was not measured."""

    name: str
    display_name: str
    licence: str
    tier: str  # standard | best
    quality_rank: int
    files: tuple[FileSpec, ...]
    presets: dict[str, Preset]
    homepage: str
    description: str
    weights_bytes_in_memory: int  # resident weights once loaded (fp32 on CPU / fp16 on GPU)
    cpu_capable: bool
    min_vram_bytes: int  # for the GPU path; 0 = any CUDA device
    cuda_only: bool = False
    verified: str = "not run yet"  # tests only | not run yet | measured on <machine>
    engine: str = "triposr"
    note: str = ""
    extra_runtime: tuple[str, ...] = field(default_factory=tuple)  # packages not bundled
    requirements: Requirements | None = None

    @property
    def download_bytes(self) -> int:
        return sum(f.size for f in self.files)

    @property
    def folder(self) -> str:
        return self.files[0].filename.split("/", 1)[0]


# --- TripoSR (Stability AI + Tripo AI, MIT) ------------------------------------------------

_TRIPOSR_FILES = (
    FileSpec(
        filename="triposr/model.ckpt",
        url=HF.format(repo="stabilityai/TripoSR", path="model.ckpt"),
        sha256="429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee",
        size=1677246742,
        licence="MIT",
        source="stabilityai/TripoSR (Hugging Face, LFS object id)",
    ),
    FileSpec(
        filename="triposr/config.yaml",
        url=HF.format(repo="stabilityai/TripoSR", path="config.yaml"),
        sha256="74ca708ce086bf68e97709ea6b3d91f14717921c04691e84043f0eb8fcc68e62",
        size=987,
        licence="MIT",
        source="stabilityai/TripoSR (Hugging Face)",
    ),
)

_TRIPOSR_PRESETS = {
    "best": Preset(
        "best",
        256,
        "Marching cubes on a 256³ grid: the finest surface TripoSR's field resolves. The "
        "most triangles and the longest extraction; the network pass is identical.",
    ),
    "balanced": Preset(
        "balanced",
        192,
        "192³ grid: about half the extraction work, slightly softer small details.",
    ),
    "fast": Preset(
        "fast",
        128,
        "128³ grid: a preview-grade surface in a fraction of the extraction time; thin parts "
        "may break up.",
    ),
}

TRIPOSR = MeshModel(
    name="triposr",
    display_name="TripoSR",
    licence="MIT",
    tier="standard",
    quality_rank=1,
    files=_TRIPOSR_FILES,
    presets=_TRIPOSR_PRESETS,
    homepage="https://huggingface.co/stabilityai/TripoSR",
    description=(
        "Feed-forward single-image reconstruction (LRM family) by Stability AI and Tripo AI: "
        "a DINO ViT reads the picture, a transformer writes a triplane, a small NeRF MLP is "
        "queried on a grid and marching cubes turns it into a mesh with vertex colours. One "
        "network pass per object; runs on a CPU."
    ),
    weights_bytes_in_memory=1677246742,  # fp32 state dict as shipped
    cpu_capable=True,
    min_vram_bytes=4 * GIB,
    verified="runs end to end in this project's tests; the first real run with the "
    "downloaded weights on your machine is what makes the estimates measured",
    engine="triposr",
    note="1.56 GiB download. The reconstruction code is vendored under engine/tsr (MIT).",
    requirements=Requirements(
        min_ram_gib=8,
        rec_ram_gib=16,
        min_vram_gib=4,
        rec_vram_gib=6,
        cpu_ok=True,
        gpu_required=False,
        summary="Any 64-bit PC with 8 GiB of RAM runs it on the CPU; 16 GiB keeps the best "
        "preset out of swap. An NVIDIA GPU with 4 GiB or more makes it much faster.",
    ),
)


# --- TRELLIS (Microsoft, MIT) ---------------------------------------------------------------

_TRELLIS_REPO = "microsoft/TRELLIS-image-large"


def _trellis(path: str, sha: str, size: int) -> FileSpec:
    return FileSpec(
        filename=f"trellis-image-large/{path}",
        url=HF.format(repo=_TRELLIS_REPO, path=path),
        sha256=sha,
        size=size,
        licence="MIT",
        source=f"{_TRELLIS_REPO} (Hugging Face)",
    )


_TRELLIS_FILES = (
    _trellis(
        "pipeline.json", "2686b6c9b3b0a100b544d5f9dc7b3ecb15c9d98be9852d68e8731bd8dbe52a42", 1987
    ),
    _trellis(
        "ckpts/ss_enc_conv3d_16l8_fp16.json",
        "12efe92a0d7dcd790f251acb94a6950957ea4398268e2838ef7319c3f20b071e",
        244,
    ),
    _trellis(
        "ckpts/ss_enc_conv3d_16l8_fp16.safetensors",
        "107874eeaa0feb82f51b19db5da7db534fb7e7f19e5a122b9ff1bc2e258bfc6d",
        119068016,
    ),
    _trellis(
        "ckpts/ss_dec_conv3d_16l8_fp16.json",
        "646781293f1cda74720de85d1cef50a957fb4aebd9a4bd014e454e32f2330ac5",
        245,
    ),
    _trellis(
        "ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
        "1c76d4a40519aa2d711cc263a8404105231ac26db31d946bed48b84fee79009a",
        147591972,
    ),
    _trellis(
        "ckpts/ss_flow_img_dit_L_16l8_fp16.json",
        "bf69161da7ece9a87394dd4441f0cfa6d4261696a214a8f68388c81fa9a6b5b7",
        385,
    ),
    _trellis(
        "ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors",
        "96dc6bfd4136fd950af564dd16b4ae533c9ba6af8f26c670646b2a9f2789b1db",
        1130770840,
    ),
    _trellis(
        "ckpts/slat_enc_swin8_B_64l8_fp16.json",
        "7b7e2bd3e831bdd5a6129cea36424a47ad73c1b9ad109d1de54e0f14af4d28a2",
        321,
    ),
    _trellis(
        "ckpts/slat_enc_swin8_B_64l8_fp16.safetensors",
        "21dceac6bee917ab6458ff52c9757ba89a779d03031c7bd17f9e7f0103bfd436",
        173242816,
    ),
    _trellis(
        "ckpts/slat_flow_img_dit_L_64l8p2_fp16.json",
        "c0a9bd57227f55cf75ff02562c27bb05c50174f06955f56a79dea2982fd7a8f9",
        442,
    ),
    _trellis(
        "ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors",
        "693fb2a58ad497bd222007301eeec49d14d60f8c12d2f2f00c221fa747b4c66c",
        1203755136,
    ),
    _trellis(
        "ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.json",
        "293900795cb69cc972d1a20a4940ee12f7e007952c2ae92d128dda91b96ac317",
        372,
    ),
    _trellis(
        "ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors",
        "3e87aba94b5786407eb06d0502c1ed0885a0027a3f2b8537bfe15b0a92c01859",
        181903412,
    ),
    _trellis(
        "ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.json",
        "ebe9d3eade6d9cb7662efae109197e97d7ab58aebd5e8ede0c8991dfce88a06a",
        843,
    ),
    _trellis(
        "ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors",
        "38c84bcef5ce0af1f48b1b5558dabc7575a13346043c41a7e0610f1fa619a161",
        171450952,
    ),
    _trellis(
        "ckpts/slat_dec_rf_swin8_B_64l8r16_fp16.json",
        "9b46a04602988b9cd83f1d0537de1cb93c4c1b6d35a0bbbcb07bb37bac66b766",
        396,
    ),
    _trellis(
        "ckpts/slat_dec_rf_swin8_B_64l8r16_fp16.safetensors",
        "737da6578d01948016b7c39786113af0d64a46f7922f6b8b5e698b84643be514",
        171450488,
    ),
)

_TRELLIS_PRESETS = {
    "best": Preset(
        "best",
        256,
        "TRELLIS's full pipeline: 25 sparse-structure steps, 25 latent steps, mesh decoder at 256.",
    ),
    "balanced": Preset(
        "balanced",
        256,
        "Same pipeline; the studio does not lower TRELLIS's steps yet (not measured).",
    ),
    "fast": Preset(
        "fast", 256, "Same pipeline; the studio does not lower TRELLIS's steps yet (not measured)."
    ),
}

TRELLIS = MeshModel(
    name="trellis-image-large",
    display_name="TRELLIS image-large",
    licence="MIT",
    tier="best",
    quality_rank=0,
    files=_TRELLIS_FILES,
    presets=_TRELLIS_PRESETS,
    homepage="https://huggingface.co/microsoft/TRELLIS-image-large",
    description=(
        "Microsoft's structured-latent 3D generator: a sparse voxel structure is sampled from "
        "the image with a flow transformer, refined into structured latents and decoded to a "
        "textured mesh. Clearly higher fidelity than TripoSR, and a discrete NVIDIA GPU with "
        "16 GB of memory or more is required (the authors' figure); it has no CPU path."
    ),
    weights_bytes_in_memory=3_300_000_000,  # fp16 checkpoints above plus the DINOv2 encoder
    cpu_capable=False,
    min_vram_bytes=16 * GIB,
    cuda_only=True,
    verified="not run yet: catalogue entry, fit check and planner tier only; the engine refuses "
    "honestly until it has been verified on a GPU machine",
    engine="trellis",
    note="3.1 GiB download plus the DINOv2 ViT-L/14 encoder (about 1.2 GiB, torch hub) and a "
    "CUDA-only runtime stack (spconv, flash-attention or xformers, nvdiffrast) that is not "
    "packaged yet.",
    extra_runtime=(
        "dinov2_vitl14_reg (torch hub)",
        "spconv",
        "xformers or flash-attn",
        "nvdiffrast",
    ),
    requirements=Requirements(
        min_ram_gib=16,
        rec_ram_gib=32,
        min_vram_gib=16,
        rec_vram_gib=24,
        cpu_ok=False,
        gpu_required=True,
        summary="Needs a discrete NVIDIA GPU with 16 GiB of memory or more (24 GiB "
        "recommended) and the CUDA build of PyTorch. There is no CPU path.",
    ),
)

MODELS: dict[str, MeshModel] = {m.name: m for m in (TRELLIS, TRIPOSR)}
TIERS = ("best", "standard")
PRESET_NAMES = ("best", "balanced", "fast")
DEFAULT_MODEL = TRIPOSR.name


def model_for_tier(tier: str) -> MeshModel:
    for m in MODELS.values():
        if m.tier == tier:
            return m
    raise KeyError(tier)


# --- background removal (same models and code path as Image Studio) ---------------------


@dataclass(frozen=True)
class ToolModel:
    name: str
    kind: str  # "background"
    display_name: str
    licence: str
    file: FileSpec
    note: str
    default: bool = False


_GH = "https://github.com/{repo}/releases/download/{tag}/{name}"

TOOL_MODELS: dict[str, ToolModel] = {
    t.name: t
    for t in (
        ToolModel(
            "isnet-general-use",
            "background",
            "IS-Net general use",
            "Apache-2.0",
            FileSpec(
                "isnet-general-use.onnx",
                _GH.format(repo="danielgatis/rembg", tag="v0.0.0", name="isnet-general-use.onnx"),
                "60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a",
                178648008,
                "Apache-2.0",
                "danielgatis/rembg releases (IS-Net, Apache-2.0)",
            ),
            "Dichotomous image segmentation at 1024 px: clean edges on hair and thin structures; "
            "the cutout TripoSR sees. 170 MB.",
            default=True,
        ),
        ToolModel(
            "u2net",
            "background",
            "U²-Net",
            "Apache-2.0",
            FileSpec(
                "u2net.onnx",
                _GH.format(repo="danielgatis/rembg", tag="v0.0.0", name="u2net.onnx"),
                "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491",
                175997641,
                "Apache-2.0",
                "danielgatis/rembg releases (U²-Net, Apache-2.0)",
            ),
            "Classic salient-object model at 320 px: faster, softer edges. 168 MB.",
        ),
    )
}


def default_tool_model(kind: str) -> ToolModel:
    return next(t for t in TOOL_MODELS.values() if t.kind == kind and t.default)


# --- the PyTorch runtime ---------------------------------------------------------------------
# The CPU build ships inside the packaged app and is what `pip install` brings on Windows and
# macOS. The CUDA build is a consented download on machines with an NVIDIA GPU. Only wheels
# for CPython 3.12 (the interpreter the packaged app is built with) are pinned here; pip
# installs resolve their own interpreter's wheel from the same index.

TORCH_VERSION = "2.14.1"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu130"
TORCH_CUDA_LABEL = "cu130"
_PT = "https://download.pytorch.org/whl/{channel}/{name}"


@dataclass(frozen=True)
class RuntimeWheel:
    flavor: str  # cpu | cuda
    platform: str  # win32 | linux | darwin
    machine: str  # x86_64 | arm64
    python: str  # cp312
    file: FileSpec
    self_contained: bool  # True when the wheel carries every native library it needs


RUNTIME_WHEELS: tuple[RuntimeWheel, ...] = (
    RuntimeWheel(
        "cpu",
        "win32",
        "x86_64",
        "cp312",
        FileSpec(
            "torch-2.14.1+cpu-cp312-cp312-win_amd64.whl",
            _PT.format(channel="cpu", name="torch-2.14.1%2Bcpu-cp312-cp312-win_amd64.whl"),
            "c52adc3bd526ff7fc5bdcc58fc4a76e6214d09951b646ffe5256ebd7ba60cef1",
            123996008,
            "BSD-3-Clause",
            "download.pytorch.org/whl/cpu",
        ),
        True,
    ),
    RuntimeWheel(
        "cpu",
        "linux",
        "x86_64",
        "cp312",
        FileSpec(
            "torch-2.14.1+cpu-cp312-cp312-manylinux_2_28_x86_64.whl",
            _PT.format(
                channel="cpu", name="torch-2.14.1%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl"
            ),
            "5a6363570c753812540a05eb82380e329469cbe668643e88111414c12627711f",
            196253677,
            "BSD-3-Clause",
            "download.pytorch.org/whl/cpu",
        ),
        True,
    ),
    RuntimeWheel(
        "cpu",
        "darwin",
        "arm64",
        "cp312",
        FileSpec(
            "torch-2.14.1-cp312-cp312-macosx_14_0_arm64.whl",
            _PT.format(channel="cpu", name="torch-2.14.1-cp312-cp312-macosx_14_0_arm64.whl"),
            "9cf3082d25560efb1eef921871595227c0cf305abb7761ca45b6b988ef6cdd45",
            127211764,
            "BSD-3-Clause",
            "download.pytorch.org/whl/cpu",
        ),
        True,
    ),
    RuntimeWheel(
        "cuda",
        "win32",
        "x86_64",
        "cp312",
        FileSpec(
            "torch-2.14.1+cu130-cp312-cp312-win_amd64.whl",
            _PT.format(channel="cu130", name="torch-2.14.1%2Bcu130-cp312-cp312-win_amd64.whl"),
            "0a09031e93632d14ef49553acd6211ece3ae9cdc0969a369b8c24e3cc4d15be9",
            1990608063,
            "BSD-3-Clause (CUDA libraries: NVIDIA EULA)",
            "download.pytorch.org/whl/cu130",
        ),
        True,  # the Windows CUDA wheel bundles the CUDA DLLs
    ),
    RuntimeWheel(
        "cuda",
        "linux",
        "x86_64",
        "cp312",
        FileSpec(
            "torch-2.14.1+cu130-cp312-cp312-manylinux_2_28_x86_64.whl",
            _PT.format(
                channel="cu130", name="torch-2.14.1%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl"
            ),
            "5359b255ac68c0cabb8293952b38e00f0d1a570815cd87f19b61af702c071410",
            554621169,
            "BSD-3-Clause (CUDA libraries: NVIDIA EULA)",
            "download.pytorch.org/whl/cu130",
        ),
        False,  # Linux CUDA wheels depend on separate nvidia-* wheels: pip only
    ),
)


def normalise_arch(machine: str) -> str:
    m = machine.lower()
    if m in ("amd64", "x86_64", "x64"):
        return "x86_64"
    if m in ("arm64", "aarch64"):
        return "arm64"
    return m


def find_runtime_wheel(flavor: str, platform: str, machine: str) -> RuntimeWheel | None:
    arch = normalise_arch(machine)
    for w in RUNTIME_WHEELS:
        if w.flavor == flavor and w.platform == platform and w.machine == arch:
            return w
    return None


# --- policy ---------------------------------------------------------------------------------

EXCLUDED_BY_POLICY: tuple[tuple[str, str], ...] = (
    (
        "Hunyuan3D 2.x (Tencent)",
        "Tencent Hunyuan Non-Commercial License: non-commercial use only, with territorial "
        "restrictions (not licensed in the EU, UK and South Korea)",
    ),
    (
        "Stable Fast 3D / SF3D (Stability AI)",
        "Stability AI Community License: free only below a revenue threshold",
    ),
    (
        "Stable Zero123 and Stable Point Aware 3D",
        "Stability AI Community License: free only below a revenue threshold",
    ),
)

LICENCES: tuple[dict[str, str], ...] = (
    {"component": "Presidium Shape Studio", "licence": "Apache-2.0"},
    {"component": "TripoSR weights (stabilityai/TripoSR)", "licence": "MIT"},
    {"component": "TripoSR code (vendored, VAST-AI-Research/TripoSR)", "licence": "MIT"},
    {"component": "Attention blocks derived from Hugging Face diffusers", "licence": "Apache-2.0"},
    {"component": "DINO ViT-B/16 architecture (Facebook Research)", "licence": "Apache-2.0"},
    {"component": "TRELLIS image-large weights (microsoft/TRELLIS-image-large)", "licence": "MIT"},
    {"component": "PyTorch", "licence": "BSD-3-Clause"},
    {"component": "scikit-image (marching cubes)", "licence": "BSD-3-Clause"},
    {"component": "IS-Net and U²-Net background models", "licence": "Apache-2.0"},
    {"component": "ONNX Runtime", "licence": "MIT"},
    {"component": "model-viewer web component (Google)", "licence": "Apache-2.0"},
)
