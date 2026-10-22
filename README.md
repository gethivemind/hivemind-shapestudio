# Presidium Shape Studio

<img src="site/assets/shape-tile.svg" width="96" height="96" alt="The Presidium Shape Studio mark" align="right">

**Website:** https://presidiumdev.github.io/presidium-shapestudio/ · **PyPI:** https://pypi.org/project/presidium-shapestudio/ · **Releases:** https://github.com/presidiumdev/presidium-shapestudio/releases

A free, local image-to-3D studio and MCP server. One picture in, a textured mesh out (GLB
and OBJ with vertex colours, plus a turntable PNG), with permissively licensed models on your
own machine. A desktop app for Windows, macOS and Linux, a web UI, a command line, and an MCP
server for Presidium, Claude, Cursor or any other agent. It runs on the CPU if that is all
there is, and uses an NVIDIA GPU when you have one.

Part of the Presidium family, sibling of [Presidium Image Studio](https://github.com/gethivemind/hivemind-imagestudio),
built on the same architecture and the same rules. Design: Hivemind ADR-051; brand: ADR-065 and 066. Licence:
Apache-2.0.

## The policy

**Quality is a setting, never a consequence.** The default preset is `best`: the finest
surface grid on the strongest permissively licensed model that fits your machine. The
machine decides how long that takes (cores, GPU, memory), never how good the result is.
Nothing downgrades silently to be faster: the studio estimates the time and tells you.

**Faster only with consent.** Presets `balanced` and `fast` exist and must be chosen
explicitly. Each states its trade-off in words, in the tool description and in
`capabilities()`:

| Preset (TripoSR) | Surface grid | Trade-off |
| --- | --- | --- |
| `best` (default) | 256³ | The finest surface the field resolves; most triangles; longest extraction. |
| `balanced` | 192³ | About half the extraction work; slightly softer small details. |
| `fast` | 128³ | A preview-grade surface in a fraction of the time; thin parts may break up. |

The network pass is identical at every preset; only the surface extraction changes. Nothing
else is lowered on its own.

**Two tiers, chosen by hardware, never silently.** The `standard` tier is TripoSR (MIT): it
runs on a CPU. The `best` tier is TRELLIS image-large (MIT): clearly higher fidelity, and it
needs a discrete NVIDIA GPU with 16 GB of memory or more. On a machine without that GPU the
plan says so in `fit_note` ("Running TripoSR because TRELLIS needs an NVIDIA GPU and this
machine has none. This is a hardware fit, not a speed choice."), and asking for `best` by
name is refused with the reason instead of being swapped for something else.

**Permissive licences only.** TripoSR weights and code: MIT. TRELLIS weights: MIT. PyTorch:
BSD. IS-Net and U²-Net background models: Apache-2.0. Excluded by policy, whatever their
quality: Hunyuan3D 2.x (Tencent Hunyuan Non-Commercial License, with territorial
restrictions), Stable Fast 3D, Stable Zero123 and Stable Point Aware 3D (Stability AI
Community License, free only below a revenue threshold).

**Offline after setup.** The models and, if you want it, the CUDA build of PyTorch download
once, each after explicit consent (`--yes` or `consent=true`), into the studio's own data
folder, pinned by SHA-256. After that nothing touches the network; there are no keys or
accounts.

**Measured, not promised.** Every time estimate comes from runs on your machine (per model,
per device) and is labelled `not measured yet` until the first one. This README contains no
timing claim this project has not measured. As of 0.1.6 the real TripoSR weights have been
loaded on a user's GPU laptop (the run surfaced two bugs, both fixed in 0.1.6); a complete
measured reconstruction is still to be reported (see Status).

## Status, stated plainly

| Part | State |
| --- | --- |
| Queue, planner, tiers, consented downloads, UI, CLI, MCP server, GLB/OBJ writer, turntable | Proven by the test suite with a fake engine (no weights needed) |
| Vendored TripoSR code (no `torchmcubes`, `transformers`, `einops`, `omegaconf`) | Runs end to end on random weights; the DINO ViT matches the reference implementation exactly at 224 px and to 1e-2 at 512 px (interpolation recipe); parameter names match the published checkpoint layout |
| TripoSR with the real `model.ckpt` | **Loaded on a user's machine (0.1.5, GTX 1660 Ti laptop); the run then failed in the background remover, fixed in 0.1.6.** A complete measured reconstruction is still to be reported; the first one writes the machine's profile automatically |
| TRELLIS engine | **Not integrated.** Catalogue entry, hash pins, fit check and planner tier exist; the engine refuses with the reason until verified on a 16 GB NVIDIA GPU |
| CUDA runtime download for the packaged Windows app | Download and unpack verified by a user (0.1.5). Loading it was broken until 0.1.6 (the launcher never activated it); Settings now shows whether the unpacked build loaded, **GPU run still to be confirmed** |
| Minimum and recommended requirements per model | Stated in the catalogue and checked against the machine (0.1.6): TripoSR 8 GiB RAM minimum, 16 recommended, NVIDIA 4 GiB optional; TRELLIS NVIDIA 16 GiB minimum, 24 recommended, no CPU path |
| Windows installer, portable zip, MSIX, macOS and Linux builds | Built by the release workflow on GitHub's runners |

## Sizes, stated plainly

| Download | Size | Licence | When |
| --- | --- | --- | --- |
| PyTorch CPU build | 124 MB (Windows), 196 MB (Linux), 127 MB (macOS arm64) | BSD-3-Clause | inside the packaged app; `pip` brings it on Windows and macOS. On Linux the default PyPI wheel is the CUDA build (555 MB plus about 2.5 GB of NVIDIA libraries); add `--extra-index-url https://download.pytorch.org/whl/cpu` for the CPU build |
| TripoSR (`model.ckpt` + `config.yaml`) | 1.56 GiB | MIT | consented download, first use |
| IS-Net general use (background removal) | 170 MB | Apache-2.0 | consented download, first use (U²-Net, 168 MB, as an alternative) |
| TRELLIS image-large | 3.07 GiB | MIT | consented download; engine not integrated yet |
| PyTorch CUDA build (cu130) | 1.9 GiB (Windows, self-contained), 0.5 GiB plus NVIDIA wheels (Linux, pip only) | BSD-3-Clause, NVIDIA EULA for the CUDA libraries | offered on machines with an NVIDIA GPU, never automatic |

Memory: TripoSR's weights are 1.68 GB in fp32; the planner budgets about 4.2 GiB of RAM for a
256³ grid on the CPU (weights plus working memory) and refuses over-budget runs unless you
unlock them. TRELLIS is budgeted at 16 GiB of GPU memory, the authors' figure.

## Install

**Windows app:** download `ShapeStudio-<version>-Setup.exe` from the
[latest release](https://github.com/presidiumdev/presidium-shapestudio/releases/latest) and run
it: Start menu entry, uninstall from Settings → Apps, no administrator rights. Prefer
portable? `…-windows-x64.zip` is the same app: unzip, run `ShapeStudio.exe`. The build is not
code-signed yet, so SmartScreen asks once per download ("More info" → "Run anyway"); the
`.sha256` file next to each download lets you verify it. `ShapeStudio.exe cli doctor` reaches
the command line. Logs: `%LOCALAPPDATA%\shapestudio\logs\app.log`.

**macOS (Apple silicon):** `ShapeStudio-<version>-macos-arm64.zip`; unzip, right-click "Shape
Studio.app" → Open the first time (not notarized yet). **Linux (x86_64):**
`ShapeStudio-<version>-linux-x86_64.tar.gz`; unpack and run `ShapeStudio`, which opens the
studio in your browser. Details in `packaging/README-LINUX-MACOS.txt`.

**Everywhere else** (and for the MCP server and CLI), Python 3.11 or newer:

```bash
pipx install presidium-shapestudio
```

or the same wheel with `pip install` / `uv tool install`. The PyPI package is not published yet:
it will be registered once as `presidium-shapestudio` when its publisher exists; until then the wheel on the GitHub release is the Python distribution. PyTorch is a
dependency and comes along (see Sizes). Then:

```bash
shapestudio doctor
```

```bash
shapestudio download-model triposr --yes
```

```bash
shapestudio download-model isnet-general-use --yes
```

```bash
shapestudio reconstruct chair.png --out meshes/
```

`doctor` shows the hardware found, the PyTorch runtime (CPU or CUDA, and whether the CUDA build
can be installed here), both tiers with whether they are downloaded and whether they fit, and
the expected time per preset, each labelled measured or not. `reconstruct` queues one job,
follows it, and copies the GLB, OBJ and turntable into `--out`. `shapestudio ui` opens the
studio in your browser; `shapestudio benchmark` runs one real reconstruction of a built-in test
image and writes `profile.json` so every later estimate is measured.

On a machine with an NVIDIA GPU:

```bash
shapestudio install-runtime --cuda --yes
```

installs the pinned CUDA build of PyTorch (pip into the same interpreter; in the packaged
Windows app a hash-verified wheel unpacked under the data folder, or Settings → Install CUDA
build). Then restart the studio. It is offered, never automatic, and `doctor` and Settings
say which build actually loaded and from where.

The GitHub Releases of this repository and the `presidium-shapestudio` package on PyPI are the
official builds (see NOTICE and TRADEMARK.md).

## The app

- **Home**: on a fresh machine a setup card names the model recommended for this hardware
  and the background model, with sizes, and one button downloads both; then a checklist with
  buttons (first mesh, try the sample box), the models sorted by fit with minimum and
  recommended requirements against this machine, and the policy.
- **Create**: drop one picture or several, or pick a built-in sample (sphere, box, mug);
  choose among the downloaded models (the others are named below with their fit and size),
  a preset, a surface grid capped at what fits this machine's memory; see the expected time
  and a load meter before you commit. One compact note explains the choice, with buttons for
  what would help (download the model, install the CUDA build). The queue shows every job
  with its picture, phase and an honest ETA; a failed job offers "Report".
- **Gallery**: every mesh with a 3D viewer ([model-viewer](https://modelviewer.dev/),
  Apache-2.0, loaded from jsdelivr pinned to 4.1.0; offline the PNG turntable is shown
  instead), downloads (GLB, OBJ, turntable), mesh tools (STL for printing scaled to a size
  in mm, lighter mesh, smooth, flip normals, scale; each a new file next to the original),
  favourites, notes, projects (named series with one-zip export), select several for a zip
  or the trash, trash with restore.
- **Settings**: hardware, which PyTorch loaded and from where, the state of the CUDA build
  (unpacked and waiting for a restart, active, or did not load and why) with Restart now;
  benchmark and the measured profile (written by your first reconstruction anyway); the
  components in three groups, Installed, Available to install and Runtime, each model with
  its minimum and recommended requirements against this machine, licence, size, status and
  Install/Show/Remove (install dialogs show the files, sizes, sources and SHA-256; a model
  that does not fit asks "Download anyway"); preferences (device, memory limit, threads,
  over-budget unlock, pause), projects, trash, "Report a problem", about.

Dark and light themes in the Presidium palette (Shape Studio coral #D5582F), phone-friendly layout, the Presidium spinner while a job runs.

## MCP tools

`shapestudio serve` speaks MCP over stdio. Every tool declares annotations (read-only,
destructive, idempotent, open-world) and its description is written for the agent.

| Tool | What it does |
| --- | --- |
| `capabilities()` | hardware, runtime, tiers with fit and download state, presets with expected times, policy, licences, exclusions |
| `plan(tier?, preset, resolution?)` | dry run: model, device, grid, estimate with its basis, load meter, fit note, warnings |
| `reconstruct(image, tier?, preset, resolution?, remove_background, foreground_ratio?)` | queue a job; returns the id and estimate at once |
| `job_status(job_id)` | phase, progress, honest summary, ETA or null, paths when done |
| `job_result(job_id, as_base64?, format=glb\|obj)` | GLB or OBJ path, turntable, counts; bytes on request |
| `jobs()` | all jobs, newest first |
| `cancel_job(job_id)` | stop a queued or running job |
| `download_model(name, consent)` | files, sizes, hashes and licence first; download only with consent=true |

Client configs for Claude Desktop, Claude Code, Cursor and VS Code are in `examples/mcp`.

## Licences

| Component | Licence |
| --- | --- |
| Presidium Shape Studio | Apache-2.0 |
| TripoSR weights (stabilityai/TripoSR) | MIT |
| TripoSR code, vendored under `src/shapestudio/engine/tsr` (VAST-AI-Research/TripoSR) | MIT |
| Attention blocks derived from Hugging Face diffusers | Apache-2.0 |
| DINO ViT-B/16 architecture (Facebook Research); reimplemented here | Apache-2.0 |
| TRELLIS image-large weights (microsoft/TRELLIS-image-large) | MIT |
| PyTorch | BSD-3-Clause (CUDA libraries inside the CUDA build: NVIDIA EULA) |
| scikit-image (marching cubes) | BSD-3-Clause |
| IS-Net and U²-Net background models | Apache-2.0 |
| ONNX Runtime | MIT |
| model-viewer web component (Google) | Apache-2.0 |

The code is free (Apache-2.0, `LICENSE`, `NOTICE`, `third_party/`). The names "Hivemind",
"Presidium Shape Studio" and "Presidium Shape Studio" and the logo identify the official
project and are not covered by the licence; the policy is in `TRADEMARK.md`. Official builds
carry the line "Presidium Shape Studio · made with ♥ by Ratul Roy (royal-programmer)".

## Development

```bash
uv venv && uv pip install -e ".[dev,verify]"
```

```bash
uv run pytest -q
```

The suite (about 50 tests) runs with a fake engine and needs no weights; the vendored
TripoSR is exercised on random weights, and the ViT cross-check runs when `transformers` is
installed. `ruff check src tests` and `ruff format --check src tests` are the lint. The data
folder is `%LOCALAPPDATA%\shapestudio` (Windows), `~/Library/Application Support/shapestudio`
(macOS) or `~/.local/share/shapestudio` (Linux); override with `SHAPESTUDIO_DATA_DIR`.

## Roadmap

1. First complete TripoSR run reported from a GPU machine (0.1.6 fixed what stopped the first
   attempt); measured numbers into this README and the site.
2. TRELLIS engine integration behind the existing tier (CUDA stack packaging).
3. Texture baking (UV atlas) in addition to vertex colours.
4. Multi-view input; quadric decimation and hole filling next to the 0.1.6 mesh tools.
