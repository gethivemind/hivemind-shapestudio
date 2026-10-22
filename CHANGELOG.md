# Changelog

All notable changes to Presidium Shape Studio. The format follows Keep a Changelog; versions
follow semantic versioning. Official builds: GitHub Releases and the `presidium-shapestudio`
package on PyPI.

## 0.1.6 — 2026-10-22

The first run on a real GPU laptop (GTX 1660 Ti, 6 GiB; 16 GiB RAM) found two blocking bugs
and a screen full of things that misled. This release fixes all of them and adds what that
first user needed.

### Fixed
- **Every real reconstruction failed** with `EngineResult.__init__() got an unexpected keyword
  argument 'output'`: the background remover (copied from Image Studio) returned Image Studio's
  result type. It has its own `BackgroundResult` now, and a test runs the real remover with a
  stub ONNX session.
- **The CUDA build never loaded in the desktop app.** "Install CUDA build" downloaded and
  unpacked the wheel, but the window launcher never called `activate_runtime`, so after the
  restart the bundled CPU build loaded again. The launcher activates the override first thing,
  logs which PyTorch loaded and from where, and Settings shows the override state (unpacked,
  restart needed / active / did not load, with the reason).
- Auto tier chose a downloaded model that can never run here (TRELLIS on a 6 GiB GPU) over one
  that fits but is not downloaded, and then refused. It now prefers the model that fits and says
  what to download. Fit reasons name the real blocker first (GPU memory before the missing CUDA
  build) and read as sentences.
- Unexpected engine errors are reported as "internal error … please report it", not a bare
  traceback fragment.
- Uploaded pictures kept their generated file name; the job title and gallery now show the
  name you uploaded.

### Changed
- In-app hints no longer tell you to run `shapestudio install-runtime --cuda` (the Store and
  installer apps put no such command on PATH); they are buttons: Download TripoSR, Install the
  CUDA build, Download the background model.
- Create lists only downloaded models in the Model menu; models that are not downloaded are
  named underneath with their fit and size, with a link to Settings. The surface-grid slider
  stops at the finest grid that fits this machine's memory.
- One compact note on Create instead of four stacked banners; further notes behind "more".
- Settings splits the table into Installed, Available to install and Runtime. Every model shows
  minimum and recommended requirements (RAM, NVIDIA GPU memory, CUDA build) against this
  machine. Installing a model that does not fit asks "Download anyway", with the reason.
- The CUDA build is one row like the others: Install… / Remove / Restart now, with the steps
  written out (download, verify, unpack, restart, then the GPU is used) and the wheel's hash.
- Home sorts the models by fit and marks the one recommended for this machine; cards carry the
  requirements table. Checklist steps have buttons.
- TripoSR's status line says what is true: it runs end to end in the project's tests and your
  first real run makes the estimates measured; "measured on this machine: N runs" once it has.

### Added
- First-run setup card on Home: detects the machine, names the recommended model and
  background model with sizes, one "Download and set up" button, then "Try the sample box".
- Three built-in sample pictures (sphere, box, mug), no download needed.
- Batch mode: drop several pictures, one job each.
- The first successful reconstruction writes the measured profile; no benchmark click needed.
- Restart from inside the desktop app (Settings, and after the CUDA build is unpacked).
- Mesh tools in the viewer: STL for printing (optionally scaled to a size in mm), lighter mesh
  (vertex clustering), smooth, flip normals, scale. Each writes a new file next to the original.
- "Report a problem": one text with version, machine, runtime, the failing job and the log
  tail, shown in full before you open a prefilled issue on presidium-releases.
- `capabilities()` and `components` carry `requirements`, `requirements_met`,
  `recommended`, `max_resolution_fits`, `setup` and `samples`; `plan` carries `hints`.

## 0.1.5 — 2026-10-21

### Fixed
- The desktop window crashed at start ("build_app() got an unexpected keyword argument
  'on_remote_change'"): the launcher still called the web app builder with Image Studio's
  remote-access hook. Affected every 0.1.x Windows, macOS and Linux app build; the CLI, web UI
  and MCP server were unaffected. A test now starts the launcher's server.
- The window's close question while a job runs now has its dialog in the page.

### Changed
- Support links point at the family's single public issue tracker,
  https://github.com/presidiumdev/presidium-releases/issues.

## 0.1.4 — 2026-10-20

### Changed
- Store MSIX built with the real Partner Center identity (repository variables set); no code changes.

## 0.1.3 — 2026-10-19

### Changed
- Distribution renamed to `presidium-shapestudio` and published to PyPI for the first time
  (`pipx install presidium-shapestudio`). Module and command stay `shapestudio`; a
  `presidium-shapestudio` command is added. Store posters, hero and box art on the family template.

## 0.1.2 — 2026-10-19

### Changed
- Repository moved to presidiumdev/presidium-shapestudio; website at
  https://presidiumdev.github.io/presidium-shapestudio/. Every release now builds both the Store
  MSIX (placeholder identity until Partner Center assigns one) and the dev-signed sideload MSIX.
- Brand: the product is now **Presidium Shape Studio** (ADR-065/066): new mark (six seats
  around one table, a cube on the table), coral #D5582F, lockup "Presidium" medium + "Shape
  Studio" regular, loading spinner. Module, command, data folder and release file names are
  unchanged; the PyPI name will be `presidium-shapestudio` when the package is first published.

## 0.1.1 — 2026-10-18

### Fixed
- Release builds pin the CPU build of PyTorch on every platform. 0.1.0's Linux tarball had pulled
  PyPI's CUDA build and weighed 3 GB; it was not published. Windows and macOS were unaffected.
- CLI output survives cp1252 consoles; the engine's progress text is plain ASCII.
- Settings offers the CUDA runtime only when an NVIDIA GPU is present; phone layout.

## 0.1.0 — 2026-10-18

First release, built as a sibling of Presidium Image Studio (then "Image Studio by Hivemind") on the same architecture.

### Added
- Image-to-3D with TripoSR (MIT) as the `standard` tier: background removal (IS-Net or U²-Net
  through ONNX Runtime), one network pass, marching cubes on a 256³/192³/128³ grid (presets
  `best`/`balanced`/`fast`), GLB and OBJ with vertex colours, a CPU-rendered turntable PNG.
  The TripoSR code is vendored without `torchmcubes`, `transformers`, `einops` or
  `omegaconf`, so Windows needs no compiler; the DINO ViT is reimplemented with the published
  parameter names and cross-checked against the reference.
- TRELLIS image-large (MIT) as the `best` tier: pinned, hash-verified catalogue entry, fit
  check (NVIDIA GPU with 16 GB) and planner tier. The engine is a labelled stub that refuses
  with the reason until it is verified on a GPU machine.
- Planner that picks the strongest tier that is downloaded and fits, states why when that is
  not the top tier, refuses GPU-only tiers on machines without the GPU, and locks over-budget
  runs behind an explicit unlock.
- Per-machine calibration of load, network and grid time; estimates labelled "not measured
  yet" until the first run; `benchmark` runs one real reconstruction and writes a profile.
- File-backed job queue with one worker per data folder, cross-process cancel, pause, exit
  handover, re-queue after a crash.
- PyTorch runtime handling: CPU build bundled in the packaged app; consented CUDA install
  (pip for Python installs, a hash-verified wheel unpacked under the data folder for the
  packaged Windows app, loaded through a meta-path finder).
- Web UI (Home, Create, Gallery with model-viewer and PNG fallback, Settings), CLI (`doctor`,
  `install-runtime`, `download-model`, `reconstruct`, `plan`, `benchmark`, `jobs`, `ui`,
  `serve`), MCP server (`capabilities`, `plan`, `reconstruct`, `job_status`, `job_result`,
  `jobs`, `cancel_job`, `download_model`) with tool annotations.
- Projects with zip export, favourites, notes, trash with restore, batch export.
- Packaging: PyInstaller windowed app, Inno Setup installer, MSIX, macOS and Linux builds,
  release workflow with PyPI trusted publishing, CI, GitHub Pages site.
