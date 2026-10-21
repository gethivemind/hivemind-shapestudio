# Microsoft Store listing: Presidium Shape Studio

Copy-paste material for Partner Center. Nothing here claims a measurement the project has not
made; the first real reconstruction on a GPU machine is still pending (see README → Status).

## Product name

Presidium Shape Studio

## Short description (≤ 200 characters)

Turn one picture into a 3D mesh on your own PC. GLB and OBJ with colours, a turntable preview, no account, no upload. Quality is a setting, never a consequence.

(176 characters)

## Full description

Presidium Shape Studio is a free, local image-to-3D studio. Drop in a photo or a rendering of
one object and get a textured mesh back: a GLB for viewers and game engines, an OBJ for
modelling tools, and a turntable preview you can judge at a glance. Everything runs on the
machine in front of you. There is no account, no upload and no subscription.

The default is the best your machine can do. The "best" preset extracts the finest surface the
model resolves; "balanced" and "fast" exist only when you choose them, and each says what it
trades. If a tier needs hardware you do not have, the studio says so instead of quietly
swapping something smaller in. Every time estimate comes from runs on your own machine and is
labelled "not measured yet" until the first one.

Two tiers, chosen by hardware. The standard tier is TripoSR (MIT licence) and runs on a CPU or
any NVIDIA GPU. The best tier is TRELLIS (MIT licence) for NVIDIA GPUs with 16 GB of memory;
this first release lists it, checks whether your machine fits it and refuses honestly until the
engine is verified.

Permissive licences only. Models under non-commercial or revenue-gated terms are excluded by
policy and named as such. The models download once, with your consent, pinned by SHA-256, and
the studio works offline afterwards.

Also inside: a gallery with a 3D viewer, favourites, notes and projects with one-zip export; a
command line; and a Model Context Protocol server so Claude, Cursor, VS Code or Presidium can
make meshes for you. Dark and light themes, phone-friendly layout.

Part of the Presidium family, alongside Presidium Image Studio. Open source under Apache-2.0.

## Feature bullets (up to five)

1. One picture in, a GLB and OBJ mesh with colours out, entirely on your PC
2. Best quality by default; faster presets only when you choose them, with the trade-off stated
3. Two tiers chosen by your hardware, never swapped silently; CPU or NVIDIA GPU
4. Permissive licences only (TripoSR, TRELLIS); models download once with your consent, then offline
5. Gallery with 3D viewer, projects and exports, a command line, and an MCP server for AI agents

## URLs

| Field | Value |
| --- | --- |
| Website | https://presidiumdev.github.io/presidium-shapestudio/ |
| Privacy policy | https://presidiumdev.github.io/presidium-shapestudio/privacy.html |
| Support | https://github.com/presidiumdev/presidium-releases/issues |
| Source | https://github.com/presidiumdev/presidium-shapestudio |

## Category and search terms

Category: Developer tools (alternative: Photo & video). Subcategory: none.

Search terms (max 7): `3D`, `image to 3D`, `mesh`, `GLB`, `TripoSR`, `MCP server`, `offline AI`

## Age rating (IARC questionnaire)

The app is a creation tool with user-supplied input and no online interaction. Answer "No" to
every content question (violence, fear, sexuality, language, controlled substances, gambling,
simulated gambling, nudity). Interactive elements: no user-to-user interaction, no sharing of
personal information, no location sharing, no in-app purchases. Expected result: Everyone /
PEGI 3.

## Properties and declarations

- `runFullTrust` justification (≈ 300 characters): "A desktop app packaged with PyInstaller. It
  runs the PyTorch inference engine locally, writes meshes and settings under the user's
  %LOCALAPPDATA%\shapestudio folder, listens on localhost for its own window and its MCP server,
  and downloads model weights from Hugging Face only after the user consents."
- `internetClient`: model and runtime downloads after consent; the 3D viewer script from the
  jsdelivr CDN. No telemetry. Privacy policy URL above is required because of this capability.
- Pricing: free. No in-app purchases.
- Supported platforms: Windows 10 version 17763 and later, x64. Minimum RAM stated as 8 GB
  (TripoSR on the CPU is budgeted at about 4.2 GiB; the planner refuses over-budget runs).

## Screenshots

PNG, 1440 × 900, captured from the running app with Playwright (`scripts/store_screenshots.py`),
committed under `packaging/store-screenshots/`:

| File | Shows |
| --- | --- |
| `01-home.png` | Home: getting-started checklist, the two tiers with their state on this machine, the policy |
| `02-create.png` | Create: a picture dropped, tier and preset, expected time with its basis, load meter, honest notes |
| `03-gallery.png` | Gallery: meshes with vertex and face counts, favourites, project filter |
| `04-viewer.png` | Viewer: the 3D model, mesh stats, downloads (GLB, OBJ, turntable), note and project |
| `05-settings.png` | Settings: hardware and runtime, benchmark, every model with licence, size and status |

The meshes in the screenshots are demonstration objects produced by the studio's own mesh
pipeline (marching cubes on a synthetic field), not TripoSR reconstructions; the caption in the
listing should say "demonstration objects" until real reconstructions exist.

## Store assets

- App icon and tiles: `packaging/msix/Assets` (generated by `scripts/make_artwork.py` from the
  Presidium brand kit; coral #D5582F tile, light mark).
- 1:1 display logos: `packaging/store-assets/display-1x1-{71,150,300}.png`.
- Posters, hero images and box art (family template, ADR-066; `python -m shapestudio.brand`):
  `packaging/store-assets/poster-9x16-720x1080.png`, `poster-9x16-1440x2160.png`,
  `hero-16x9-1920x1080.png`, `hero-16x9-3840x2160.png`, `box-art-1x1-1080.png`, `box-art-1x1-2160.png`.

## Package

Upload `ShapeStudio-<version>-windows-x64-store.msix` from the GitHub release. Until the
repository variables `MSIX_IDENTITY_NAME`, `MSIX_PUBLISHER` and `MSIX_PUBLISHER_DISPLAY` hold the
values Partner Center shows under Product identity, the package carries a PLACEHOLDER identity
and the Store will reject it; set the variables and re-tag (or re-run the Windows job).
