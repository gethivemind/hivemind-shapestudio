# Installer and the Microsoft Store

The release workflow produces three installable forms of the Windows app:

| File | Signed by | For |
| --- | --- | --- |
| `ShapeStudio-<ver>-Setup.exe` | nobody (yet) | the normal installer: Start menu, search, Pin to Start, uninstall; SmartScreen asks once |
| `ShapeStudio-<ver>-windows-x64.zip` | nobody (yet) | portable: unzip and run; SmartScreen asks once |
| `ShapeStudio-<ver>-windows-x64-store.msix` | Microsoft, on submission | upload to Partner Center |

Without the three repository variables below, a `…-sideload.msix` with a throw-away
certificate is built instead so the packaging path is exercised on every release.

## Store steps, once

1. In Partner Center (the same free individual account as Image Studio), create the app with
   the name **Shape Studio by Hivemind**. Reserving the name is the trademark-relevant step.
2. Open *Product management → Product identity* and copy `Package/Identity/Name`,
   `Package/Identity/Publisher` and `Package/Properties/PublisherDisplayName`.
3. Add them to this repository as Actions **variables**: `MSIX_IDENTITY_NAME`,
   `MSIX_PUBLISHER`, `MSIX_PUBLISHER_DISPLAY`. Every release then builds the Store package.

## Per release

1. Download `ShapeStudio-<ver>-windows-x64-store.msix` from the GitHub Release.
2. Partner Center → the app → new submission → *Packages* → upload the `.msix`.
3. Listing: category *Developer tools* or *Photo & video*; the description is the hero text on
   the website; `internetClient` capability means a privacy policy URL is required:
   https://gethivemind.github.io/hivemind-shapestudio/privacy.html. `runFullTrust`
   justification: a PyInstaller desktop app that runs PyTorch locally and writes meshes under
   `%LOCALAPPDATA%\shapestudio`.

## Status

Not submitted yet (0.1.0). The app is about 400 MB unpacked because PyTorch's CPU build and
scikit-image are inside; the Store accepts packages up to 25 GB.
