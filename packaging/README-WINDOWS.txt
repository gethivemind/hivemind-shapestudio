Presidium Shape Studio - Windows app
====================================

1. Unzip this folder anywhere (for example C:\Apps\ShapeStudio), or run the Setup.exe.
2. Run ShapeStudio.exe. The studio opens in a window of its own. Closing it during a job
   offers to keep working in the background (tray icon; click it to come back). Starting it
   again focuses the running window. ShapeStudio.exe --browser uses your browser instead.
   Logs: %LOCALAPPDATA%\shapestudio\logs\app.log
3. First time: Home shows a getting-started checklist. Download TripoSR (1.56 GiB, MIT) and a
   background-removal model (170 MB) from there or from Settings. Downloads are hash-verified
   and need your consent. The CPU build of PyTorch is inside this folder; machines with an
   NVIDIA GPU are offered the CUDA build (about 1.9 GiB) under Settings.

Everything runs on this machine. Nothing is uploaded.

Windows may warn
----------------
This build is not code-signed yet, so Windows SmartScreen shows "Windows protected your PC"
the first time you run a downloaded copy. Click "More info", then "Run anyway". Check the
download against the .sha256 file on the release page if you want to be sure it is the
official build.

Command line
------------
ShapeStudio.exe cli doctor
ShapeStudio.exe cli reconstruct C:\pictures\chair.png --out C:\meshes
ShapeStudio.exe cli --help

Licence: Apache-2.0 (LICENSE, NOTICE). The name and logo are trademarks (TRADEMARK.md).
Made with love by Ratul Roy (royal-programmer). https://presidiumdev.github.io/presidium-shapestudio/
