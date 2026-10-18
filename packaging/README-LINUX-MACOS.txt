Shape Studio by Hivemind - macOS and Linux builds
==================================================

macOS (Apple silicon)
---------------------
1. Unzip, drag "Shape Studio.app" to Applications (or run it from anywhere).
2. First start: the app is not notarized yet, so macOS says it "cannot be opened because the
   developer cannot be verified". Right-click (or Control-click) the app, choose Open, then
   Open again. Or: System Settings > Privacy & Security > Open Anyway. Check the download
   against the .sha256 file on the release page if you want to be sure it is the official build.
3. The studio opens in a window of its own (the system WebKit view). There is no tray icon on
   macOS: closing the window during a job asks whether to stop it.
   Logs: ~/Library/Application Support/shapestudio/logs/app.log
4. TripoSR runs on the CPU here (PyTorch's CPU build is inside the app; Apple GPUs are not
   used yet). The model (1.56 GiB) is downloaded by the studio on first use, hash-verified,
   after your consent.

Linux (x86_64, glibc 2.28 or newer)
-----------------------------------
1. Unpack:   tar -xzf ShapeStudio-<version>-linux-x86_64.tar.gz
2. Run:      ./ShapeStudio-<version>/ShapeStudio
   The studio starts its local server and opens in your default browser; Ctrl+C stops it.
   A native window would need GTK or Qt that this build cannot carry, so the browser is the
   honest choice here (the same interface, same quality). --no-browser runs the server only.
   Logs: ~/.local/share/shapestudio/logs/app.log
3. The CPU build of PyTorch is inside. For an NVIDIA GPU, the packaged Linux build cannot swap
   in the CUDA runtime (the Linux CUDA wheels depend on separate nvidia-* packages); install
   with pipx instead:  pipx install hivemind-shapestudio  then  shapestudio install-runtime --cuda

Everything runs on this machine. Nothing is uploaded.

Command line (both)
-------------------
./ShapeStudio cli doctor
./ShapeStudio cli reconstruct ~/pictures/chair.png --out ~/meshes

Prefer Python?  pipx install hivemind-shapestudio   then   shapestudio ui
