# PyInstaller spec for the desktop app on Windows, macOS and Linux. Build from the repository
# root:  pyinstaller packaging/ShapeStudio.spec --noconfirm
# Produces dist/ShapeStudio/ (one folder; zip it for distribution) and, on macOS, the
# "Shape Studio.app" bundle next to it. A one-file build was
# rejected on purpose: it unpacks itself into a temp folder on every start, which is slow
# and looks more suspicious to endpoint protection than a plain folder.

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

WIN = sys.platform == "win32"
MAC = sys.platform == "darwin"

hidden = (
    collect_submodules("uvicorn")
    + collect_submodules("starlette")
    + collect_submodules("mcp", filter=lambda n: not n.startswith("mcp.cli"))
    + collect_submodules("shapestudio")
    + collect_submodules("skimage.measure")  # lazy-loaded upstream; PyInstaller cannot see it
    + ["truststore", "PIL.PngImagePlugin", "PIL.JpegImagePlugin", "PIL.WebPImagePlugin", "PIL.IcoImagePlugin",
       "skimage", "skimage.measure", "scipy.ndimage", "scipy.spatial", "scipy.sparse.csgraph"]
)
datas = [(str(SRC / "shapestudio" / "static"), "shapestudio/static")]
datas += collect_data_files("truststore")
datas += collect_data_files("mcp")
datas += collect_data_files("skimage")  # lazy_loader stubs and data
datas += collect_data_files("torch", include_py_files=False)
datas += [(str(ROOT / "packaging" / "ShapeStudio.ico"), ".")]
if WIN or MAC:
    # the native window; Linux runs in the browser (packaging/launcher.py)
    hidden += collect_submodules("webview")
    datas += collect_data_files("webview")
if WIN:
    hidden += collect_submodules("pystray") + ["clr"]
try:
    import onnxruntime  # noqa: F401

    hidden += collect_submodules("onnxruntime")
    datas += collect_data_files("onnxruntime")
except ImportError:
    pass

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(SRC)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "pandas", "IPython", "pytest", "playwright", "mcp.cli", "typer", "transformers"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ShapeStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=not (WIN or MAC),  # a real window; logs go to <data>/logs/app.log
    icon=str(ROOT / "packaging" / ("ShapeStudio.ico" if WIN else "store-assets/display-1x1-300.png")),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="ShapeStudio",
)
if MAC:
    app = BUNDLE(
        coll,
        name="Shape Studio.app",
        icon=str(ROOT / "packaging" / "store-assets" / "display-1x1-300.png"),
        bundle_identifier="com.hivemind.shapestudio",
        info_plist={
            "CFBundleDisplayName": "Presidium Shape Studio",
            "CFBundleShortVersionString": os.environ.get("SHAPESTUDIO_VERSION", "0.0.0"),
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "13.0",
        },
    )
