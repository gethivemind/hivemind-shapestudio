"""Pack dist/ShapeStudio (the PyInstaller folder) into an MSIX installer.

    python packaging/build_msix.py --version 0.3.2 [--store | --sideload]

--store    identity from environment variables MSIX_IDENTITY_NAME, MSIX_PUBLISHER,
           MSIX_PUBLISHER_DISPLAY (the values Partner Center shows under Product identity).
           The package is left unsigned: the Store signs it on submission.
--sideload (default) identity Hivemind.ShapeStudio / CN=Hivemind Shape Studio Dev, signed with
           a self-signed certificate created on the spot (exported next to the package as
           ShapeStudio-dev.cer). To install: double-click the .cer, "Install Certificate",
           Local Machine, "Trusted People"; then double-click the .msix. For testing only.

Needs the Windows 10/11 SDK (makeappx.exe, signtool.exe); found automatically.
"""

# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
APP_DIR = DIST / "ShapeStudio"
MSIX_DIR = ROOT / "packaging" / "msix"


def sdk_tool(name: str) -> Path:
    env = os.environ.get("WINDOWS_SDK_BIN")
    if env and (Path(env) / name).exists():
        return Path(env) / name
    kits = (
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"))
        / "Windows Kits"
        / "10"
        / "bin"
    )
    candidates = sorted(kits.glob(f"10.*/x64/{name}"), reverse=True)
    if not candidates:
        found = shutil.which(name)
        if found:
            return Path(found)
        sys.exit(f"{name} not found; install the Windows 10/11 SDK or set WINDOWS_SDK_BIN")
    return candidates[0]


def four_part(version: str) -> str:
    parts = version.split("-")[0].split(".")
    while len(parts) < 4:
        parts.append("0")
    return ".".join(parts[:4])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--store", action="store_true")
    mode.add_argument("--sideload", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not (APP_DIR / "ShapeStudio.exe").exists():
        sys.exit(
            f"{APP_DIR / 'ShapeStudio.exe'} missing; "
            "run pyinstaller packaging/ShapeStudio.spec first"
        )

    if args.store:
        ident = os.environ.get("MSIX_IDENTITY_NAME")
        publisher = os.environ.get("MSIX_PUBLISHER")
        display = os.environ.get("MSIX_PUBLISHER_DISPLAY", "Hivemind")
        if not ident or not publisher:
            sys.exit("--store needs MSIX_IDENTITY_NAME and MSIX_PUBLISHER (from Partner Center)")
    else:
        ident, publisher, display = (
            "Hivemind.ShapeStudio",
            "CN=Hivemind Shape Studio Dev",
            "Hivemind (dev build)",
        )

    staging = DIST / "msix-staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copytree(APP_DIR, staging / "ShapeStudio")
    shutil.copytree(MSIX_DIR / "Assets", staging / "Assets")
    manifest = (MSIX_DIR / "AppxManifest.template.xml").read_text(encoding="utf-8")
    for k, v in {
        "{{IDENTITY_NAME}}": ident,
        "{{PUBLISHER}}": publisher,
        "{{PUBLISHER_DISPLAY}}": display,
        "{{VERSION}}": four_part(args.version),
    }.items():
        manifest = manifest.replace(k, v)
    (staging / "AppxManifest.xml").write_text(manifest, encoding="utf-8")

    suffix = "store" if args.store else "sideload"
    out = (
        Path(args.out)
        if args.out
        else DIST / f"ShapeStudio-{args.version}-windows-x64-{suffix}.msix"
    )
    out.unlink(missing_ok=True)
    subprocess.run(
        [str(sdk_tool("makeappx.exe")), "pack", "/d", str(staging), "/p", str(out), "/o"],
        check=True,
    )
    print("packed", out, f"{out.stat().st_size / 2**20:.1f} MB")

    if not args.store:
        pfx = DIST / "ShapeStudio-dev.pfx"
        cer = DIST / "ShapeStudio-dev.cer"
        pwd = "shapestudio-dev"
        ps = (
            "$c = New-SelfSignedCertificate -Type Custom -Subject 'CN=Hivemind Shape Studio Dev' "
            "-KeyUsage DigitalSignature -FriendlyName 'Shape Studio dev signing' "
            "-CertStoreLocation 'Cert:\\CurrentUser\\My' "
            "-TextExtension @('2.5.29.37={text}1.3.6.1.5.5.7.3.3', '2.5.29.19={text}'); "
            f"$p = ConvertTo-SecureString -String '{pwd}' -Force -AsPlainText; "
            f"Export-PfxCertificate -Cert $c -FilePath '{pfx}' -Password $p | Out-Null; "
            f"Export-Certificate -Cert $c -FilePath '{cer}' | Out-Null; "
            "Remove-Item -Path ('Cert:\\CurrentUser\\My\\' + $c.Thumbprint)"
        )
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
        subprocess.run(
            [
                str(sdk_tool("signtool.exe")),
                "sign",
                "/fd",
                "SHA256",
                "/a",
                "/f",
                str(pfx),
                "/p",
                pwd,
                str(out),
            ],
            check=True,
        )
        pfx.unlink(missing_ok=True)
        print("signed for sideload; certificate to trust:", cer)
    shutil.rmtree(staging, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
