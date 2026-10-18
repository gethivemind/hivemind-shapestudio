"""Which model files are installed and verified. A small JSON manifest avoids re-hashing
gigabytes on every start: a file counts as installed when it exists with the pinned size and
the manifest records its hash as verified."""
# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable
from pathlib import Path

from shapestudio import download as dl
from shapestudio.config import Settings
from shapestudio.registry import MODELS, TOOL_MODELS, FileSpec, MeshModel, ToolModel

ConsentFn = Callable[[str, int], bool]  # (description, bytes) -> allowed?
ProgressFn = Callable[[str, int, int], None]  # (filename, done, total)


def _size_matches(path: Path, size: int) -> bool:
    """Catches truncation after verification. Tests patch this instead of writing 1.6 GB."""
    return path.stat().st_size == size


class ModelStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._manifest: dict[str, str] = {}
        self._mtime: int | None = None
        self._load()

    # -- manifest ------------------------------------------------------------------------

    def _load(self) -> None:
        p = self.settings.manifest_file
        if p.exists():
            try:
                self._manifest = json.loads(p.read_text(encoding="utf-8"))
                self._mtime = p.stat().st_mtime_ns
            except (OSError, ValueError):
                self._manifest = {}
        else:
            self._mtime = None

    def _refresh(self) -> None:
        """Re-read the manifest when another process changed it (a download finished by the
        CLI while the UI was running). One stat per query."""
        p = self.settings.manifest_file
        try:
            m = p.stat().st_mtime_ns if p.exists() else None
        except OSError:
            return
        if m != self._mtime:
            self._load()

    def _save(self) -> None:
        self.settings.models_dir.mkdir(parents=True, exist_ok=True)
        self.settings.manifest_file.write_text(
            json.dumps(self._manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        with contextlib.suppress(OSError):
            self._mtime = self.settings.manifest_file.stat().st_mtime_ns

    # -- queries -------------------------------------------------------------------------

    def path_for(self, spec: FileSpec) -> Path:
        return self.settings.models_dir / spec.filename

    def file_installed(self, spec: FileSpec) -> bool:
        self._refresh()
        p = self.path_for(spec)
        return (
            p.exists()
            and _size_matches(p, spec.size)
            and self._manifest.get(spec.filename) == spec.sha256
        )

    def model_installed(self, model: MeshModel) -> bool:
        return all(self.file_installed(f) for f in model.files)

    def installed_models(self) -> list[str]:
        return [m.name for m in MODELS.values() if self.model_installed(m)]

    def missing_bytes(self, model: MeshModel) -> int:
        return sum(f.size for f in model.files if not self.file_installed(f))

    def model_paths(self, model: MeshModel) -> dict[str, Path]:
        """Basename -> path for every file of an installed model (what the engine loads)."""
        return {Path(f.filename).name: self.path_for(f) for f in model.files}

    def tool_installed(self, tool: ToolModel) -> bool:
        return self.file_installed(tool.file)

    def installed_tools(self) -> list[str]:
        return [t.name for t in TOOL_MODELS.values() if self.tool_installed(t)]

    # -- install -------------------------------------------------------------------------

    def _install_files(
        self,
        files: list[FileSpec],
        desc: str,
        consent: ConsentFn,
        progress: ProgressFn | None,
    ) -> bool:
        missing = [f for f in files if not self.file_installed(f)]
        if not missing:
            return True
        if not consent(desc, sum(f.size for f in missing)):
            return False
        for f in missing:

            def _p(done: int, tot: int, name: str = Path(f.filename).name) -> None:
                if progress:
                    progress(name, done, tot)

            dl.download(f.url, self.path_for(f), f.sha256, f.size, progress=_p)
            self._manifest[f.filename] = f.sha256
            self._save()
        return True

    def install_model(
        self, model: MeshModel, consent: ConsentFn, progress: ProgressFn | None = None
    ) -> bool:
        """Download every missing file of ``model`` after ``consent`` approves the total.
        Returns False (and downloads nothing) when consent is refused."""
        missing = [f for f in model.files if not self.file_installed(f)]
        desc = f"{model.display_name} ({model.licence}): " + ", ".join(
            Path(f.filename).name for f in missing
        )
        return self._install_files(list(model.files), desc, consent, progress)

    def install_tool(
        self, tool: ToolModel, consent: ConsentFn, progress: ProgressFn | None = None
    ) -> bool:
        desc = f"{tool.display_name} ({tool.licence}): {tool.file.filename}"
        return self._install_files([tool.file], desc, consent, progress)

    def remove_files(self, files) -> int:
        """Delete the given files and forget their hashes. Returns bytes freed."""
        freed = 0
        for f in files:
            p = self.path_for(f)
            if p.exists():
                freed += p.stat().st_size
                p.unlink()
            part = p.with_suffix(p.suffix + ".part")
            if part.exists():
                part.unlink()
            self._manifest.pop(f.filename, None)
        self._save()
        return freed

    def file_info(self, spec: FileSpec) -> dict:
        p = self.path_for(spec)
        return {
            "filename": spec.filename,
            "path": str(p),
            "installed": self.file_installed(spec),
            "bytes": p.stat().st_size if p.exists() else 0,
            "size": spec.size,
            "licence": spec.licence,
            "source": spec.source,
            "sha256": spec.sha256,
        }

    def verify_file(self, spec: FileSpec) -> bool:
        """Re-hash a file on disk and update the manifest. Slow; used by ``doctor --verify``."""
        p = self.path_for(spec)
        if not p.exists():
            return False
        ok = dl.sha256_of(p) == spec.sha256
        if ok:
            self._manifest[spec.filename] = spec.sha256
        else:
            self._manifest.pop(spec.filename, None)
        self._save()
        return ok
