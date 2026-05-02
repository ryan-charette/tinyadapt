from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / "README.md").exists():
            return candidate
    return current


def find_default_checkpoint(start: Path | None = None) -> Path | None:
    root = find_project_root(start)
    candidates = [
        root / "checkpoints" / "bignet.pth",
        root / "bignet.pth",
    ]
    return next((path for path in candidates if path.exists()), None)
