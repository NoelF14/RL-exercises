"""Portable SHA-256 manifest helpers used by evaluation and result-only analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .spec import sha256_file


def write_manifest(base: str | Path, paths: Iterable[str | Path], output: str | Path) -> Path:
    root, destination = Path(base).resolve(), Path(output)
    entries = []
    for item in paths:
        path = Path(item).resolve()
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"manifest path escapes base directory: {path}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append((relative.as_posix(), sha256_file(path)))
    if len(entries) != len({name for name, _ in entries}):
        raise ValueError("manifest paths must be unique")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(entries)), encoding="utf-8")
    return destination


def verify_manifest(base: str | Path, manifest: str | Path) -> list[Path]:
    root, source = Path(base).resolve(), Path(manifest)
    if not source.is_file():
        raise FileNotFoundError(source)
    paths, seen = [], set()
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split(None, 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"invalid SHA-256 manifest line {line_number}")
        digest, relative = parts[0], parts[1].strip()
        if any(char not in "0123456789abcdef" for char in digest) or relative in seen:
            raise ValueError(f"invalid or duplicate SHA-256 manifest line {line_number}")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("manifest path escapes base directory") from exc
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"manifest hash verification failed: {relative}")
        seen.add(relative); paths.append(path)
    if not paths:
        raise ValueError("evaluation manifest may not be empty")
    return paths
