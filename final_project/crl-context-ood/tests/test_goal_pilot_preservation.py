from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).parents[1]
PROTECTED = (
    ROOT / "configs/phase0.yaml",
    ROOT / "configs/diagnostic",
    ROOT / "configs/audit",
    ROOT / "results/phase0",
    ROOT / "results/phase0_diagnostic",
    ROOT / "results/phase0_audit",
)


def test_protected_artifacts_match_captured_before_manifest():
    expected = _read_manifest(ROOT / "manifests/goal_pilot_protected_before.sha256")
    actual = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for target in PROTECTED
        for path in ([target] if target.is_file() else sorted(target.rglob("*")))
        if path.is_file()
    }
    assert actual == expected


def test_before_and_after_protected_manifests_are_identical():
    before = (ROOT / "manifests/goal_pilot_protected_before.sha256").read_bytes()
    after = (ROOT / "manifests/goal_pilot_protected_after.sha256").read_bytes()
    assert before == after


def _read_manifest(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        result[relative] = digest
    return result
