from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).parents[1]
PROTECTED = (
    ROOT / "configs/phase0.yaml", ROOT / "configs/diagnostic", ROOT / "configs/audit",
    ROOT / "configs/goal_pilot", ROOT / "results/phase0", ROOT / "results/phase0_diagnostic",
    ROOT / "results/phase0_audit", ROOT / "results/goal_pilot",
)


def test_all_completed_artifacts_match_mechanistic_audit_baseline():
    expected = _manifest(ROOT / "manifests/mechanistic_audit_protected_before.sha256")
    actual = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for target in PROTECTED
        for path in ([target] if target.is_file() else sorted(target.rglob("*")))
        if path.is_file()
    }
    assert actual == expected


def test_mechanistic_before_and_after_manifests_are_identical():
    before = ROOT / "manifests/mechanistic_audit_protected_before.sha256"
    after = ROOT / "manifests/mechanistic_audit_protected_after.sha256"
    if after.exists():
        assert before.read_bytes() == after.read_bytes()


def _manifest(path: Path) -> dict[str, str]:
    return {relative: digest for digest, relative in
            (line.split("  ", 1) for line in path.read_text(encoding="utf-8").splitlines())}

