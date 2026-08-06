from __future__ import annotations

import hashlib
from pathlib import Path

ROOT=Path(__file__).parents[1]
PROTECTED=(ROOT/"configs/phase0.yaml",ROOT/"configs/diagnostic",ROOT/"configs/audit",ROOT/"configs/goal_pilot",ROOT/"results/phase0",ROOT/"results/phase0_diagnostic",ROOT/"results/phase0_audit",ROOT/"results/goal_pilot",ROOT/"results/goal_pilot_mechanistic_audit")


def test_every_protected_file_matches_manifest():
    expected=_manifest(ROOT/"manifests/pointrobot_gate_protected_before.sha256")
    actual={path.relative_to(ROOT).as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for target in PROTECTED for path in ([target] if target.is_file() else sorted(target.rglob("*"))) if path.is_file()}
    assert actual==expected


def test_before_after_manifests_byte_identical():
    assert (ROOT/"manifests/pointrobot_gate_protected_before.sha256").read_bytes()==(ROOT/"manifests/pointrobot_gate_protected_after.sha256").read_bytes()


def _manifest(path:Path)->dict[str,str]:
    return {relative:digest for digest,relative in (line.split("  ",1) for line in path.read_text().splitlines())}
