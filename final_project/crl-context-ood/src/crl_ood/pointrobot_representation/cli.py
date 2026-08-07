"""CLI with a lightweight dry-run path and lazy heavy-dependency loading."""

from __future__ import annotations

import argparse
from pathlib import Path

from .spec import dry_run_lines, load_spec, validate_primary_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("dry-run", "evaluate"))
    parser.add_argument("--spec", default="configs/pointrobot_representation/spec.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve(); spec = load_spec(root / args.spec)
    if args.command == "dry-run":
        for line in dry_run_lines(spec, root):
            print(line)
        return
    validate_primary_provenance(spec, root)
    from .evaluation import evaluate_all
    print(evaluate_all(spec, root))
