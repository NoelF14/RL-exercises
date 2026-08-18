from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml

from crl_ood.pointrobot_encoders.dataset import load_dataset
from crl_ood.pointrobot_encoders.training import train_encoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cartpole_encoders/force_primary.yaml"),
    )
    parser.add_argument("--method", choices=("vae", "contrastive"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    dataset_dir = Path(config["dataset"]["path"])
    _, metadata = load_dataset(dataset_dir)

    expected = str(config["dataset"]["checksum"])
    actual = str(metadata["dataset_checksum"])

    if actual != expected:
        raise ValueError(
            f"dataset checksum mismatch: expected {expected}, got {actual}"
        )

    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    # Stored inside checkpoint payload via the frozen config.
    config["experiment"]["source_git_commit"] = source_commit

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path(config["experiment"]["results_dir"])
            / "encoders"
            / args.method
            / f"seed_{args.seed}"
        )

    print("method:          ", args.method)
    print("seed:            ", args.seed)
    print("dataset:         ", dataset_dir)
    print("dataset checksum:", actual)
    print("source commit:   ", source_commit)
    print("output:          ", output_dir)
    print("max updates:     ", args.max_updates or config["encoder"]["max_updates"])

    train_encoder(
        config,
        dataset_dir,
        args.method,
        args.seed,
        output_dir,
        max_updates=args.max_updates,
    )


if __name__ == "__main__":
    main()
