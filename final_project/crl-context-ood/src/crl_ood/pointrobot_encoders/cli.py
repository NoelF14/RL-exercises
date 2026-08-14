"""Command-line interface for datasets, matched encoders, and downstream plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crl_ood.pointrobot_encoders.dataset import collect_arrays, load_spec, save_dataset
from crl_ood.pointrobot_encoders.downstream import build_jobs, load_yaml, run_job
from crl_ood.pointrobot_encoders.training import evaluate_frozen, train_encoder
from crl_ood.utils.paths import project_root


def _dataset_path(config: dict, budget: str) -> Path:
    return project_root() / config["experiment"]["results_dir"] / "datasets" / budget


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pointrobot_encoders/primary.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    dataset = sub.add_parser("dataset")
    dataset.add_argument("--budget", choices=("tiny", "small", "full"), required=True)
    dataset.add_argument("--dry-run", action="store_true")
    train = sub.add_parser("train")
    train.add_argument("--dataset", required=True); train.add_argument("--method", choices=("vae", "contrastive", "contrastive_alternative"), required=True)
    train.add_argument("--seed", type=int, required=True); train.add_argument("--output", required=True)
    train.add_argument("--max-updates", type=int); train.add_argument("--resume", action="store_true")
    pilot = sub.add_parser("pilot")
    pilot.add_argument("--dataset", required=True); pilot.add_argument("--method", choices=("vae", "contrastive", "contrastive_alternative", "all"), default="all")
    pilot.add_argument("--resume", action="store_true")
    frozen = sub.add_parser("evaluate-frozen")
    frozen.add_argument("--dataset", required=True); frozen.add_argument("--checkpoint", required=True); frozen.add_argument("--output", required=True)
    downstream = sub.add_parser("downstream")
    downstream.add_argument("--downstream-config", default="configs/pointrobot_encoders/downstream.yaml")
    downstream.add_argument("--matrix", choices=("integration_pilot", "full_primary"), default="integration_pilot")
    downstream.add_argument("--vae-checkpoint"); downstream.add_argument("--contrastive-checkpoint")
    downstream.add_argument("--contrastive-alternative-checkpoint")
    downstream.add_argument("--dataset-checksum"); downstream.add_argument("--dry-run", action="store_true")
    downstream.add_argument("--timesteps-override", type=int)
    downstream.add_argument("--methods", nargs="+", choices=("no_context", "oracle", "vae", "contrastive", "contrastive_alternative"))
    downstream.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args(argv)
    config = load_spec(args.config)
    if args.command == "dataset":
        arrays, metadata = collect_arrays(config, args.budget)
        if args.dry_run:
            print(json.dumps({key: metadata[key] for key in ("budget", "episode_count", "transition_count", "dataset_checksum")}, indent=2))
        else:
            path = save_dataset(_dataset_path(config, args.budget), arrays, metadata)
            print(f"collected immutable dataset: {path} checksum={metadata['dataset_checksum']}")
    elif args.command == "train":
        path = train_encoder(config, args.dataset, args.method, args.seed, args.output,
                             max_updates=args.max_updates, resume=args.resume)
        print(f"completed encoder run: {path}")
    elif args.command == "pilot":
        methods = config["pilot"]["methods"] if args.method == "all" else [args.method]
        for method in methods:
            for seed in config["pilot"]["seeds"]:
                output = project_root() / config["experiment"]["results_dir"] / "runs" / method / f"seed_{seed}"
                if (output / "run.log").is_file():
                    print(f"skip complete: {output}"); continue
                train_encoder(config, args.dataset, method, int(seed), output, resume=args.resume and output.exists())
                print(f"complete: {output}")
    elif args.command == "evaluate-frozen":
        print(f"frozen evaluation: {evaluate_frozen(args.dataset, args.checkpoint, args.output)}")
    else:
        downstream_config = load_yaml(args.downstream_config)
        checkpoints = {key: value for key, value in {"vae": args.vae_checkpoint,
            "contrastive": args.contrastive_checkpoint,
            "contrastive_alternative": args.contrastive_alternative_checkpoint}.items() if value}
        jobs = build_jobs(downstream_config, args.matrix, checkpoints, args.dataset_checksum,
                          timesteps_override=args.timesteps_override)
        if args.methods:
            jobs = [job for job in jobs if job.method in args.methods]
        if args.seeds:
            jobs = [job for job in jobs if job.seed in args.seeds]
        if args.dry_run:
            for job in jobs:
                print(f"{job.method} seed={job.seed} steps={job.total_timesteps} output={job.output_dir} checkpoint={job.checkpoint}")
        else:
            gate = load_yaml(project_root() / downstream_config["experiment"]["source_gate_config"])
            for job in jobs:
                run_job(job, gate); print(f"complete: {job.output_dir}")


if __name__ == "__main__":
    main()
