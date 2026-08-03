"""Plan or sequentially run the two fixed-center oracle sanity jobs."""

from __future__ import annotations

import argparse
from pathlib import Path

from crl_ood.audit.fixed_center import build_fixed_center_matrix, inspect_fixed_center_run
from crl_ood.diagnostic.matrix import DiagnosticJob, RunState


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-config", type=Path, default=Path("configs/audit/matrix.yaml"))
    parser.add_argument("--job-id", help="Run exactly one seed job")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip only validated complete jobs")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")

    jobs = build_fixed_center_matrix(args.matrix_config)
    if args.job_id:
        jobs = [job for job in jobs if job.job_id == args.job_id]
        if not jobs:
            parser.error(f"unknown --job-id {args.job_id!r}")
    statuses = [(job, inspect_fixed_center_run(job)) for job in jobs]
    _print_plan(statuses)
    if args.dry_run:
        return

    runnable: list[DiagnosticJob] = []
    for job, status in statuses:
        if args.resume:
            if status.state is RunState.COMPLETE:
                print(f"SKIP validated complete: {job.job_id}", flush=True)
                continue
            if status.state is RunState.PARTIAL:
                raise SystemExit(f"Refusing ambiguous partial directory for {job.job_id}: {status.detail}")
        elif status.state is not RunState.PENDING and not args.overwrite:
            raise SystemExit(
                f"Refusing existing output for {job.job_id} ({status.state.value}): "
                f"{status.detail}. Use --resume or explicitly --overwrite."
            )
        runnable.append(job)

    # Heavy RL dependencies are imported only when a training job will run.
    from crl_ood.training.train_ppo import train_one

    for index, job in enumerate(runnable, start=1):
        print(f"START {index}/{len(runnable)} {job.job_id}", flush=True)
        train_one(job.config, job.feature, job.mode, job.seed, overwrite=args.overwrite)
        status = inspect_fixed_center_run(job)
        if status.state is not RunState.COMPLETE:
            raise RuntimeError(f"Invalid completed job {job.job_id}: {status.detail}")
        print(f"COMPLETE {job.job_id}: {job.output_dir}", flush=True)


def _print_plan(statuses: list[tuple[DiagnosticJob, object]]) -> None:
    print("job_id\ttimesteps\tmode\tseed\tstate\toutput_dir")
    for job, status in statuses:
        print(
            f"{job.job_id}\t{job.total_timesteps}\t{job.mode}\t{job.seed}\t"
            f"{status.state.value}\t{job.output_dir}"
        )
    print(f"jobs={len(statuses)} concurrency=1")


if __name__ == "__main__":
    main()

