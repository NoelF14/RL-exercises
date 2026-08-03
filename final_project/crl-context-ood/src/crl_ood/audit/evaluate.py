"""Evaluation-only specialist transfer and contextual-oracle ablation audits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from crl_ood.environments.context_splits import carl_feature_key
from crl_ood.evaluation.evaluate import load_evaluation_plan
from crl_ood.utils.metadata import load_context_manifest
from crl_ood.utils.paths import project_root
from crl_ood.utils.seeding import seed_everything

SPLITS = ("train", "id_test", "ood_low", "ood_high")
SPECIALISTS = ("low", "center", "high")
ABLATION_MODES = ("true_context", "zero_context", "shuffled_context")

SPECIALIST_EPISODE_FIELDS = (
    "training_specialist_context", "checkpoint_experiment", "seed", "evaluation_split",
    "evaluation_context_value", "context_id", "episode_index", "episode_seed", "return",
    "episode_length", "termination_type",
)
SPECIALIST_CONTEXT_FIELDS = (
    "training_specialist_context", "checkpoint_experiment", "seed", "evaluation_split",
    "evaluation_context_value", "context_id", "episodes", "mean_return", "std_return",
)
ABLATION_EPISODE_FIELDS = (
    "observation_mode", "seed", "evaluation_split", "environment_context_value",
    "context_id", "episode_index", "episode_seed", "policy_context_scalar", "return",
    "episode_length", "termination_type",
)
ABLATION_CONTEXT_FIELDS = (
    "observation_mode", "seed", "evaluation_split", "environment_context_value",
    "context_id", "policy_context_scalar", "episodes", "mean_return", "std_return",
)
MAPPING_FIELDS = (
    "seed", "evaluation_split", "context_id", "environment_context_value",
    "true_normalized_context", "shuffled_normalized_context", "source_split",
    "source_context_id", "source_environment_context_value",
)


@dataclass(frozen=True)
class AuditEvaluationJob:
    kind: str
    label: str
    seed: int
    checkpoint: Path
    source_run_dir: Path
    output_dir: Path


def load_audit_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("audit"), dict):
        raise ValueError("Audit YAML must contain an 'audit' mapping")
    return config


def resolve_checkpoint(
    diagnostic_root: str | Path, experiment: str, mode: str, seed: int
) -> Path:
    """Resolve and validate one immutable diagnostic checkpoint path."""
    path = Path(diagnostic_root) / experiment / "length" / mode / f"seed_{seed}" / "model.zip"
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Checkpoint is not a valid ZIP archive: {path}")
    return path.resolve()


def build_specialist_jobs(config: dict[str, Any]) -> list[AuditEvaluationJob]:
    """Build the complete 3-specialist x 2-seed evaluation matrix."""
    audit = config["audit"]
    root = _root_path(audit["diagnostic_results_dir"])
    output_root = _root_path(audit["results_dir"]) / "specialist_transfer" / "checkpoints"
    jobs = []
    for label in SPECIALISTS:
        experiment = str(audit["specialist_experiments"][label])
        for seed in audit["seeds"]:
            seed = int(seed)
            checkpoint = resolve_checkpoint(root, experiment, "hidden", seed)
            source = root / str(audit["contextual_experiment"]) / "length" / "hidden" / f"seed_{seed}"
            _validate_source_protocol(source)
            jobs.append(
                AuditEvaluationJob(
                    "specialist_transfer", label, seed, checkpoint, source.resolve(),
                    output_root / label / f"seed_{seed}",
                )
            )
    _validate_unique_jobs(jobs, expected=6)
    return jobs


def build_oracle_jobs(config: dict[str, Any]) -> list[AuditEvaluationJob]:
    """Build the two contextual-oracle checkpoint audit jobs."""
    audit = config["audit"]
    root = _root_path(audit["diagnostic_results_dir"])
    output_root = _root_path(audit["results_dir"]) / "oracle_ablation" / "checkpoints"
    experiment = str(audit["contextual_experiment"])
    jobs = []
    for seed in audit["seeds"]:
        seed = int(seed)
        checkpoint = resolve_checkpoint(root, experiment, "oracle", seed)
        source = root / experiment / "length" / "oracle" / f"seed_{seed}"
        _validate_source_protocol(source)
        jobs.append(
            AuditEvaluationJob(
                "oracle_ablation", "contextual_oracle", seed, checkpoint,
                source.resolve(), output_root / f"seed_{seed}",
            )
        )
    _validate_unique_jobs(jobs, expected=2)
    return jobs


def build_shuffled_mapping(
    evaluation_plan: list[dict[str, Any]], normalization: tuple[float, float], seed: int
) -> list[dict[str, Any]]:
    """Create a deterministic derangement of context scalars across contexts."""
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for row in evaluation_plan:
        key = (str(row["split"]), int(row["context_id"]))
        if key not in seen:
            seen.add(key)
            unique.append(row)
    if len(unique) < 2:
        raise ValueError("Shuffled-context ablation requires at least two contexts")
    center, scale = normalization
    scalars = np.asarray(
        [(float(row["context_value"]) - center) / scale for row in unique], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(unique))
    for _ in range(1000):
        if np.all(permutation != np.arange(len(unique))):
            break
        permutation = rng.permutation(len(unique))
    else:
        permutation = np.roll(np.arange(len(unique)), 1)
    if np.all(permutation == np.arange(len(unique))):
        raise RuntimeError("Shuffled mapping unexpectedly maps every context to itself")

    rows = []
    for target_index, target in enumerate(unique):
        source_index = int(permutation[target_index])
        source = unique[source_index]
        rows.append(
            {
                "seed": seed,
                "evaluation_split": str(target["split"]),
                "context_id": int(target["context_id"]),
                "environment_context_value": float(target["context_value"]),
                "true_normalized_context": float(scalars[target_index]),
                "shuffled_normalized_context": float(scalars[source_index]),
                "source_split": str(source["split"]),
                "source_context_id": int(source["context_id"]),
                "source_environment_context_value": float(source["context_value"]),
            }
        )
    if sorted(row["shuffled_normalized_context"] for row in rows) != sorted(scalars.tolist()):
        raise RuntimeError("Shuffled mapping failed to preserve normalized-context multiset")
    return rows


def modify_oracle_observation(
    observation: np.ndarray, mode: str, *, shuffled_scalar: float | None = None
) -> np.ndarray:
    """Copy an oracle observation and alter only its final policy scalar."""
    if mode not in ABLATION_MODES:
        raise ValueError(f"Unknown oracle observation mode: {mode}")
    original = np.asarray(observation)
    if original.ndim != 1 or original.size < 2:
        raise ValueError("Expected a flat state-plus-context oracle observation")
    if mode == "true_context":
        return original.copy()
    changed = original.copy()
    if mode == "zero_context":
        changed[-1] = 0.0
    else:
        if shuffled_scalar is None:
            raise ValueError("shuffled_context requires a mapped scalar")
        changed[-1] = shuffled_scalar
    return changed


def evaluate_specialist_job(job: AuditEvaluationJob, model: Any | None = None) -> list[dict[str, Any]]:
    splits, normalization, feature = load_context_manifest(job.source_run_dir / "contexts.yaml")
    plan = load_evaluation_plan(job.source_run_dir / "evaluation_plan.csv")
    _validate_plan(plan, job.seed, "hidden")
    model = model or _load_model(job.checkpoint)
    rows = _rollout(model, splits, normalization, plan, feature, "hidden", job.seed)
    result = [
        {
            "training_specialist_context": job.label,
            "checkpoint_experiment": job.checkpoint.parents[3].name,
            "seed": job.seed,
            "evaluation_split": row["split"],
            "evaluation_context_value": row["context_value"],
            "context_id": row["context_id"],
            "episode_index": row["episode_index"],
            "episode_seed": row["episode_seed"],
            "return": row["return"],
            "episode_length": row["episode_length"],
            "termination_type": row["termination_type"],
        }
        for row in rows
    ]
    _write_job_output(job, result, SPECIALIST_EPISODE_FIELDS, SPECIALIST_CONTEXT_FIELDS)
    return result


def evaluate_oracle_job(job: AuditEvaluationJob, model: Any | None = None) -> list[dict[str, Any]]:
    splits, normalization, feature = load_context_manifest(job.source_run_dir / "contexts.yaml")
    plan = load_evaluation_plan(job.source_run_dir / "evaluation_plan.csv")
    _validate_plan(plan, job.seed, "oracle")
    mapping = build_shuffled_mapping(plan, normalization, job.seed)
    mapped = {
        (row["evaluation_split"], row["context_id"]): row["shuffled_normalized_context"]
        for row in mapping
    }
    model = model or _load_model(job.checkpoint)
    results = []
    for mode in ABLATION_MODES:
        raw = _rollout(
            model, splits, normalization, plan, feature, "oracle", job.seed,
            observation_mode=mode, shuffled_mapping=mapped,
        )
        for row in raw:
            true_scalar = (float(row["context_value"]) - normalization[0]) / normalization[1]
            scalar = true_scalar if mode == "true_context" else 0.0
            if mode == "shuffled_context":
                scalar = mapped[(str(row["split"]), int(row["context_id"]))]
            results.append(
                {
                    "observation_mode": mode,
                    "seed": job.seed,
                    "evaluation_split": row["split"],
                    "environment_context_value": row["context_value"],
                    "context_id": row["context_id"],
                    "episode_index": row["episode_index"],
                    "episode_seed": row["episode_seed"],
                    "policy_context_scalar": scalar,
                    "return": row["return"],
                    "episode_length": row["episode_length"],
                    "termination_type": row["termination_type"],
                }
            )
    _write_job_output(job, results, ABLATION_EPISODE_FIELDS, ABLATION_CONTEXT_FIELDS)
    _write_csv(job.output_dir / "oracle_shuffled_mapping.csv", MAPPING_FIELDS, mapping)
    return results


def _rollout(
    model: Any,
    splits: dict[str, dict[int, dict[str, float]]],
    normalization: tuple[float, float],
    plan: list[dict[str, Any]],
    feature: str,
    env_mode: str,
    seed: int,
    *,
    observation_mode: str = "true_context",
    shuffled_mapping: dict[tuple[str, int], float] | None = None,
) -> list[dict[str, Any]]:
    from crl_ood.environments.factory import make_pendulum_env

    deterministic = True
    key = carl_feature_key(feature)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for episode in plan:
        grouped[(str(episode["split"]), int(episode["context_id"]))].append(episode)
    rows = []
    for (split, context_id), episodes in grouped.items():
        context = splits[split][context_id]
        env = make_pendulum_env(
            {context_id: context}, feature, env_mode, seed,
            context_normalization=normalization, static_context=True,
        )
        expected_context = float(context[key])
        for episode in episodes:
            observation, _ = env.reset(seed=int(episode["episode_seed"]))
            if float(env.active_context[key]) != expected_context:
                raise RuntimeError("Actual CARL environment context changed before evaluation")
            terminated = truncated = False
            episode_return = 0.0
            length = 0
            replacement = None
            if observation_mode == "shuffled_context":
                replacement = (shuffled_mapping or {})[(split, context_id)]
            while not (terminated or truncated):
                policy_observation = observation
                if env_mode == "oracle":
                    policy_observation = modify_oracle_observation(
                        observation, observation_mode, shuffled_scalar=replacement
                    )
                action, _ = model.predict(policy_observation, deterministic=deterministic)
                observation, reward, terminated, truncated, _ = env.step(action)
                if float(env.active_context[key]) != expected_context:
                    raise RuntimeError("Actual CARL environment context changed during evaluation")
                episode_return += float(reward)
                length += 1
            rows.append(
                {
                    **episode,
                    "return": episode_return,
                    "episode_length": length,
                    "termination_type": _termination_type(terminated, truncated),
                }
            )
        env.close()
    return rows


def run_audit_task(
    task: str, config: dict[str, Any], *, resume: bool = False, overwrite: bool = False,
    dry_run: bool = False,
) -> list[Path]:
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    if task == "specialist-transfer":
        jobs = build_specialist_jobs(config)
    elif task == "oracle-ablation":
        jobs = build_oracle_jobs(config)
    else:
        raise ValueError(f"Unknown audit task: {task}")
    task_root = _root_path(config["audit"]["results_dir"]) / (
        "specialist_transfer" if task == "specialist-transfer" else "oracle_ablation"
    )
    states = [(job, inspect_evaluation_job(job)) for job in jobs]
    for job, state in states:
        print(f"{job.kind}\t{job.label}\tseed={job.seed}\t{state}\t{job.output_dir}")
    if dry_run:
        return []
    if overwrite and task_root.exists():
        shutil.rmtree(task_root)
        states = [(job, "pending") for job in jobs]
    elif not resume and task_root.exists() and any(task_root.iterdir()):
        raise FileExistsError(
            f"Audit output already exists: {task_root}. Use --resume or explicitly --overwrite."
        )

    for job, state in states:
        if state == "complete" and resume:
            print(f"SKIP validated complete: {job.label} seed={job.seed}")
            continue
        if state == "partial":
            raise RuntimeError(f"Refusing partial audit output: {job.output_dir}")
        job.output_dir.mkdir(parents=True, exist_ok=False)
        seed_everything(job.seed)
        if task == "specialist-transfer":
            evaluate_specialist_job(job)
        else:
            evaluate_oracle_job(job)
        if inspect_evaluation_job(job) != "complete":
            raise RuntimeError(f"Audit output failed validation: {job.output_dir}")
    return _aggregate_task(task, jobs, task_root)


def inspect_evaluation_job(job: AuditEvaluationJob) -> str:
    """Return pending, partial, or complete after validating persisted rows."""
    if not job.output_dir.exists() or (job.output_dir.is_dir() and not any(job.output_dir.iterdir())):
        return "pending"
    if not job.output_dir.is_dir():
        return "partial"
    required = ["episode_returns.csv", "context_returns.csv", "completion.json"]
    if job.kind == "oracle_ablation":
        required.append("oracle_shuffled_mapping.csv")
    if any(not (job.output_dir / name).is_file() for name in required):
        return "partial"
    try:
        with (job.output_dir / "completion.json").open(encoding="utf-8") as handle:
            completion = json.load(handle)
        if completion != {
            "checkpoint_sha256": _sha256(job.checkpoint),
            "kind": job.kind,
            "label": job.label,
            "seed": job.seed,
        }:
            return "partial"
        episodes = _read_csv(job.output_dir / "episode_returns.csv")
        contexts = _read_csv(job.output_dir / "context_returns.csv")
        expected = len(load_evaluation_plan(job.source_run_dir / "evaluation_plan.csv"))
        multiplier = 3 if job.kind == "oracle_ablation" else 1
        if len(episodes) != expected * multiplier or not contexts:
            return "partial"
        pair_fields = ("evaluation_split", "context_id", "episode_index", "episode_seed")
        if job.kind == "oracle_ablation":
            by_mode = {
                mode: [tuple(row[field] for field in pair_fields) for row in episodes if row["observation_mode"] == mode]
                for mode in ABLATION_MODES
            }
            if set(map(tuple, by_mode.values())) and not all(by_mode[mode] == by_mode["true_context"] for mode in ABLATION_MODES):
                return "partial"
            mapping = _read_csv(job.output_dir / "oracle_shuffled_mapping.csv")
            if len(mapping) != len({(row["split"], row["context_id"]) for row in load_evaluation_plan(job.source_run_dir / "evaluation_plan.csv")}):
                return "partial"
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return "partial"
    return "complete"


def _write_job_output(
    job: AuditEvaluationJob,
    rows: list[dict[str, Any]],
    episode_fields: tuple[str, ...],
    context_fields: tuple[str, ...],
) -> None:
    _write_csv(job.output_dir / "episode_returns.csv", episode_fields, rows)
    group_fields = context_fields[:-3]
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(float(row["return"]))
    summaries = [
        dict(zip(context_fields, (*key, len(values), statistics.fmean(values), statistics.pstdev(values)), strict=True))
        for key, values in grouped.items()
    ]
    _write_csv(job.output_dir / "context_returns.csv", context_fields, summaries)
    completion = {
        "checkpoint_sha256": _sha256(job.checkpoint),
        "kind": job.kind,
        "label": job.label,
        "seed": job.seed,
    }
    (job.output_dir / "completion.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _aggregate_task(task: str, jobs: list[AuditEvaluationJob], root: Path) -> list[Path]:
    if task == "specialist-transfer":
        episode_rows = _combine(jobs, "episode_returns.csv")
        context_rows = _combine(jobs, "context_returns.csv")
        episode_path = root / "specialist_transfer_episode_returns.csv"
        context_path = root / "specialist_transfer_context_returns.csv"
        matrix_path = root / "specialist_transfer_matrix_by_seed.csv"
        summary_path = root / "specialist_transfer_summary.csv"
        plot_path = root / "specialist_transfer_heatmap.png"
        _write_csv(episode_path, SPECIALIST_EPISODE_FIELDS, episode_rows)
        _write_csv(context_path, SPECIALIST_CONTEXT_FIELDS, context_rows)
        _write_csv(matrix_path, SPECIALIST_CONTEXT_FIELDS, context_rows)
        summary = _seed_summary(
            context_rows,
            ("training_specialist_context", "evaluation_split", "evaluation_context_value"),
        )
        _write_csv(summary_path, tuple(summary[0]), summary)
        _plot_specialist_heatmap(context_rows, plot_path)
        return [episode_path, context_path, matrix_path, summary_path, plot_path]
    episode_rows = _combine(jobs, "episode_returns.csv")
    context_rows = _combine(jobs, "context_returns.csv")
    mapping_rows = _combine(jobs, "oracle_shuffled_mapping.csv")
    episode_path = root / "oracle_ablation_episode_returns.csv"
    context_path = root / "oracle_ablation_context_returns.csv"
    by_seed_path = root / "oracle_ablation_by_seed.csv"
    summary_path = root / "oracle_ablation_summary.csv"
    mapping_path = root / "oracle_shuffled_mapping.csv"
    plot_path = root / "oracle_ablation_plot.png"
    _write_csv(episode_path, ABLATION_EPISODE_FIELDS, episode_rows)
    _write_csv(context_path, ABLATION_CONTEXT_FIELDS, context_rows)
    by_seed = _context_weighted_summary(context_rows, ("observation_mode", "seed", "evaluation_split"))
    _write_csv(by_seed_path, tuple(by_seed[0]), by_seed)
    summary = _seed_summary(by_seed, ("observation_mode", "evaluation_split"))
    _write_csv(summary_path, tuple(summary[0]), summary)
    _write_csv(mapping_path, MAPPING_FIELDS, mapping_rows)
    _plot_oracle_ablation(by_seed, plot_path)
    return [episode_path, context_path, by_seed_path, summary_path, mapping_path, plot_path]


def _context_weighted_summary(rows: list[dict[str, str]], groups: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in groups)].append(float(row["mean_return"]))
    return [
        {**dict(zip(groups, key, strict=True)), "number_contexts": len(values), "mean_return": statistics.fmean(values)}
        for key, values in sorted(grouped.items())
    ]


def _seed_summary(rows: list[dict[str, Any]], groups: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[field]) for field in groups)].append(float(row["mean_return"]))
    return [
        {
            **dict(zip(groups, key, strict=True)),
            "n_seeds": len(values),
            "mean_of_seed_means": statistics.fmean(values),
            "std_of_seed_means": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min_seed_mean": min(values),
            "max_seed_mean": max(values),
        }
        for key, values in sorted(grouped.items())
    ]


def _plot_specialist_heatmap(rows: list[dict[str, str]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = sorted({float(row["evaluation_context_value"]) for row in rows})
    matrix = np.empty((3, len(values)))
    for row_index, specialist in enumerate(SPECIALISTS):
        for column, value in enumerate(values):
            selected = [float(row["mean_return"]) for row in rows if row["training_specialist_context"] == specialist and float(row["evaluation_context_value"]) == value]
            matrix[row_index, column] = statistics.fmean(selected)
    fig, ax = plt.subplots(figsize=(14, 3.8))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set(yticks=range(3), yticklabels=SPECIALISTS, xticks=range(len(values)), xticklabels=[f"{value:g}" for value in values], xlabel="Evaluation length", ylabel="Training specialist context", title="Specialist transfer (mean across two checkpoint seeds)")
    fig.colorbar(image, ax=ax, label="Mean return")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def _plot_oracle_ablation(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
    for ax, split in zip(axes, SPLITS, strict=True):
        selected = [row for row in rows if row["evaluation_split"] == split]
        for seed in sorted({int(row["seed"]) for row in selected}):
            indexed = {row["observation_mode"]: float(row["mean_return"]) for row in selected if int(row["seed"]) == seed}
            ax.plot(ABLATION_MODES, [indexed[mode] for mode in ABLATION_MODES], marker="o", label=f"seed {seed}")
        ax.set_title(split + (" (descriptive)" if split.startswith("ood") else "")); ax.tick_params(axis="x", rotation=30); ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean return"); axes[0].legend(); fig.suptitle("Contextual oracle observation ablations"); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def _load_model(path: Path) -> Any:
    from stable_baselines3 import PPO
    return PPO.load(path)


def _validate_source_protocol(path: Path) -> None:
    for name in ("contexts.yaml", "evaluation_plan.csv"):
        if not (path / name).is_file():
            raise FileNotFoundError(f"Missing saved evaluation protocol: {path / name}")


def _validate_plan(plan: list[dict[str, Any]], seed: int, method: str) -> None:
    if not plan or {row["seed"] for row in plan} != {seed} or {row["method"] for row in plan} != {method}:
        raise ValueError("Saved evaluation plan does not match checkpoint seed and mode")
    if tuple(dict.fromkeys(str(row["split"]) for row in plan)) != SPLITS:
        raise ValueError("Saved evaluation plan does not preserve original split order")


def _validate_unique_jobs(jobs: list[AuditEvaluationJob], expected: int) -> None:
    if len(jobs) != expected or len({job.output_dir.resolve() for job in jobs}) != expected:
        raise ValueError(f"Expected {expected} unique audit jobs")


def _combine(jobs: Iterable[AuditEvaluationJob], filename: str) -> list[dict[str, str]]:
    return [row for job in jobs for row in _read_csv(job.output_dir / filename)]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _root_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root() / path


def _termination_type(terminated: bool, truncated: bool) -> str:
    if terminated and truncated: return "terminated_and_truncated"
    if terminated: return "terminated"
    if truncated: return "truncated"
    raise RuntimeError("Evaluation ended without termination or truncation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("specialist-transfer", "oracle-ablation"))
    parser.add_argument("--config", type=Path, default=Path("configs/audit/oracle_audit.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in run_audit_task(
        args.task, load_audit_config(args.config), resume=args.resume,
        overwrite=args.overwrite, dry_run=args.dry_run,
    ):
        print(path)


if __name__ == "__main__":
    main()
