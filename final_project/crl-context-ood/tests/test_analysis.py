from __future__ import annotations

import csv
import subprocess
import sys

import pandas as pd
import pytest

from crl_ood.analysis.analyze_phase0 import analyze_results

SPLITS = ("train", "id_test", "ood_low", "ood_high")


def test_analysis_import_does_not_load_carl():
    command = (
        "import sys; import crl_ood.analysis.analyze_phase0; "
        "assert not any(name == 'carl' or name.startswith('carl.') for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", command], check=True)


def _write_synthetic_run(root, method, seed):
    run_dir = root / "gravity" / method / f"seed_{seed}"
    run_dir.mkdir(parents=True)
    episode_fields = (
        "run_id",
        "method",
        "seed",
        "context_feature",
        "context_value",
        "split",
        "context_id",
        "episode_index",
        "episode_seed",
        "return",
        "episode_length",
        "termination_type",
    )
    gaps = {"train": 4.0, "id_test": 5.0, "ood_low": -2.0, "ood_high": 8.0}
    with (run_dir / "episode_returns.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=episode_fields)
        writer.writeheader()
        for split_index, split in enumerate(SPLITS):
            for episode_index, offset in enumerate((-1.0, 1.0)):
                hidden_return = -10.0 + seed + split_index + offset
                value = hidden_return if method == "hidden" else hidden_return + gaps[split]
                writer.writerow(
                    {
                        "run_id": f"synthetic__gravity__{method}__seed_{seed}",
                        "method": method,
                        "seed": seed,
                        "context_feature": "gravity",
                        "context_value": 5.0 + split_index,
                        "split": split,
                        "context_id": 0,
                        "episode_index": episode_index,
                        "episode_seed": 1000 + split_index * 10 + episode_index,
                        "return": value,
                        "episode_length": 200,
                        "termination_type": "truncated",
                    }
                )
    training_fields = (
        "run_id",
        "method",
        "seed",
        "context_feature",
        "environment_steps",
        "episode_index",
        "episode_return",
        "episode_length",
    )
    with (run_dir / "training_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=training_fields)
        writer.writeheader()
        for episode_index in range(2):
            writer.writerow(
                {
                    "run_id": f"synthetic__gravity__{method}__seed_{seed}",
                    "method": method,
                    "seed": seed,
                    "context_feature": "gravity",
                    "environment_steps": (episode_index + 1) * 200,
                    "episode_index": episode_index,
                    "episode_return": -1000 + episode_index * 100,
                    "episode_length": 200,
                }
            )


def test_result_only_analysis_pairs_methods_and_preserves_gap_signs(tmp_path):
    results_root = tmp_path / "phase0"
    for seed in (0, 1):
        for method in ("hidden", "oracle"):
            _write_synthetic_run(results_root, method, seed)

    output_dir = tmp_path / "analysis"
    paths = analyze_results(results_root, output_dir)
    gaps = pd.read_csv(paths["paired_gaps"])
    context_pairs = pd.read_csv(paths["context_pairs"])
    screening = pd.read_csv(paths["screening"])

    for split, expected in {
        "train": 4.0,
        "id_test": 5.0,
        "ood_low": -2.0,
        "ood_high": 8.0,
    }.items():
        assert gaps[gaps["split"] == split]["oracle_gap"].tolist() == pytest.approx(
            [expected, expected]
        )
        assert context_pairs[context_pairs["split"] == split][
            "oracle_gap"
        ].tolist() == pytest.approx([expected, expected])

    assert len(screening) == 16
    assert "confidence_interval" not in screening.columns
    for path in paths.values():
        assert path.is_file() and path.stat().st_size > 0
