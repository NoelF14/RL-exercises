from __future__ import annotations

import csv
import json

from crl_ood.analysis.analyze_oracle_audit import analyze_oracle_audit


def test_result_only_oracle_audit_analyzer_with_synthetic_artifacts(tmp_path):
    audit = tmp_path / "phase0_audit"
    diagnostic = tmp_path / "phase0_diagnostic"
    specialist_path = audit / "specialist_transfer/specialist_transfer_matrix_by_seed.csv"
    specialist_path.parent.mkdir(parents=True)
    specialist_rows = []
    for specialist_index, specialist in enumerate(("low", "center", "high")):
        for seed in (0, 1):
            for split, context in (("train", 0.8), ("id_test", 1.0), ("ood_high", 1.4)):
                specialist_rows.append(
                    {
                        "training_specialist_context": specialist,
                        "seed": seed,
                        "evaluation_split": split,
                        "evaluation_context_value": context,
                        "mean_return": -100 + specialist_index * 5 + seed + context,
                    }
                )
    _write(specialist_path, specialist_rows)

    ablation_path = audit / "oracle_ablation/oracle_ablation_by_seed.csv"
    ablation_path.parent.mkdir(parents=True)
    ablation_rows = []
    offsets = {"true_context": 10.0, "zero_context": 0.0, "shuffled_context": -5.0}
    for mode, offset in offsets.items():
        for seed in (0, 1):
            for split in ("train", "id_test", "ood_low", "ood_high"):
                ablation_rows.append(
                    {"observation_mode": mode, "seed": seed, "evaluation_split": split, "mean_return": -100 + seed + offset}
                )
    _write(ablation_path, ablation_rows)

    paths = analyze_oracle_audit(audit, diagnostic)
    findings = json.loads(paths["findings"].read_text(encoding="utf-8"))
    assert findings["scope"]["confidence_intervals_computed"] is False
    assert findings["scope"]["ood_used_for_selection_or_tuning"] is False
    assert findings["questions"]["true_context_better_than_ablations"]["true_better_than_both_count"] == 4
    assert findings["questions"]["fixed_center_oracle_matches_hidden"]["available"] is False
    for path in paths.values():
        assert path.is_file() and path.stat().st_size > 0
        if path.suffix == ".csv":
            assert "confidence" not in path.read_text(encoding="utf-8").splitlines()[0].lower()


def _write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)

