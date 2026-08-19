from pathlib import Path

import pandas as pd


SEEDS = [0, 1, 2, 3, 4]

ANALYSIS = Path(
    "results/cartpole_force_encoders/analysis"
)
ANALYSIS.mkdir(parents=True, exist_ok=True)

METHODS = {
    "hidden": (
        "results/cartpole_force_phase0/"
        "force_mag/hidden/seed_{seed}/context_returns.csv"
    ),
    "oracle_raw": (
        "results/cartpole_force_phase0/"
        "force_mag/oracle/seed_{seed}/context_returns.csv"
    ),
    "oracle_clipped": (
        "results/cartpole_force_phase0/"
        "force_mag/oracle_clipped/seed_{seed}/context_returns.csv"
    ),
    "vae_history": (
        "results/cartpole_force_history_downstream/"
        "vae/seed_{seed}/context_returns.csv"
    ),
    "contrastive_history": (
        "results/cartpole_force_history_downstream/"
        "contrastive/seed_{seed}/context_returns.csv"
    ),
}


# ============================================================
# Load context-level results
# ============================================================

frames = []

for method, template in METHODS.items():
    for seed in SEEDS:
        path = Path(
            template.format(seed=seed)
        )

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)

        assert len(df) == 27

        frames.append(
            pd.DataFrame(
                {
                    "method": method,
                    "seed": seed,
                    "split": df["split"].astype(str),
                    "force_mag":
                        df["context_value"].astype(float),
                    "mean_return":
                        df["mean_return"].astype(float),
                }
            )
        )

        print(
            f"{method:20s} seed={seed} PASS"
        )


combined = pd.concat(
    frames,
    ignore_index=True,
)


# ============================================================
# Split means per seed, then across seeds
# ============================================================

by_seed = (
    combined
    .groupby(
        [
            "method",
            "seed",
            "split",
        ],
        as_index=False,
    )
    .agg(
        mean_return=(
            "mean_return",
            "mean",
        )
    )
)


summary = (
    by_seed
    .groupby(
        [
            "method",
            "split",
        ],
        as_index=False,
    )
    .agg(
        mean_return=(
            "mean_return",
            "mean",
        ),
        sd_across_seeds=(
            "mean_return",
            "std",
        ),
        min_seed_return=(
            "mean_return",
            "min",
        ),
        max_seed_return=(
            "mean_return",
            "max",
        ),
        seeds=(
            "seed",
            "size",
        ),
    )
)


# ============================================================
# Force-wise means across seeds
# ============================================================

force_summary = (
    combined
    .groupby(
        [
            "method",
            "split",
            "force_mag",
        ],
        as_index=False,
    )
    .agg(
        mean_return=(
            "mean_return",
            "mean",
        ),
        sd_across_seeds=(
            "mean_return",
            "std",
        ),
    )
)


# ============================================================
# Paired seed deltas
# ============================================================

wide = (
    by_seed
    .pivot(
        index=["seed", "split"],
        columns="method",
        values="mean_return",
    )
    .reset_index()
)

comparisons = [
    (
        "oracle_raw_minus_hidden",
        "oracle_raw",
        "hidden",
    ),
    (
        "oracle_clipped_minus_hidden",
        "oracle_clipped",
        "hidden",
    ),
    (
        "vae_history_minus_hidden",
        "vae_history",
        "hidden",
    ),
    (
        "contrastive_history_minus_hidden",
        "contrastive_history",
        "hidden",
    ),
    (
        "vae_minus_contrastive",
        "vae_history",
        "contrastive_history",
    ),
    (
        "oracle_clipped_minus_raw",
        "oracle_clipped",
        "oracle_raw",
    ),
]

delta_rows = []

for label, a, b in comparisons:
    for split in sorted(
        wide["split"].unique()
    ):
        block = wide[
            wide["split"] == split
        ].sort_values("seed")

        delta = (
            block[a] - block[b]
        )

        row = {
            "comparison": label,
            "split": split,
            "mean_delta":
                float(delta.mean()),
            "sd_delta":
                float(delta.std()),
        }

        for seed, value in zip(
            block["seed"],
            delta,
            strict=True,
        ):
            row[
                f"seed_{int(seed)}_delta"
            ] = float(value)

        delta_rows.append(row)


deltas = pd.DataFrame(
    delta_rows
)


# ============================================================
# Print report-facing results
# ============================================================

ORDER = [
    "hidden",
    "oracle_raw",
    "oracle_clipped",
    "vae_history",
    "contrastive_history",
]

SPLITS = [
    "train",
    "id_test",
    "ood_low",
    "ood_high",
]


print()
print("=" * 88)
print("FIVE-SEED CARTPOLE SPLIT RESULTS")
print("=" * 88)

for method in ORDER:
    print(f"\n{method}")

    for split in SPLITS:
        row = summary[
            (summary["method"] == method)
            & (summary["split"] == split)
        ].iloc[0]

        print(
            f"  {split:10s} "
            f"{row['mean_return']:8.2f} "
            f"± {row['sd_across_seeds']:7.2f}"
            f"   range=["
            f"{row['min_seed_return']:.2f}, "
            f"{row['max_seed_return']:.2f}]"
        )


print()
print("=" * 88)
print("FIVE-SEED OOD PAIRED DELTAS")
print("=" * 88)

print(
    deltas[
        deltas["split"].isin(
            ["ood_low", "ood_high"]
        )
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)


print()
print("=" * 88)
print("FIVE-SEED OOD RETURNS BY FORCE")
print("=" * 88)

ood = force_summary[
    force_summary["split"].isin(
        ["ood_low", "ood_high"]
    )
]

pivot = ood.pivot(
    index=[
        "split",
        "force_mag",
    ],
    columns="method",
    values="mean_return",
)

pivot = pivot[
    ORDER
]

print(
    pivot.to_string(
        float_format=lambda x: f"{x:.1f}",
    )
)


# ============================================================
# Save
# ============================================================

combined.to_csv(
    ANALYSIS
    / "cartpole_force_5seed_context_returns.csv",
    index=False,
)

by_seed.to_csv(
    ANALYSIS
    / "cartpole_force_5seed_by_seed_split.csv",
    index=False,
)

summary.to_csv(
    ANALYSIS
    / "cartpole_force_5seed_split_summary.csv",
    index=False,
)

force_summary.to_csv(
    ANALYSIS
    / "cartpole_force_5seed_force_summary.csv",
    index=False,
)

deltas.to_csv(
    ANALYSIS
    / "cartpole_force_5seed_paired_deltas.csv",
    index=False,
)

print()
print("Saved final five-seed downstream artifacts.")
