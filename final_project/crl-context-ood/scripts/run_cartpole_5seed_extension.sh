#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT/final_project/crl-context-ood" 2>/dev/null || cd "$ROOT"

RESULT_ROOT="results/cartpole_force_5seed_extension"
mkdir -p "$RESULT_ROOT"

LOG="$RESULT_ROOT/run.log"
STATUS="$RESULT_ROOT/STATUS.txt"

exec > >(tee -a "$LOG") 2>&1

export LD_LIBRARY_PATH="/run/current-system/sw/share/nix-ld/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MPLCONFIGDIR=/tmp/cartpole_5seed_mpl

fail() {
    rc=$?
    {
        echo "FAILED"
        echo "time: $(date -Is)"
        echo "exit_code: $rc"
        echo "line: ${BASH_LINENO[0]}"
        echo "command: ${BASH_COMMAND}"
    } > "$STATUS"

    echo
    echo "============================================================"
    echo "CARTPOLE FIVE-SEED EXTENSION FAILED"
    echo "============================================================"
    cat "$STATUS"

    exit "$rc"
}

trap fail ERR

{
    echo "RUNNING"
    echo "started: $(date -Is)"
    echo "host: $(hostname)"
    echo "head: $(git rev-parse HEAD)"
} > "$STATUS"

echo "============================================================"
echo "CARTPOLE FIVE-SEED EXTENSION"
echo "============================================================"
echo "started: $(date -Is)"
echo "repo:    $(pwd)"
echo "HEAD:    $(git rev-parse HEAD)"
echo


# ------------------------------------------------------------
# Preflight: scientific specification
# ------------------------------------------------------------

echo "============================================================"
echo "PREFLIGHT"
echo "============================================================"

.venv/bin/python - <<'PY'
from pathlib import Path
import yaml

encoder = yaml.safe_load(
    Path(
        "configs/cartpole_encoders/force_primary.yaml"
    ).read_text()
)

phase0 = yaml.safe_load(
    Path(
        "configs/cartpole_force_phase0.yaml"
    ).read_text()
)

assert encoder["encoder"]["history_length"] == 5
assert encoder["encoder"]["future_horizon"] == 5
assert encoder["encoder"]["latent_dim"] == 8
assert encoder["encoder"]["hidden_size"] == 64
assert encoder["encoder"]["max_updates"] == 20000

assert (
    encoder["dataset"]["checksum"]
    ==
    "be2942887fd919057efb89525fcc8809c0d7dfb5ae6985020520cbabfd578ebf"
)

assert (
    phase0["training"]["total_timesteps"]
    == 100000
)

print("Frozen experiment specification: PASS")
PY


# Critical training files must have no uncommitted changes.
CRITICAL=(
    configs/cartpole_encoders/force_primary.yaml
    configs/cartpole_force_phase0.yaml
    scripts/train_cartpole_force_encoder.py
    scripts/train_cartpole_force_history_policy.py
    src/crl_ood/cartpole_encoders/wrapper.py
    src/crl_ood/pointrobot_encoders/training.py
    src/crl_ood/pointrobot_encoders/models.py
    src/crl_ood/pointrobot_encoders/dataset.py
    src/crl_ood/training/train_ppo.py
)

git diff --quiet -- "${CRITICAL[@]}"
git diff --cached --quiet -- "${CRITICAL[@]}"

echo "Critical training files clean: PASS"
echo


echo "Running test suite..."
.venv/bin/python -m pytest -q
echo


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

check_empty_or_complete() {
    local dir="$1"
    local marker="$2"

    if [[ -e "$marker" ]]; then
        return 0
    fi

    if [[ -d "$dir" ]] && [[ -n "$(find "$dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "ERROR: incomplete/nonempty run directory:"
        echo "  $dir"
        exit 1
    fi

    return 1
}


# ------------------------------------------------------------
# 1. Encoder seeds 2, 3, 4
# ------------------------------------------------------------

for seed in 2 3 4
do
    for method in vae contrastive
    do
        dir="results/cartpole_force_encoders/encoders/${method}/seed_${seed}"
        marker="${dir}/best.pt"

        echo
        echo "============================================================"
        echo "ENCODER ${method} — seed ${seed}"
        echo "============================================================"

        if check_empty_or_complete "$dir" "$marker"
        then
            echo "Already complete; skipping."
        else
            .venv/bin/python \
                scripts/train_cartpole_force_encoder.py \
                --method "$method" \
                --seed "$seed"
        fi
    done
done


# ------------------------------------------------------------
# 2. Hidden + raw-oracle PPO seeds 2, 3, 4
# ------------------------------------------------------------

for seed in 2 3 4
do
    for mode in hidden oracle
    do
        dir="results/cartpole_force_phase0/force_mag/${mode}/seed_${seed}"
        marker="${dir}/model.zip"

        echo
        echo "============================================================"
        echo "CONTROL PPO ${mode} — seed ${seed}"
        echo "============================================================"

        if check_empty_or_complete "$dir" "$marker"
        then
            echo "Already complete; skipping."
        else
            .venv/bin/python -m crl_ood.training.train_ppo \
                --config configs/cartpole_force_phase0.yaml \
                --feature force_mag \
                --mode "$mode" \
                --seed "$seed"
        fi
    done
done


# ------------------------------------------------------------
# 3. Clipped-oracle diagnostic seeds 2, 3, 4
# ------------------------------------------------------------

for seed in 2 3 4
do
    dir="results/cartpole_force_phase0/force_mag/oracle_clipped/seed_${seed}"
    marker="${dir}/context_returns.csv"

    echo
    echo "============================================================"
    echo "CLIPPED ORACLE — seed ${seed}"
    echo "============================================================"

    if check_empty_or_complete "$dir" "$marker"
    then
        echo "Already complete; skipping."
    else
        .venv/bin/python \
            scripts/evaluate_cartpole_force_clipped_oracle_seed.py \
            --seed "$seed"
    fi
done


# ------------------------------------------------------------
# 4. Learned-history PPO seeds 2, 3, 4
# ------------------------------------------------------------

for seed in 2 3 4
do
    for method in vae contrastive
    do
        checkpoint="results/cartpole_force_encoders/encoders/${method}/seed_${seed}/best.pt"
        dir="results/cartpole_force_history_downstream/${method}/seed_${seed}"
        marker="${dir}/model.zip"

        echo
        echo "============================================================"
        echo "HISTORY PPO ${method} — seed ${seed}"
        echo "============================================================"

        if [[ ! -f "$checkpoint" ]]
        then
            echo "Missing encoder checkpoint: $checkpoint"
            exit 1
        fi

        if check_empty_or_complete "$dir" "$marker"
        then
            echo "Already complete; skipping."
        else
            .venv/bin/python \
                scripts/train_cartpole_force_history_policy.py \
                --method "$method" \
                --seed "$seed" \
                --checkpoint "$checkpoint"
        fi
    done
done


# ------------------------------------------------------------
# 5. Final audit
# ------------------------------------------------------------

echo
echo "============================================================"
echo "FINAL FIVE-SEED EXTENSION AUDIT"
echo "============================================================"

.venv/bin/python - <<'PY'
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch


checksum = (
    "be2942887fd919057efb89525fcc8809c0d7dfb5ae6985020520cbabfd578ebf"
)


for seed in [2, 3, 4]:

    # Encoders
    for method in ["vae", "contrastive"]:

        ckpt = (
            Path("results/cartpole_force_encoders/encoders")
            / method
            / f"seed_{seed}"
            / "best.pt"
        )

        payload = torch.load(
            ckpt,
            map_location="cpu",
            weights_only=False,
        )

        assert payload["seed"] == seed
        assert payload["method"] == method
        assert payload["dataset_checksum"] == checksum

    # Phase-0 controls
    for mode in ["hidden", "oracle"]:

        run = (
            Path("results/cartpole_force_phase0")
            / "force_mag"
            / mode
            / f"seed_{seed}"
        )

        assert (run / "model.zip").exists()
        assert (run / "context_returns.csv").exists()
        assert (run / "episode_returns.csv").exists()

    # Clipped diagnostic
    clipped = (
        Path("results/cartpole_force_phase0")
        / "force_mag"
        / "oracle_clipped"
        / f"seed_{seed}"
    )

    assert (
        clipped / "context_returns.csv"
    ).exists()

    # Learned-history policies
    for method in ["vae", "contrastive"]:

        run = (
            Path("results/cartpole_force_history_downstream")
            / method
            / f"seed_{seed}"
        )

        meta = json.loads(
            (run / "provenance.json").read_text()
        )

        assert meta["ppo_total_timesteps"] == 100000
        assert (run / "model.zip").exists()
        assert (run / "context_returns.csv").exists()
        assert (run / "episode_returns.csv").exists()

    # Strong clipped-oracle sanity:
    # train + ID context observations are already inside [-1, 1],
    # so raw and clipped oracle should be identical there.
    raw = pd.read_csv(
        Path("results/cartpole_force_phase0")
        / "force_mag"
        / "oracle"
        / f"seed_{seed}"
        / "context_returns.csv"
    )

    clip = pd.read_csv(
        clipped / "context_returns.csv"
    )

    keep = ["train", "id_test"]

    raw = raw[
        raw["split"].isin(keep)
    ][
        ["split", "context_value", "mean_return"]
    ].rename(
        columns={
            "mean_return": "raw"
        }
    )

    clip = clip[
        clip["split"].isin(keep)
    ][
        ["split", "context_value", "mean_return"]
    ].rename(
        columns={
            "mean_return": "clipped"
        }
    )

    merged = raw.merge(
        clip,
        on=["split", "context_value"],
        validate="one_to_one",
    )

    assert np.allclose(
        merged["raw"],
        merged["clipped"],
        atol=0.0,
        rtol=0.0,
    )

    print(f"seed {seed}: PASS")


print("\nCARTPOLE SEEDS 2-4: COMPLETE")
PY


{
    echo "DONE"
    echo "finished: $(date -Is)"
    echo "head: $(git rev-parse HEAD)"
    echo "seeds_added: 2 3 4"
} > "$STATUS"

echo
echo "============================================================"
echo "ALL CARTPOLE FIVE-SEED EXTENSION JOBS FINISHED"
echo "============================================================"
cat "$STATUS"
