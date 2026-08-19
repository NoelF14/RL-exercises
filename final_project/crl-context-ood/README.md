# OOD Generalization of Learned Context Representations in cRL

This repository accompanies the paper **“OOD Generalization of Learned Context
Representations in cRL.”** It studies whether context representations learned
from short transition histories support out-of-distribution (OOD) control. The
final comparison freezes either a predictive VAE-style history encoder or a
ContraBAR-inspired contrastive history encoder, then trains a PPO policy on the
resulting latent representation. All final comparisons use five independent
seeds (`0,1,2,3,4`).

For the exact experimental specification and scientific interpretation, see
the accompanying report. This README focuses on implementation and
reproducibility.

Stable-Baselines3 supplies PPO and CARL supplies the auxiliary CartPole
environment. The PointRobot environment, dataset builders, encoder models and
objectives, frozen-history wrappers, ablation, probes, audits, and final
analysis code were implemented for this project.

## Main findings

- Strong in-distribution control and context decodability do not guarantee OOD
  control.
- In PointRobot, the original contrastive objective exhibits a severe
  right-OOD failure that disappears across all five seeds when the
  negative-construction scheme is changed, although performance trades off
  elsewhere.
- In CartPole, the VAE latent decodes force magnitude almost perfectly
  (`R² = 0.995`), while the contrastive latent does not (`R² = 0.022`), yet the
  contrastive policy has better high-force OOD control.
- Raw oracle CartPole context can hurt OOD extrapolation. Clipping only the
  policy-side normalized context to the training range restores high-force
  performance without changing the physical environment context.

## Environments

### PointRobot

The primary task is a custom deterministic dense PointRobot. Its hidden context
is the angle of a goal on the unit circle; changing the context changes the
reward while the state/action spaces and dynamics stay fixed. The fixed train, ID, left-OOD,
and right-OOD angles are recorded in
`configs/pointrobot_primary/spec.yaml`. The environment has a 50-step horizon
and dense post-transition rewards.

The custom PointRobot environment was independently implemented for this
project and was inspired by the Semi-Circle PointRobot setting used in
ContraBAR; no external PointRobot environment implementation was copied.

### CARL CartPole

The auxiliary dynamics-shift stress test uses CARL CartPole with `force_mag` as
the hidden context. A context-oblivious policy already solves train and ID
contexts, so CartPole is **not** claimed to be a context-necessity task. Its
purpose is to test high- and low-force OOD behavior, learned-history policies,
and the raw-versus-clipped oracle diagnostic.

## Installation

The final runs used Python 3.13.14, PyTorch 2.7.1+cpu, Gymnasium 0.29.1,
Stable-Baselines3 2.7.0, CARL 1.1.1, and CPU execution.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest
```

The experiment machine had an Intel Core i7-12700 (12 physical cores / 20
hardware threads), 31 GiB RAM, and no NVIDIA GPU.

### Optional lab/Nix runtime note

On the lab's Nix setup, dynamically linked Python packages sometimes required:

```bash
export LD_LIBRARY_PATH="/run/current-system/sw/share/nix-ld/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

This is a machine-specific workaround, not a general dependency. Likewise,
setting a writable Matplotlib cache directory can avoid cache warnings:

```bash
export MPLCONFIGDIR=/tmp/pointrobot_report_mpl
```

## Repository / result structure

Important source and configuration locations are:

```text
src/crl_ood/pointrobot_gate/          custom PointRobot environment
src/crl_ood/pointrobot_encoders/      datasets, encoders, frozen wrappers, PPO integration
src/crl_ood/pointrobot_primary/       frozen primary PointRobot runner/spec validation
src/crl_ood/cartpole_encoders/        CartPole dataset and history wrapper
src/crl_ood/analysis/                 result-only primary and representation analyses
configs/pointrobot_primary/spec.yaml  frozen PointRobot primary specification
configs/pointrobot_encoders/primary.yaml
                                      matched dataset and encoder objective source
configs/pointrobot_encoders/negative_ablation.yaml
                                      adjacent-negative encoder specification
configs/pointrobot_encoders/downstream_negative_ablation.yaml
                                      adjacent-negative PPO specification
configs/pointrobot_negative_ablation/spec_teammate.yaml
                                      preserved ablation comparison/provenance spec
configs/pointrobot_representation/    frozen representation-probe specification
configs/cartpole_force_phase0.yaml    CartPole control-policy specification
configs/cartpole_encoders/force_primary.yaml
                                      frozen CartPole encoder specification
scripts/                              experiment, audit, analysis, and plotting entry points
```

The final saved artifacts are organized as follows:

```text
results/pointrobot_primary/                 primary dataset, encoders, PPO, analysis
results/pointrobot_negative_ablation/       adjacent-negative encoders, PPO, summaries
results/pointrobot_representation/          frozen-latent probes and PCA diagnostics
results/cartpole_force_phase0/              hidden, raw-oracle, clipped-oracle controls
results/cartpole_force_encoders/dataset/full_frozen
results/cartpole_force_encoders/encoders/   CartPole VAE/contrastive checkpoints
results/cartpole_force_history_downstream/  learned-history PPO runs
results/cartpole_force_encoders/analysis/   CartPole aggregate and probe tables
figures/                                    final report figures
```

## Reproducing the final paper results

Run commands from the repository root. Training commands below are expensive
and are documented for reproduction only; none is needed to regenerate the
analyses from the saved artifacts.

### Fast path: reproduce analyses and report figures from saved artifacts
The `results/` directory is intentionally excluded from Git because it contains
large generated experiment artifacts. The commands in this subsection are the
fast path for a working copy that already contains the final saved artifacts.
For a fresh clone, use the verified from-scratch procedures below to regenerate
the corresponding datasets, checkpoints, and result tables.

PointRobot primary aggregation and frozen-representation analysis read saved
artifacts. The three report plotters take no arguments and write to `figures/`.

```bash
export MPLCONFIGDIR=/tmp/pointrobot_report_mpl
.venv/bin/python scripts/analyze_pointrobot_primary.py
.venv/bin/python scripts/analyze_pointrobot_representation.py
.venv/bin/python scripts/plot_report_main_figure.py
.venv/bin/python scripts/plot_report_probe_by_angle.py
.venv/bin/python scripts/plot_report_return_by_angle.py
```

The following no-argument mechanistic audits verify the original contrastive
negative mapping and quantify its reward-relabel signal. They read the primary
dataset identified by its checksum and write diagnostics to
`results/pointrobot_p3/`.

```bash
.venv/bin/python scripts/audit_pointrobot_negative_mapping.py
.venv/bin/python scripts/audit_pointrobot_reward_relabel_signal.py
```

CartPole five-seed aggregation, latent probing, and the report figure are:

```bash
.venv/bin/python scripts/analyze_cartpole_force_5seed.py
.venv/bin/python scripts/probe_cartpole_force_latents_5seed.py
export MPLCONFIGDIR=/tmp/cartpole_report_mpl
.venv/bin/python scripts/plot_cartpole_force_5seed.py
```

An additional probe plot exists but is not a main report figure:

```bash
.venv/bin/python scripts/plot_cartpole_force_probe_5seed.py
```

### PointRobot from scratch

The frozen primary runner is the verified entry point. It constructs 2,000
matched episodes (100,000 transitions), trains both encoder methods for seeds
0–4 for up to 20,000 gradient updates, validates checkpoints, and trains all
four downstream methods. Every learned policy seed is paired with the encoder
of the same seed.

First inspect the plans without collecting data or training:

```bash
.venv/bin/python scripts/run_pointrobot_primary.py \
  --spec configs/pointrobot_primary/spec.yaml dataset --dry-run
.venv/bin/python scripts/run_pointrobot_primary.py \
  --spec configs/pointrobot_primary/spec.yaml encoder-matrix --dry-run
.venv/bin/python scripts/run_pointrobot_primary.py \
  --spec configs/pointrobot_primary/spec.yaml downstream --dry-run
```

In a fresh output namespace (the runners refuse nonempty run directories), run
the stages in order:

```bash
.venv/bin/python scripts/run_pointrobot_primary.py \
  --spec configs/pointrobot_primary/spec.yaml dataset
.venv/bin/python scripts/run_pointrobot_primary.py \
  --spec configs/pointrobot_primary/spec.yaml encoder-matrix
.venv/bin/python scripts/run_pointrobot_primary.py \
  --spec configs/pointrobot_primary/spec.yaml validate-encoders
.venv/bin/python scripts/run_pointrobot_primary.py \
  --spec configs/pointrobot_primary/spec.yaml downstream
```

The requested PPO budget is 200,000 steps. Because PPO finishes complete
2,048-step rollouts, each saved primary run records 200,704 actual environment
steps. The immutable dataset must match:

```text
results/pointrobot_primary/dataset/full
cb826e04b344eb875662b8775b89f9c60bdb9bae895f25a260d25ef422a589fa
```

The alternative-negative ablation reuses that exact dataset. There is no
single checked-in ablation runner; the verified per-stage commands are:

```bash
for seed in 0 1 2 3 4; do
  .venv/bin/python scripts/run_pointrobot_encoders.py \
    --config configs/pointrobot_encoders/negative_ablation.yaml train \
    --dataset results/pointrobot_primary/dataset/full \
    --method contrastive_alternative --seed "$seed" \
    --output "results/pointrobot_negative_ablation/encoders/contrastive_alternative/seed_${seed}"
done

for seed in 0 1 2 3 4; do
  .venv/bin/python scripts/run_pointrobot_encoders.py \
    --config configs/pointrobot_encoders/negative_ablation.yaml downstream \
    --downstream-config configs/pointrobot_encoders/downstream_negative_ablation.yaml \
    --matrix full_primary \
    --contrastive-alternative-checkpoint \
      "results/pointrobot_negative_ablation/encoders/contrastive_alternative/seed_${seed}/best.pt" \
    --dataset-checksum cb826e04b344eb875662b8775b89f9c60bdb9bae895f25a260d25ef422a589fa \
    --methods contrastive_alternative --seeds "$seed"
done
```

The preserved ablation summaries are in
`results/pointrobot_negative_ablation/analysis/`. No standalone checked-in
aggregator for those summary CSVs could be verified, so none is claimed here.

To rebuild the frozen representation evaluations and probes in a fresh
`results/pointrobot_representation/` directory, inspect and execute the frozen
plan, then analyze it:

```bash
.venv/bin/python scripts/run_pointrobot_representation.py dry-run \
  --spec configs/pointrobot_representation/spec.yaml
.venv/bin/python scripts/run_pointrobot_representation.py evaluate \
  --spec configs/pointrobot_representation/spec.yaml
.venv/bin/python scripts/analyze_pointrobot_representation.py
```

The evaluation runner verifies the primary dataset and all ten checkpoint
hashes and refuses a nonempty evaluation directory. Use the fast-path plot and
audit commands above after analysis.

### CartPole from scratch

Train the hidden and raw-oracle controls for five seeds with the frozen
100,000-step configuration:

```bash
.venv/bin/crl-ood-phase0 --config configs/cartpole_force_phase0.yaml \
  --features force_mag --modes hidden oracle --seeds 0 1 2 3 4
```

Build the fixed encoder dataset. The builder has no CLI flags: it reads
`configs/cartpole_force_phase0.yaml`, writes the path below, and refuses to
overwrite an existing immutable dataset.

```bash
.venv/bin/python scripts/build_cartpole_force_encoder_dataset.py
```

The resulting dataset must match:

```text
results/cartpole_force_encoders/dataset/full_frozen
be2942887fd919057efb89525fcc8809c0d7dfb5ae6985020520cbabfd578ebf
```

Train the two frozen encoders for up to 20,000 updates and their same-seed
learned-history PPO policies:

```bash
for seed in 0 1 2 3 4; do
  for method in vae contrastive; do
    .venv/bin/python scripts/train_cartpole_force_encoder.py \
      --config configs/cartpole_encoders/force_primary.yaml \
      --method "$method" --seed "$seed"
  done
done

for seed in 0 1 2 3 4; do
  for method in vae contrastive; do
    .venv/bin/python scripts/train_cartpole_force_history_policy.py \
      --config configs/cartpole_force_phase0.yaml \
      --method "$method" --seed "$seed" \
      --checkpoint "results/cartpole_force_encoders/encoders/${method}/seed_${seed}/best.pt"
  done
done
```

Each CartPole PPO run requests 100,000 environment steps. The clipped-oracle
diagnostic performs no training: it reloads each raw-oracle model, keeps the
physical force unchanged, reuses the saved evaluation plan, and clips only the
normalized context shown to the policy.

```bash
for seed in 0 1 2 3 4; do
  .venv/bin/python scripts/evaluate_cartpole_force_clipped_oracle_seed.py \
    --config configs/cartpole_force_phase0.yaml --seed "$seed"
done
```

Finally run the CartPole fast-path analysis/probe/plot commands above.
`scripts/run_cartpole_5seed_extension.sh` is retained as provenance for the
actual extension of an existing seeds-0/1 experiment to seeds 2–4. It is **not**
a clean full-from-scratch five-seed runner.

## Experimental safeguards / reproducibility

- PointRobot and CartPole encoder datasets are immutable and checksum-verified.
  The expected SHA-256 values are frozen in their final specs.
- Dataset behavior actions are matched across training contexts, so differences
  are not caused by different action sequences.
- Train, ID, and directional OOD splits are fixed. Encoder fitting and
  checkpoint selection use training-context data only; OOD is descriptive
  scientific evaluation.
- Encoders are frozen during downstream PPO training. Learned methods pair
  policy seed `s` with encoder seed `s`.
- Evaluation contexts and episode seeds/plans are persisted with runs so later
  config edits cannot silently change evaluation.
- Resolved configs, dataset/checkpoint hashes, package/device metadata, source
  commits, and manifests are stored alongside final artifacts.
- The five end-to-end seeds are independent replicates. Contexts, samples, and
  deterministic repeated evaluation episodes are aggregated within seed and
  are not counted as independent replicates.

## Preliminary CARL Phase-0 experiments

The earlier CARL Pendulum experiments varied `gravity`, `length`, or `dt` to
screen task/context necessity and included a separate task-validity diagnostic.
They are preliminary task-selection work, **not** the final learned-context
representation study. Their still-valid entry points are retained for
historical reproducibility. `configs/smoke.yaml` is the short smoke matrix,
`configs/phase0.yaml` is the preliminary Pendulum matrix, and
`configs/diagnostic/matrix.yaml` declares the separate diagnostic jobs.

```bash
.venv/bin/crl-ood-phase0 --config configs/smoke.yaml

.venv/bin/crl-ood-phase0 --config configs/phase0.yaml \
  --features gravity length dt --seeds 0 1

.venv/bin/crl-ood-phase0 --config configs/phase0.yaml \
  --features gravity --seeds 0 1 2 3 4

.venv/bin/crl-ood-train --config configs/phase0.yaml \
  --feature gravity --mode hidden --seed 0

.venv/bin/crl-ood-evaluate --config configs/phase0.yaml \
  --model results/phase0/gravity/hidden/seed_0/model.zip \
  --feature gravity --mode hidden --seed 0 \
  --output-dir results/phase0/gravity/hidden/seed_0

.venv/bin/crl-ood-analyze --results-root results/phase0 \
  --output-dir results/phase0/analysis
```

The separate diagnostic matrix can be planned, run atomically/resumably, and
analyzed with:

```bash
.venv/bin/python scripts/run_phase0_diagnostic.py \
  --matrix-config configs/diagnostic/matrix.yaml --dry-run

.venv/bin/python scripts/run_phase0_diagnostic.py \
  --matrix-config configs/diagnostic/matrix.yaml \
  --job-id default_100k__length__hidden__seed_0

.venv/bin/python scripts/run_phase0_diagnostic.py \
  --matrix-config configs/diagnostic/matrix.yaml --resume

.venv/bin/python -m crl_ood.analysis.analyze_diagnostic \
  --results-root results/phase0_diagnostic \
  --output-dir results/phase0_diagnostic/analysis
```

## External software and licenses

- CARL 1.1.1 — contextual environment benchmark — Apache-2.0
- Stable-Baselines3 2.7.0 — PPO implementation — MIT
- Gymnasium 0.29.1 — environment API — MIT
- PyTorch 2.7.1 — representation learning framework — BSD-style license

Scientific citations for CARL, PPO, Stable-Baselines3, VariBAD, and ContraBAR
are included in the report.

## Contributions

- Leana Meyer contributed the CartPole experiment.
- Noel Freimuth contributed the PointRobot experiment.
- Both authors contributed to encoder design, experimental design,
  interpretation, and the final report.
