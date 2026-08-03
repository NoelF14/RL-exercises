# CRL Context OOD

An empirical Phase 0 study of whether hiding an environment's active context
materially hurts policy performance relative to an oracle that observes it.
The current implementation varies one `CARLPendulum` context feature at a time:

- `gravity` maps to CARL's `g` feature;
- `length` maps to CARL's `l` feature;
- `dt` is the simulation timestep.

This repository intentionally does not contain VAE or contrastive context
encoders. PPO is provided by Stable-Baselines3 and is not reimplemented here.

## Installation

Python 3.10 or newer is required. A local environment can be created with:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
```

CARL's base install supplies the classic-control environment used here. No
optional Box2D, Brax, DMC, Mario, or RNA dependencies are needed.

## Experiment commands

Run the tests and the short hidden-versus-oracle smoke experiment:

```bash
.venv/bin/python -m pytest
.venv/bin/crl-ood-phase0 --config configs/smoke.yaml
```

Screen all three candidate context features with two seeds:

```bash
.venv/bin/crl-ood-phase0 --config configs/phase0.yaml \
  --features gravity length dt --seeds 0 1
```

After selecting a feature, run the five-seed comparison (shown for gravity):

```bash
.venv/bin/crl-ood-phase0 --config configs/phase0.yaml \
  --features gravity --seeds 0 1 2 3 4
```

Train one atomic run or reevaluate an existing checkpoint with:

```bash
.venv/bin/crl-ood-train --config configs/phase0.yaml \
  --feature gravity --mode hidden --seed 0

.venv/bin/crl-ood-evaluate --config configs/phase0.yaml \
  --model results/phase0/gravity/hidden/seed_0/model.zip \
  --feature gravity --mode hidden --seed 0 \
  --output-dir results/phase0/gravity/hidden/seed_0
```

## Design

Each configuration defines explicit train, ID-test, OOD-low, and OOD-high
ranges as multipliers of CARL's default feature value. Context values are
assigned with a configured seed. Training uses CARL's deterministic round-robin
selector. Evaluation pins one context per environment and uses the same context
values and episode seeds for hidden and oracle agents.

The hidden observation is the three-dimensional Pendulum state. The oracle
observation concatenates only the single active context value, normalized so
the minimum and maximum training contexts map to `-1` and `1`; OOD values may
therefore lie outside that interval. This produces a four-dimensional oracle
observation without exposing CARL's other context fields. Both methods use
SB3's `MlpPolicy` and deterministic prediction during evaluation.

Every atomic run writes the following under
`results/<experiment>/<feature>/<mode>/seed_<seed>/`:

- `resolved_config.yaml` and `seed.txt`;
- `metadata.json` with Git commit/dirty state, package versions, and device data;
- `contexts.yaml` with the fully expanded CARL contexts passed to the environment;
- `contexts.csv` and `evaluation_plan.csv` with ordered context values and seeds;
- `model.zip`, the SB3 PPO checkpoint;
- `episode_returns.csv`, one tidy row per evaluated episode, including run,
  method, context, seed, length, return, and termination type;
- `context_returns.csv`, one aggregate row per split and context.

Standalone evaluation loads `contexts.yaml` and `evaluation_plan.csv` from the
checkpoint directory by default. This prevents later configuration edits from
silently changing the evaluation set.
