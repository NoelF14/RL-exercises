# Target-angle context-necessity pilot

This namespace is independent of the completed Phase 0, diagnostic, and audit
namespaces. Real jobs always use 300,000 PPO timesteps and the unchanged Phase
0 PPO hyperparameters. The runner is sequential (`concurrency=1`) and is meant
to run independently of Codex.

The matrix has 14 experimental roles but 12 unique atomic runs. In each seed,
the hidden specialist trained at target `0.0` is explicitly reused as the
fixed-center hidden baseline. It is never trained twice.

Dry-run the full matrix:

```bash
python scripts/run_goal_pilot.py --matrix-config configs/goal_pilot/matrix.yaml --dry-run
```

Run one job (replace the ID with any dry-run row):

```bash
python scripts/run_goal_pilot.py --matrix-config configs/goal_pilot/matrix.yaml --job-id contextual__all_train__hidden__seed_0
```

Resume the sequential matrix, skipping only fully validated jobs:

```bash
python scripts/run_goal_pilot.py --matrix-config configs/goal_pilot/matrix.yaml --resume
```

Analyze complete persisted results without importing CARL, Gym/Gymnasium,
Stable-Baselines3, or Torch:

```bash
python scripts/analyze_goal_pilot.py --results-root results/goal_pilot
```

## Predeclared gate

Only train and ID enter the gate. OOD-left and OOD-right stay separate and are
descriptive only. No confidence interval is computed from two seeds.

- Every specialist/seed own-goal mean return must be at least `-300`.
- Within each seed, specialists must collectively have at least two distinct
  best train/ID evaluation goals; one common best goal for all fails.
- For every seed and each of train and ID, contextual oracle mean return must
  be strictly higher than contextual hidden mean return.
- At target zero in every seed, fixed-center oracle relative degradation must
  be no worse than `-0.25` versus the explicitly reused center specialist.
- Relative oracle improvement is `(oracle - hidden) / (abs(hidden) + 1e-8)`.

Acceptance requires all checks. The numerical thresholds live in `pilot.yaml`
and must not be changed after real runs begin.
