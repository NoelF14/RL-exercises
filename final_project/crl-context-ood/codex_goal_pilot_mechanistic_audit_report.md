# Goal-pilot mechanistic target-angle audit

The audit is complete. No real training was launched, no encoder was added, no
gate threshold was changed, and no completed config or result tree was edited.
The original predeclared gate remains **REJECT** and is not reinterpreted here.

## Main findings

- Target angle `0.0` is now a literal identity case: four paired episodes and
  800 deterministic paired steps had identical initial/returned observations,
  underlying state, reward, termination flags, truncation flags, and episode
  lengths. Maximum observed reward difference was exactly `0.0`.
- Native Gymnasium Pendulum computes reward from the **pre-transition state**.
  The observed timing probe reward differed from the pre-state formula by
  `2.38e-11`, versus `1.04e-2` for the post-state formula. The target wrapper
  uses the same convention.
- CARLPendulum observations are ordered `[cos(theta), sin(theta), theta_dot]`.
  Across resets, manual states, 128 random transitions, and values adjacent to
  both `-pi` and `+pi`, maximum wrapped reconstruction error was `6.57e-8`.
  The explicit swapped-sin/cos guard failed as intended.
- The target reward matched the declared formula over 405 cases spanning goals
  `-0.6`, `0.0`, `+0.6`, boundary angles, target-adjacent angles, velocities,
  and actions. Maximum formula error was `1.62e-8`; maximum mirror error was
  `2.66e-15`.
- Six paired manual transitions were exactly mirror-symmetric in state and
  observation. Reset symmetry was deliberately tested separately.
- CARLPendulum resets are severely asymmetric. In 10,000 fixed-seed resets,
  every sampled `theta` and every sampled `theta_dot` was positive. Observed
  supports were `[0.0000895, 3.1415342]` and `[0.0000943, 0.9998080]`.
  Mean initial distances to goals `-0.6`, `0`, and `+0.6` were approximately
  `2.053`, `1.563`, and `1.079` radians, respectively.
- The 100-seed checkpoint reevaluation only partially confirms the original
  five-episode failures. Three of six specialists remain below `-300`:
  negative-goal seed 0 (`-497.65`) and both positive-goal seeds (`-731.68`,
  `-499.30`). Negative-goal seed 1 now passes at `-256.03`; both center runs
  pass. The contextual pattern is not reproduced: oracle wins train and ID for
  seed 0, while it loses train and ID for seed 1.
- Saved training curves show that positive-goal seed 0, both contextual oracle
  runs, and both fixed-center oracle runs were still improving over the final
  50,000 steps. Positive-goal seed 1 is the only highlighted run classified as
  collapsed. The failures therefore are not explained by wholesale PPO
  collapse; they show non-convergence and seed sensitivity, with one clear
  collapse.

## Discovered defects and validity decision

Two implementation issues were identified:

1. CARL 1.1.1's `CARLPendulum.reset()` uses one-sided `uniform(high=...)`
   sampling for angle and angular velocity. This was measured and reported
   exactly; CARL behavior was not silently changed.
2. The target-zero wrapper formerly reconstructed reward from float32
   observations, producing differences of a few ulps from native Pendulum. The
   wrapper now returns the native reward when `target_angle == 0.0`, making the
   identity invariant exact. This numerical issue was far too small to explain
   any completed pilot result.

The completed pilot does **not** need to be invalidated and rerun for its stated
within-CARLPendulum comparison: all methods experienced the same persisted
reset behavior, the reward/dynamics mechanics are correct, and the predeclared
gate already rejected the setup. It must not be interpreted as a study under
the standard symmetric Pendulum reset distribution. If that distribution is a
scientific requirement, it needs a separately predeclared future experiment;
this audit intentionally does not prepare or launch one.

The target-angle implementation is mechanically correct after the exact-zero
identity fix. PPO is seed-sensitive and not converged at 300k steps for several
runs; calling every failure “training collapse” would be inaccurate.

## Exact invariants tested

- Same reset seeds and deterministic actions imply identical target-zero and
  original CARL observations, internal states, rewards, flags, and lengths.
- Native reward is closer to the pre-transition formula than the post-transition
  formula, and target reward caches the pre-transition observation.
- `atan2(obs[1], obs[0])` matches normalized internal theta; swapping indices
  produces a material error.
- Reward is non-positive, follows wrapped `theta - target_angle`, and is
  invariant under simultaneous sign reversal of theta, velocity, action, and
  target.
- Mirrored states and actions produce mirrored internal transitions without
  assuming mirrored resets.
- The persisted reevaluation seed set has exactly 100 unique new seeds and is
  paired across checkpoints and goals. Evaluation always uses deterministic
  prediction and never calls a training method.
- The result-only analyzer has no imports of CARL, Gym/Gymnasium, SB3, Torch,
  NumPy, or pandas.
- Every one of the 607 protected files matches its baseline SHA-256.

## Files changed

- `src/crl_ood/goal_pilot/environment.py`
- `src/crl_ood/mechanistic_audit/{__init__.py,environment_audit.py,reevaluate.py,result_analysis.py}`
- `scripts/{run_goal_pilot_mechanistic_audit.py,reevaluate_goal_pilot.py,analyze_goal_pilot_mechanistic_audit.py}`
- `tests/test_mechanistic_{environment_audit,reevaluation,result_analysis,preservation}.py`
- `manifests/mechanistic_audit_protected_{before,after}.sha256`
- `results/goal_pilot_mechanistic_audit/` (new diagnostic tree only)
- `codex_goal_pilot_mechanistic_audit_report.md`

Pre-existing untracked reports and manifests were left untouched.

## Protected-artifact hashes

Both manifests cover 607 files and have SHA-256:

`3cecf044e056b5e0ab4cf01b8c07dcb38adb9b496f04046b11febf1f1033fde7`

The before and after manifests are byte-identical (`cmp` passed).

## Exact audit commands

```bash
find configs/phase0.yaml configs/diagnostic configs/audit configs/goal_pilot \
  results/phase0 results/phase0_diagnostic results/phase0_audit \
  results/goal_pilot -type f -print0 | sort -z | xargs -0 sha256sum \
  > manifests/mechanistic_audit_protected_before.sha256

LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib \
MPLCONFIGDIR=/tmp/crl_mechanistic_mpl \
.venv/bin/python scripts/run_goal_pilot_mechanistic_audit.py \
  --output-dir results/goal_pilot_mechanistic_audit/environment \
  --reset-samples 10000 --reset-seed-offset 4000000

LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib \
.venv/bin/python scripts/reevaluate_goal_pilot.py \
  --results-root results/goal_pilot \
  --output-dir results/goal_pilot_mechanistic_audit/evaluation \
  --episodes 100 --seed-offset 5000000

.venv/bin/python scripts/analyze_goal_pilot_mechanistic_audit.py \
  --pilot-root results/goal_pilot \
  --audit-root results/goal_pilot_mechanistic_audit \
  --output-dir results/goal_pilot_mechanistic_audit/analysis
```

The after manifest used the same `find | sort | xargs sha256sum` command with
the output redirected to `mechanistic_audit_protected_after.sha256`, followed
by `cmp -s` against the baseline.

## Outputs

Environment diagnostics:

- `environment_audit_findings.json`
- `reset_distribution_summary.csv`
- `reset_theta_histogram.png`
- `reset_theta_dot_histogram.png`
- `initial_distance_to_goal.csv`

Evaluation diagnostics:

- `evaluation_seeds.csv`
- `specialist_own_goal_episode_returns.csv`
- `specialist_own_goal_summary.csv`
- `contextual_train_id_episode_returns.csv`
- `contextual_train_id_summary.csv`

Result-only analysis:

- `training_curve_summary.csv`
- `reevaluation_confirmation.csv`
- `result_only_report.md`

## Tests and results

- Targeted environment, evaluation-only smoke, synthetic result-analysis, and
  preservation tests: **15 passed**.
- Complete repository suite: **70 passed**, with 34 non-fatal dependency/CARL
  warnings.
- `.venv/bin/python -m compileall -q src scripts tests`: passed.
- `git diff --check`: passed.
- Final protected-manifest `cmp`: passed.

Only the repository's existing tiny test fixtures performed any training.
