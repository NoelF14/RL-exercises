# PointRobot probe-validity and trajectory-identifiability audit report

## Outcome

The separate **PointRobot probe-audit v2 passes** all prespecified train/ID-only
criteria. The original PointRobot gate v1 remains byte-identical and remains
**REJECT**; no v1 finding or threshold was changed.

PointRobot is suitable for beginning matched encoder work. This conclusion is
limited to environment/control adequacy plus probe validity and trajectory
identifiability. No VAE, contrastive encoder, downstream PPO run, or OOD-based
selection was implemented or launched.

## Files changed

New configuration:

- `configs/pointrobot_probe_audit/audit.yaml`

New implementation and entry points:

- `src/crl_ood/pointrobot_probe_audit/{__init__.py,core.py,run.py}`
- `src/crl_ood/analysis/analyze_pointrobot_probe_audit.py`
- `scripts/{run_pointrobot_probe_audit.py,analyze_pointrobot_probe_audit.py}`

New tests and report:

- `tests/test_pointrobot_probe_audit.py`
- `codex_pointrobot_probe_audit_report.md`

New result namespace:

- `results/pointrobot_probe_audit/protected_{before,after}.sha256`
- `results/pointrobot_probe_audit/probe_alignment_audit.json`
- `results/pointrobot_probe_audit/analytic_estimator_by_history.csv`
- `results/pointrobot_probe_audit/behavior_geometry_summary.csv`
- `results/pointrobot_probe_audit/probe_model_results_by_seed.csv`
- `results/pointrobot_probe_audit/probe_model_summary.csv`
- `results/pointrobot_probe_audit/probe_v1_vs_v2_comparison.csv`
- `results/pointrobot_probe_audit/pointrobot_probe_audit_findings.{json,md}`
- `results/pointrobot_probe_audit/{analytic_error_vs_history,full_rank_fraction_vs_history,probe_model_error_vs_history,condition_number_distribution}.png`

The two pre-existing untracked files `codex_goal_pilot_report.md` and
`codex_oracle_audit_report.md` were left untouched.

## Identifiability derivation

Let `p = p_{t+1}`, `lambda = action_penalty`, and `||g|| = 1`. Expanding the
reward gives

```text
r_t = -(||p - g||^2 + lambda ||a_t||^2)
    = -||p||^2 + 2 p^T g - 1 - lambda ||a_t||^2.
```

Therefore

```text
b_t = (r_t + ||p_{t+1}||^2 + 1 + lambda ||a_t||^2) / 2
p_{t+1}^T g = b_t.
```

For a history, the resulting-position rows form `P` and the scalar values form
`b`. The estimator is `g_hat = pinv(P)b`; its normalized two-vector determines
the predicted angle. Rank two is the geometric identifiability condition. The
implementation uses only states, actions, rewards, the unit goal radius, and
the known reward equation. Context labels are used only as supervised targets
or for evaluation, never as estimator/probe inputs.

## Trajectory-alignment and leakage findings

All checks passed across 3,120 generated episode rows and 240 matched-action
groups. The existing exploratory probe was reconstructed from its immutable
configuration and action seeds, and all 3,200 persisted prediction metadata
rows matched reconstructed trajectories.

- Every transition is exactly `(s_t, a_t, r_t, s_{t+1})`.
- Reward uses the post-transition position. The largest absolute formula
  residual was `2.6445833700705634e-07`, attributable to saved float32 state
  observations versus float64 environment arithmetic.
- Suffix histories end at the horizon and contain no future transition.
- Targets were constant within every episode.
- Probe inputs contain only state, action, reward, and next state. Perturbing
  goal/context/split/filename metadata left raw, engineered, and sequence
  features unchanged.
- Behavior-policy action arrays were byte-identical across contexts for every
  matched policy/seed/trajectory group.
- Fitting used only train contexts. ID and the two OOD sides were evaluation
  sets, and the four model families used the same 200 training episodes per
  seed/history.
- The deliberate off-by-one reward negative control was rejected.

## Behavior-policy geometry

The table below reports ID results, which are the only evaluation results used
for the audit decision. OOD-left and OOD-right are present in the saved tables
and plots as descriptive-only results and did not affect policy choice or
acceptance.

| Policy | H | Full rank | Median condition | P90 condition | Analytic ID angle MAE |
|---|---:|---:|---:|---:|---:|
| existing exploratory | 1 | 0% | inf | inf | 0.789456 |
| existing exploratory | 2 | 100% | 21.0799 | 119.419 | 4.29e-7 |
| existing exploratory | 3 | 100% | 14.7188 | 38.3356 | 1.40e-7 |
| existing exploratory | 5 | 100% | 10.3102 | 23.5079 | 8.11e-8 |
| existing exploratory | 10 | 100% | 8.07769 | 16.9470 | 2.51e-8 |
| isotropic random | 1 | 0% | inf | inf | 0.794744 |
| isotropic random | 2 | 100% | 18.9539 | 69.9819 | 3.46e-7 |
| isotropic random | 3 | 100% | 12.3875 | 32.2049 | 1.38e-7 |
| isotropic random | 5 | 100% | 9.74233 | 22.2566 | 9.63e-8 |
| isotropic random | 10 | 100% | 7.48800 | 15.8717 | 2.22e-8 |
| deterministic orthogonal | 1 | 0% | inf | inf | 0.785398 |
| deterministic orthogonal | 2 | 100% | 2.61803 | 2.61803 | 1.41e-9 |
| deterministic orthogonal | 3 | 100% | 2.61803 | 2.61803 | 1.41e-9 |
| deterministic orthogonal | 5 | 100% | 2.23607 | 2.23607 | 1.41e-9 |
| deterministic orthogonal | 10 | 100% | 1.86388 | 1.86388 | 1.41e-9 |

The deterministic orthogonal policy was prespecified before evaluation. Its
four-action symmetric cycle makes the last two resulting positions `(0.1,0)`
and `(0.1,0.1)`, so H=2 spans both goal dimensions and is much better
conditioned than either random policy.

## Analytic-estimator result

At the first eligible history, H=2, the selected orthogonal policy had a 100%
ID full-rank fraction over 320 histories and ID circular MAE
`1.4075771659349812e-09`. This passes both the `>=95%` full-rank requirement
and the `<=0.01` error requirement.

## Probe-model results

All results below are mean ID circular-angle MAE over the same two seeds and
the same prespecified orthogonal-policy dataset.

| Model | H=1 | H=2 | H=3 | H=5 | H=10 |
|---|---:|---:|---:|---:|---:|
| raw-history ridge | 0.047738 | 0.001219 | 0.001219 | 0.000474 | 0.000183 |
| engineered linear | 0.047646 | 0.001383 | 0.001383 | 0.000212 | 0.000034 |
| two-layer MLP | 0.008234 | 0.005048 | 0.004199 | 0.002056 | 0.014313 |
| diagnostic GRU | 0.013166 | 0.001184 | 0.001490 | 0.001814 | 0.002875 |

The state-only ID baseline was `0.300000`. The fixed H=5 MLP reduced it by
99.31% and improved by 75.03% relative to its H=1 error. The fixed H=5 GRU
reduced it by 99.40% and improved by 86.22% relative to its H=1 error. Thus the
nonlinear criterion passes. The MLP and GRU are supervised diagnostics only;
they are not proposed or implemented encoders. At the selected H=5 comparison,
their parameter budgets are of the same order (1,458 for the MLP and 1,234 for
the GRU).

The original v1 raw-history ridge remains separately reported. Its two seeds
had state-only ID MAE `0.300/0.300`, H=1 `0.292995/0.297069`, H=5
`0.289998/0.279390`, and H=10 `0.270544/0.260913`. Its best reductions versus
state-only were only 9.82% and 13.03%, so the unchanged v1 probe criterion and
overall v1 gate remain failed. The new result does not replace that record; it
shows that the old raw ridge combined a varying trajectory design with a model
that could not implement the per-history inverse, despite the trajectories
being analytically identifiable.

## v1 and v2 decisions

- PointRobot gate v1: **REJECT**, unchanged.
- PointRobot probe-audit v2 analytic identifiability: **PASS**.
- PointRobot probe-audit v2 nonlinear learnability: **PASS**.
- PointRobot probe-audit v2 leakage/alignment: **PASS**.
- Overall PointRobot probe-audit v2: **PASS**.

Only train and held-out ID results entered v2. OOD results never affected
acceptance.

## Tests and protected-artifact hashes

- Focused unit/synthetic suite: `13 passed`.
- Complete repository suite: `103 passed`, `34 warnings`.
- `compileall`: passed.
- `git diff --check`: passed.
- The sorted before/after manifests each contain 848 protected files, are
  byte-identical, and each has SHA-256
  `593c99b0fe0b0307fc6bec11f574ce631ceb8c4424e3f1eb5e91b17a8b4baed5`.

The protected scope covers every config that existed before the audit and all
requested trees under `results/phase0`, `results/phase0_diagnostic`,
`results/phase0_audit`, `results/goal_pilot`,
`results/goal_pilot_mechanistic_audit`, and `results/pointrobot_gate`. Every
listed file remained byte-identical.

## Exact commands

Pre-audit preservation manifest:

```bash
mkdir -p results/pointrobot_probe_audit
find configs results/phase0 results/phase0_diagnostic results/phase0_audit results/goal_pilot results/goal_pilot_mechanistic_audit results/pointrobot_gate -type f -print0 | sort -z | xargs -0 sha256sum > results/pointrobot_probe_audit/protected_before.sha256
```

Focused unit and synthetic tests:

```bash
env LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib MPLCONFIGDIR=/tmp/pointrobot_probe_audit_mpl .venv/bin/python -m pytest tests/test_pointrobot_probe_audit.py -q
```

Real probe audit (executed twice; the second execution persisted the final
expanded alignment evidence):

```bash
env LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib MPLCONFIGDIR=/tmp/pointrobot_probe_audit_mpl .venv/bin/python -u scripts/run_pointrobot_probe_audit.py --config configs/pointrobot_probe_audit/audit.yaml
```

Result-only analysis (executed twice; the second execution persisted the final
v1 comparison and condition-number distribution plot):

```bash
env LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib MPLCONFIGDIR=/tmp/pointrobot_probe_audit_mpl .venv/bin/python -u scripts/analyze_pointrobot_probe_audit.py --results-dir results/pointrobot_probe_audit --config configs/pointrobot_probe_audit/audit.yaml
```

After-manifest and byte comparison:

```bash
cut -c 67- results/pointrobot_probe_audit/protected_before.sha256 | xargs -d '\n' sha256sum > results/pointrobot_probe_audit/protected_after.sha256
cmp results/pointrobot_probe_audit/protected_before.sha256 results/pointrobot_probe_audit/protected_after.sha256
```

Compile, whitespace, and complete test suite:

```bash
env LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib MPLCONFIGDIR=/tmp/pointrobot_probe_audit_mpl .venv/bin/python -m compileall -q src scripts tests
git diff --check
env LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib MPLCONFIGDIR=/tmp/pointrobot_probe_audit_mpl .venv/bin/python -m pytest -q
```

No command invoked a PointRobot PPO runner or modified any protected result.
