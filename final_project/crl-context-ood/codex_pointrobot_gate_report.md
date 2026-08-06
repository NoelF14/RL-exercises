# Dense Semi-Circle PointRobot gate infrastructure report

## Outcome

Implemented and validated the Dense Semi-Circle PointRobot environment, its
immutable 12-run gate matrix, sequential/resumable PPO execution, paired
evaluation, history-identifiability ridge probe, result-only analyzer, plots,
provenance, and focused tests. No real 200,000-step job was launched and no
`results/pointrobot_gate/` artifact was created.

## 1. Files changed

New configuration/documentation:

- `configs/pointrobot_gate/{README.md,gate.yaml,matrix.yaml}`
- `codex_pointrobot_gate_report.md`

New implementation and entry points:

- `src/crl_ood/pointrobot_gate/{__init__.py,environment.py,spec.py,matrix.py,run.py,probe.py}`
- `src/crl_ood/analysis/analyze_pointrobot_gate.py`
- `scripts/{run_pointrobot_gate.py,run_pointrobot_probe.py,analyze_pointrobot_gate.py}`

New tests and manifests:

- `tests/test_pointrobot_gate_{environment,matrix_probe,analysis,preservation}.py`
- `manifests/pointrobot_gate_protected_{before,after}.sha256`

Updated:

- `README.md` (candidate direction and attribution)
- `pyproject.toml` (three CLI entry points)

The pre-existing dirty `src/crl_ood/goal_pilot/environment.py` change and all
pre-existing mechanistic-audit/report artifacts were not modified by this work.

## 2. Environment equations and defaults

State is `p=(x,y)`, action is clipped to `a in [-1,1]^2`, and context is a
fixed episode angle `phi`. With `R=1.0`:

```text
goal(phi) = R [cos(phi), sin(phi)]
p_next = clip(p + 0.1 * clip(action, -1, 1), -1.5, 1.5)
reward = -(||p_next - goal(phi)||^2 + 0.01 ||clip(action,-1,1)||^2)
```

Reward is explicitly post-transition. Start is exactly `[0,0]`; reset noise is
supported symmetrically but defaults to zero. Horizon is exactly 50 with no
early termination; the episode truncates after step 50. Success is inclusive
at distance `<=0.10`. Hidden observations are exactly `[x,y]`; oracle
observations are exactly `[x,y,cos(phi),sin(phi)]`. Hidden `info` is empty until
the final action has been selected, then carries only success/final/minimum
distance and first-success timestep—not angle, coordinates, or context ID.

## 3. Attribution

The domain is inspired by the Semi-Circle PointRobot hidden-goal task used in
ContraBAR. This implementation was written independently for this project and
does not copy ContraBAR code. It differs through dense reward, fixed-horizon
single-episode interaction, state observations, explicit continuous train/ID/
OOD splits, and planned frozen-encoder benchmarking. It is a contextual MDP
because context changes reward while state/action spaces and dynamics remain
fixed. CARLCartPole remains the predeclared second environment;
HalfCheetahVel was not implemented.

## 4. Tests run and results

- Focused unit/synthetic/probe/tiny-PPO/tiny-evaluation suite: `20 passed`.
- Full repository unit and smoke regression suite: `90 passed`, 34 existing
  dependency/runtime warnings.
- Tiny PPO smoke: 128 steps, one episode per each of 13 split goals; validated
  model, SB3 logs, success metrics, and distance metrics.
- Synthetic analyzer: pass fixture, primary-gate failure fixture, and OOD-only
  non-interference fixture passed; all required tables/plots were written.
- Synthetic probe: paired context-independent actions, state/history isolation,
  circular error, all CSVs, and plot passed.
- `python -m compileall -q src scripts tests`: passed.
- `git diff --check`: passed.
- Matrix dry run: passed; all 12 jobs pending, concurrency 1.

The local environment needs the Nix `LD_LIBRARY_PATH` shown below for Torch and
`MPLCONFIGDIR` for a writable Matplotlib cache.

## 5. Protected-artifact hash verification

Both sorted manifests contain 620 files and are byte-identical. Each manifest
has SHA-256:

```text
4c99c61b48f74be3ae8a3e47c16f214108946ebe50d2661ef95c9d2d383e2d6e
```

The preservation tests independently recomputed every protected file hash.
Thus all requested files under `configs/phase0.yaml`, `configs/diagnostic/`,
`configs/audit/`, `configs/goal_pilot/`, and the five protected result trees
remain byte-identical.

## 6. Unique gate job count

Exactly **12 unique atomic runs**: 4 contextual, 6 specialist, and 2
fixed-center oracle. The two center hidden specialists are reused as the
fixed-center hidden baselines and are not duplicated.

## 7. Exact dry-run command

```bash
env LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib \
  MPLCONFIGDIR=/tmp/pointrobot_mpl \
  .venv/bin/python -u scripts/run_pointrobot_gate.py \
  --matrix-config configs/pointrobot_gate/matrix.yaml --dry-run
```

## 8. Exact one-job command

```bash
env LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib \
  MPLCONFIGDIR=/tmp/pointrobot_mpl \
  .venv/bin/python -u scripts/run_pointrobot_gate.py \
  --matrix-config configs/pointrobot_gate/matrix.yaml \
  --job-id contextual__all_train__hidden__seed_0
```

## 9. Exact resumable matrix command

```bash
env LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib \
  MPLCONFIGDIR=/tmp/pointrobot_mpl \
  .venv/bin/python -u scripts/run_pointrobot_gate.py \
  --matrix-config configs/pointrobot_gate/matrix.yaml --resume
```

## 10. Exact probe command

```bash
env LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib \
  MPLCONFIGDIR=/tmp/pointrobot_mpl \
  .venv/bin/python -u scripts/run_pointrobot_probe.py \
  --config configs/pointrobot_gate/gate.yaml \
  --output-dir results/pointrobot_gate/probe
```

## 11. Exact analysis command

```bash
env LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib \
  MPLCONFIGDIR=/tmp/pointrobot_mpl \
  .venv/bin/python -u scripts/analyze_pointrobot_gate.py \
  --results-root results/pointrobot_gate \
  --config configs/pointrobot_gate/gate.yaml \
  --output-dir results/pointrobot_gate/analysis
```

## 12. Expected output files

Each atomic training directory contains:

- `resolved_config.yaml`, `environment_spec.yaml`, `contexts.yaml`,
  `contexts.csv`, `evaluation_plan.csv`, `metadata.json`,
  `source_provenance.json`, `seed.txt`, and `run.log`;
- `model.zip`, `sb3_logs/progress.csv`, and `sb3_monitor.csv`;
- `training_metrics.csv`, `episode_returns.csv`, `context_returns.csv`,
  `success_metrics.csv`, and `distance_metrics.csv`.

Probe output contains:

- `probe_results_by_seed.csv`, `probe_history_length_summary.csv`,
  `probe_predictions.csv`, and `probe_error_vs_history.png`.

Analysis output contains:

- `pointrobot_gate_seed_results.csv`,
  `pointrobot_contextual_gaps_by_seed.csv`,
  `pointrobot_train_id_summary.csv`,
  `pointrobot_specialist_transfer_by_seed.csv`,
  `pointrobot_specialist_summary.csv`,
  `pointrobot_fixed_center_comparison.csv`,
  `pointrobot_probe_summary.csv`, `pointrobot_ood_descriptive.csv`,
  `pointrobot_gate_findings.json`, and `pointrobot_gate_findings.md`;
- `return_by_goal.png`, `success_by_goal.png`,
  `specialist_transfer_heatmap.png`, `paired_oracle_gap.png`,
  `probe_error_vs_history.png`, and `training_curves.png`.

## 13. Unresolved risks

- The real 200,000-step matrix and full configured probe were deliberately not
  run, so learnability, contextual gaps, and gate acceptance remain unknown.
- PPO adequacy at 200,000 steps and the chosen exploratory probe policy must be
  assessed from completed results; the infrastructure cannot guarantee a pass.
- Two seeds support descriptive paired checks only; no confidence intervals are
  computed.
- The analyzer intentionally refuses incomplete matrices and missing probe
  outputs. OOD-left/right are still reported but cannot affect acceptance.
- The explicit Nix shared-library path is machine-specific and may differ on
  another execution host.
