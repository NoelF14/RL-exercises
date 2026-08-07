# PointRobot frozen representation-analysis specification report

## Outcome

The final diagnostic representation-quality workflow is implemented and frozen for the ten already-selected PointRobot primary encoder checkpoints. The real checkpoint evaluation was deliberately not executed. No encoder or PPO policy was trained, no checkpoint was selected or reselected, and no artifact was written under `results/pointrobot_primary`.

The lightweight dry run, synthetic result analysis, full repository test suite, compile checks, whitespace checks, and protected-artifact verification were run.

## Files changed

- `configs/pointrobot_representation/spec.yaml`: machine-readable frozen protocol and exact checkpoint hashes.
- `src/crl_ood/pointrobot_representation/spec.py`: dependency-light specification, checkpoint-plan, and primary-provenance validation.
- `src/crl_ood/pointrobot_representation/manifest.py`: internal SHA-256 manifest creation and verification.
- `src/crl_ood/pointrobot_representation/evaluation.py`: future frozen trajectory generation, latent extraction, independent linear probes, state-only probes, and independent PCA fits. This is the only new module that imports Torch/environment code.
- `src/crl_ood/pointrobot_representation/cli.py`: lightweight dry-run CLI with lazy loading of the heavy evaluation module.
- `src/crl_ood/analysis/analyze_pointrobot_representation.py`: result-only aggregation, validation, control joins, findings, figures, and final manifests.
- `scripts/run_pointrobot_representation.py`: evaluation/dry-run entry point.
- `scripts/analyze_pointrobot_representation.py`: result-only analysis entry point.
- `tests/test_pointrobot_representation.py`: synthetic and read-only provenance regression tests.
- `results/pointrobot_representation/protected_before.sha256`: sorted 702-file protected baseline.
- `results/pointrobot_representation/protected_after.sha256`: sorted 702-file protected after-manifest.
- `codex_pointrobot_representation_spec_report.md`: this handoff report.

Pre-existing untracked reports `codex_goal_pilot_report.md` and `codex_oracle_audit_report.md` were left untouched.

## Frozen analysis protocol

The analysis is diagnostic only. Its results cannot participate in checkpoint or configuration selection. The authoritative primary source snapshot is `6b9cd43da2cb9e276b6c772e1047435f142de1c5`; the completed primary data and encoder runs record execution source `41b796cd315013c5398ff15727a05d263a7393d6`. The primary specification checksum is `60bdc763b14860b2c3b243b99c45f278f4dc1dbef48c9ba82b50e72d5de67724`, and the immutable dataset checksum is `cb826e04b344eb875662b8775b89f9c60bdb9bae895f25a260d25ef422a589fa`.

Contexts are exactly:

- train: `[-0.6, -0.3, 0.0, 0.3, 0.6]`
- ID: `[-0.45, -0.15, 0.15, 0.45]`
- OOD-left: `[-1.0, -0.8]`
- OOD-right: `[0.8, 1.0]`

Every latent uses deterministic frozen inference with dimension 8. At decision time `t`, the encoder receives only normalized completed transitions from `max(0,t-5)` through `t-1`, using the trained seven-field transition representation `(state, action, reward, next_state)`. The current state is separately persisted for the state-only control. At `t=0`, the exact trained empty-history path is required to return the zero latent.

Each signed angle is retained. OOD `|angle|=0.8` is labeled near and `|angle|=1.0` far, with left and right never pooled away. Encoder seeds 0–4 are the five independent replicates within each method; contexts, trajectories, timesteps, and samples are never treated as seeds.

## Exact checkpoint mapping

| Method | Encoder seed | Frozen `best.pt` SHA-256 |
|---|---:|---|
| VAE | 0 | `035b57cbe66170dd2469dfc51daee35fed89d2073ef360f7c15e3d2d5ca538f6` |
| VAE | 1 | `de891ac65c3a88c94bccdba85872d7e3647a29bf7cae4d7499d34934cd486398` |
| VAE | 2 | `f0a332bd03d4f0b1677165f665bab9d26dcc404c47921495ca9bd2a167d96cc5` |
| VAE | 3 | `176e8cfbc1a839cb00f6f8f72efef576ee5d4ff5def10896a59a778d186f6ea0` |
| VAE | 4 | `2acccd4332b69ab4f8ceb027756515aa0a693f08d03f253d280ad6701e0d691a` |
| contrastive | 0 | `136a3884c4dda6454e9864f5681a2549bd9af495b8cb3562f9d623c7a330c20d` |
| contrastive | 1 | `9db27375aca8fe960c88005ae9e01786174efcdd79679d17d3671521ebd5c08b` |
| contrastive | 2 | `0b679ce4f8688a428148c60e6fe38a85ca865390d3e6cb93d87435032de6fdf4` |
| contrastive | 3 | `76be02b86b792f2a870d5b9df6400d30daba981c12f3efc54dbb0bda6af3e7f1` |
| contrastive | 4 | `30358029dda439a54d7b328013a554a05721e4d4c24c3dcae650fdd7340e7234` |

The rule is exactly `results/pointrobot_primary/encoders/{method}/seed_{seed}/best.pt`. Before future evaluation, each file is rehashed and checked against the new frozen spec, its run-local `checkpoint_manifest.json`, and `primary_encoder_checkpoint_manifest.csv`. Method, seed, dataset, execution source, held-out-training-only checkpoint selection, and `ood_used=false` provenance are also required. There is no search or reselection code in this workflow.

## Diagnostic trajectory protocol

The evaluator reuses the existing PointRobot encoder-evaluation semantics and immutable primary action arrays. It takes the first 40 primary action sequences, fixed at trajectory seeds 41000 through 41039. Each sequence consists of the trained `orthogonal_then_isotropic` context-independent policy: the four fixed orthogonal actions followed by seeded IID uniform actions in `[-1,1]^2`.

The same 40 action arrays and trajectory IDs are used for both methods, all encoder seeds, and every signed context. The analyzer rejects evaluation inputs whose `(split, goal angle, trajectory ID, trajectory seed, timestep)` keys differ across checkpoints. Each 50-step episode contributes all decision times 0 through 49. States and context-dependent rewards are produced with the existing `DenseSemiCirclePointRobot` implementation. PPO trajectories are forbidden. The shared action arrays, rolled-out states/rewards, trajectory index, seeds, behavior policy, dataset checksum, and configuration checksum are persisted under `results/pointrobot_representation/evaluation_protocol/`.

This produces 13 contexts × 40 trajectories × 50 timesteps = 26,000 samples per checkpoint and 260,000 latent samples overall.

## Probe fitting rule

For each checkpoint independently, ordinary least squares with an intercept predicts the signed goal angle directly from its eight latent coordinates. Coefficients are fitted only on rows whose context split is `train`; the frozen coefficients are then applied to train, ID, OOD-left, and OOD-right. Error is

`abs(atan2(sin(prediction - target), cos(prediction - target)))`.

The state-only OLS control uses only current-state `x,y` from the exact same diagnostic samples. It receives no transition, action, reward, history length as a feature, or latent. Per-sample predictions/errors, per-angle and per-seed/split metrics, and five-seed descriptive summaries are retained. Probe results have diagnostic-only status and cannot select anything.

## PCA fitting rule

For each checkpoint independently, the latent mean and two principal axes are fitted only on train-context samples. Standardization is frozen to `none`. The train-fitted transform is applied without refitting to every split. Means, components, explained variances, and explained-variance ratios are persisted in NPZ/CSV outputs. No axis rotation, sign alignment, Procrustes transform, averaging, or other cross-seed latent alignment is performed. Seed 0 is prespecified for the compact figure; a separate 2×5 figure retains all five seeds for both methods.

## Output schema

Evaluation-time per-checkpoint CSV rows carry method, encoder seed, split, signed goal angle, trajectory ID/seed, timestep, actual history length, predictions/errors as applicable, and fitting-scope provenance. Latent rows additionally carry `z_0` through `z_7`, dataset checksum, absolute checkpoint path, checkpoint SHA-256, authoritative source snapshot, and representation-configuration checksum. Per-checkpoint NPZ files retain OLS coefficients and independent PCA parameters.

After verifying `representation_evaluation_files.sha256`, the result-only analyzer produces:

- `representation_latent_index.csv`
- `representation_probe_predictions.csv`
- `representation_probe_by_angle.csv`
- `representation_probe_by_seed.csv`
- `representation_probe_summary.csv`
- `representation_state_only_probe.csv`
- `representation_pca_coordinates.csv`
- `representation_pca_summary.csv`
- `representation_checkpoint_manifest.csv`
- `representation_control_by_seed.csv`
- `representation_findings.json`
- `representation_findings.md`

It also produces 300-DPI PNG and vector PDF versions of `probe_mae_by_split`, `probe_mae_by_angle`, `latent_pca_seed_0`, `latent_pca_all_seeds`, and the three directional `probe_vs_return_*` figures.

The control join reads only `primary_summary_by_seed.csv` and `primary_oracle_gap_closure.csv`. For each learned method and paired seed it retains ID, OOD-left, and OOD-right probe MAE, corresponding primary return, and oracle-gap closure. Existing ID closure is used directly; directional OOD closure is calculated descriptively from the already-persisted learned/no-context/oracle seed-level returns. No inferential correlation test is run at n=5.

After a real evaluation, the workflow creates and verifies:

- `representation_evaluation_files.sha256` over frozen evaluator outputs;
- `representation_analysis_files.sha256` over result-only tables, findings, PNGs, and PDFs;
- `representation_final_files.sha256` over both manifests and all final analysis artifacts.

## Dry-run output

The dry run printed one line for each mapping above, in VAE seeds 0–4 then contrastive seeds 0–4 order, with its exact absolute checkpoint path, frozen SHA-256, and output directory:

```text
FROZEN REPRESENTATION PLAN: 10 checkpoints; no Torch/environment loading; no execution
01..05 vae seeds 0..4 -> results/pointrobot_representation/evaluations/vae/seed_S
06..10 contrastive seeds 0..4 -> results/pointrobot_representation/evaluations/contrastive/seed_S
evaluation_manifest=results/pointrobot_representation/representation_evaluation_files.sha256
analysis_output=results/pointrobot_representation
```

It loaded neither Torch nor an environment, and it did not create evaluation artifacts.

## Future commands

Exact future real frozen evaluation command (not run in this task):

```bash
env LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib PYTHONPATH=src .venv/bin/python scripts/run_pointrobot_representation.py evaluate --spec configs/pointrobot_representation/spec.yaml --root .
```

Exact future result-only analysis command, run only after the evaluation manifest exists:

```bash
env MPLCONFIGDIR=/tmp/pointrobot_representation_mpl PYTHONPATH=src .venv/bin/python scripts/analyze_pointrobot_representation.py --results-dir results/pointrobot_representation --spec configs/pointrobot_representation/spec.yaml --primary-dir results/pointrobot_primary
```

The plan can be reprinted without heavy dependencies with:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_pointrobot_representation.py dry-run --spec configs/pointrobot_representation/spec.yaml --root .
```

## Preservation

Protected coverage is exactly the requested six namespaces: `configs/pointrobot_primary`, `results/pointrobot_primary`, `results/pointrobot_gate`, `results/pointrobot_probe_audit`, `results/pointrobot_encoder_pilot_v1`, and `results/pointrobot_encoders`.

- Before manifest entries: 702
- After manifest entries: 702
- SHA-256 of each sorted manifest: `be1767fb430c0701a27f13d3c14c7b518a8c727f8fcd4f4de41be7a6f4608d9c`
- Byte comparison: identical
- Every listed protected file was rehashed successfully.

## Verification performed

- New focused suite: 11 passed.
- Full repository suite: 143 passed, 34 third-party warnings.
- Result-only subprocess import isolation: passed; Torch, Gym/Gymnasium, CARL, and Stable-Baselines3 were absent from `sys.modules`.
- Lightweight dry-run subprocess isolation: passed.
- Synthetic end-to-end result-only aggregation, joins, figures, and all three manifest types: passed.
- Evaluation-manifest tamper rejection: passed.
- `compileall`: passed.
- `git diff --check`: passed.
- Protected before/after comparison and individual SHA-256 verification: passed.

Only tests, synthetic fixtures in pytest temporary directories, dry runs, compilation, diff checking, and preservation verification were executed. No real representation evaluation, encoder training, PPO training, or PPO rollout was run.

## Remaining scientific risks

- A linear probe can miss nonlinear context information; it is intentionally a simple, fixed diagnostic.
- Direct signed-angle OLS is appropriate for the frozen `[-1,1]` radian range, while circular MAE guards the evaluation metric, but the fit itself is not a circular-regression model.
- The fixed diagnostic behavior is matched and context independent, but representation quality remains conditional on its state-action visitation distribution.
- Reward is a history feature, so goal identifiability grows with completed transitions; the empty and short-history rows may dominate some aggregate views unless history-stratified checks are consulted.
- Directional OOD is extrapolation beyond the train range and should remain descriptive.
- PCA axis signs are arbitrary even within each independent fit, and axes are deliberately not comparable through alignment across seeds.
- Five seeds support descriptive variability, not reliable inferential correlation claims; control joins must not be presented as hypothesis tests.
- The future evaluator is frozen to the current existing environment/evaluation semantics. Its evaluation manifest and source/configuration checksums must be retained so later code changes cannot silently mix outputs.
