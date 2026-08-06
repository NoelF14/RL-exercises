# Codex PointRobot primary specification report

## Outcome

The frozen primary experiment and result-only analysis are implemented. No full dataset was collected, no encoder was trained, and no PPO job was launched. The downstream matrix correctly refuses even a dry run until the immutable full dataset and all ten validated same-dataset encoder checkpoints exist.

The existing scientific status remains unchanged: gate v1 REJECT, probe audit v2 PASS, and integration pilot technical PASS. OOD remains a descriptive/scientific outcome and has no selection or tuning role.

## Files changed

- `configs/pointrobot_primary/spec.yaml`: complete machine-readable frozen specification.
- `src/crl_ood/pointrobot_primary/__init__.py`: primary package marker.
- `src/crl_ood/pointrobot_primary/spec.py`: strict spec, dataset, encoder, checkpoint, seed-pairing, and timestep validators.
- `src/crl_ood/pointrobot_primary/run.py`: dataset planning/collection, ten-run encoder matrix, encoder validation, twenty-job downstream matrix, per-episode evaluation, context metrics, provenance, and job-level resume.
- `src/crl_ood/analysis/analyze_pointrobot_primary.py`: result-only final analysis and plotting.
- `scripts/run_pointrobot_primary.py`: execution entry point.
- `scripts/analyze_pointrobot_primary.py`: analysis entry point.
- `tests/test_pointrobot_primary.py`: synthetic primary specification, matrix, provenance, metric, analysis, bootstrap, dependency, and preservation tests.
- `results/pointrobot_primary/protected_before.sha256`: sorted protected baseline.
- `results/pointrobot_primary/protected_after.sha256`: sorted protected after-manifest.
- `codex_pointrobot_primary_spec_report.md`: this report.

No existing completed artifact or existing configuration was modified.

## Frozen scientific specification

The full dataset budget is inherited and checked against `configs/pointrobot_encoders/primary.yaml`: 400 trajectories for each of the five training contexts, hence 2,000 episodes and 100,000 transitions. It uses only angles `[-0.6, -0.3, 0.0, 0.3, 0.6]`, matched context-independent action sequences, the four-action orthogonal prefix followed by isotropic i.i.d. uniform random actions, episode-level train/validation assignment, and normalization fitted only on training-assignment episodes from training contexts. ID and OOD are excluded from fitting and checkpoint selection.

The encoder matrix is exactly VAE and contrastive crossed with seeds 0--4. Both retain the matched one-layer GRU backbone with hidden size 64, latent size 8, 14,536 retained parameters, H=5, K=5, and 20,000 updates. The VAE and hard-negative contrastive objectives, weights, temperature, optimizer, and train-context validation selection rule are frozen unchanged.

The downstream matrix is exactly four methods crossed with policy seeds 0--4: 20 unique jobs requesting 200,000 PPO timesteps. Learned policy seed `s` must use the best checkpoint from the same method's encoder seed `s`; each learned method must therefore provide five distinct checkpoint paths. No-context and oracle share those five policy seeds.

Each final evaluation uses 10 deterministic episodes per individual goal angle. It persists return mean and episode standard deviation, success rate, final and minimum distances, first-success timestep, and episode count. Train, ID, OOD-left, and OOD-right remain separate. Near OOD (`|angle|=0.8`), far OOD (`|angle|=1.0`), side, and every signed angle remain visible.

## Why five paired end-to-end seeds

Encoder training and PPO training both contribute randomness. Pairing VAE encoder seed `s` with VAE policy seed `s`, and likewise for contrastive, makes each seed a complete end-to-end replicate while holding the seed index aligned across methods. The five seed-level VAE-minus-contrastive values—not contexts and not episodes—are the independent differences used for reporting. The deterministic paired bootstrap uses analysis seed 20260806 and 10,000 resamples; its 95% intervals are explicitly labeled low-sample descriptive intervals.

## Exact commands

All commands run from the repository root.

Full dataset collection (not run in this task):

```bash
env LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib .venv/bin/python scripts/run_pointrobot_primary.py dataset
```

Ten-run encoder matrix, resumable at run boundaries (not run):

```bash
env LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib .venv/bin/python scripts/run_pointrobot_primary.py encoder-matrix --resume
```

Encoder and checkpoint validation:

```bash
env LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib .venv/bin/python scripts/run_pointrobot_primary.py validate-encoders
```

Exact twenty-job dry run (intentionally unavailable until dataset and encoder validation pass):

```bash
env LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib .venv/bin/python scripts/run_pointrobot_primary.py downstream --dry-run
```

Resumable downstream matrix, skipping only jobs already marked complete (not run):

```bash
env LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib .venv/bin/python scripts/run_pointrobot_primary.py downstream --resume
```

Primary result-only analysis:

```bash
env MPLCONFIGDIR=/tmp/pointrobot_primary_mpl .venv/bin/python scripts/analyze_pointrobot_primary.py --results-dir results/pointrobot_primary --spec configs/pointrobot_primary/spec.yaml
```

For a collection-free confirmation of the dataset plan:

```bash
env LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib .venv/bin/python scripts/run_pointrobot_primary.py dataset --dry-run
```

## Readiness and refusal rules

The encoder matrix refuses without `results/pointrobot_primary/dataset/full/dataset.sha256` and matching dataset JSON metadata. The downstream matrix refuses unless all ten runs are complete and every run has matching method/seed/source/dataset provenance, training-context-only selection, OOD exclusion, retained parameter count, explicit best checkpoint, manifest hash, and internal checkpoint payload. Checkpoint hashes are recomputed before downstream planning. Missing or partial provenance is an error.

Requested timesteps are validated against full PPO rollout completion. With the frozen rollout quantum 2,048, a request of 200,000 must persist exactly 200,704 actual timesteps.

## Expected artifacts

Before analysis, the primary namespace will contain:

- `dataset/full/{trajectories.npz,dataset.json,dataset.sha256}`;
- ten `encoders/{vae,contrastive}/seed_{0..4}` run directories with best/final checkpoints, losses, selection records, manifests, provenance, and completion logs;
- twenty `downstream/{method}/seed_{0..4}` directories with model, `evaluation_episodes.csv`, `context_metrics.csv`, `training_progress.csv`, complete provenance, and completion log.

Analysis produces all twelve requested tables/findings files and all eight requested plots:

- `primary_context_results.csv`, `primary_summary_by_seed.csv`, `primary_summary_across_seeds.csv`;
- paired return and success gaps, oracle closure, OOD degradation, near/far OOD, checkpoint manifest, and budget verification CSVs;
- `primary_findings.json`, `primary_findings.md`;
- return, success, final-distance, oracle-closure, ID-vs-OOD, near-vs-far, paired VAE-vs-contrastive, and training-curve PNGs.

## Preservation hashes

Both sorted manifests cover 469 files: every pre-existing configuration and every file under the four completed namespaces `pointrobot_gate`, `pointrobot_probe_audit`, `pointrobot_encoders`, and `pointrobot_encoder_pilot_v1`.

The before and after manifests are byte-identical. Their SHA-256 is:

```text
80285a5caab208d6c7fd839e36b714b483ec144a7863ba42db67d4a39180a320
```

## Verification

- Primary synthetic tests: 11 passed.
- Combined primary plus existing matched-encoder regression tests: 29 passed.
- Dataset dry run: reported 2,000 planned episodes and 100,000 planned transitions; collected nothing.
- Downstream dry run before prerequisites: refused on the missing immutable full-dataset checksum, as required.
- `git diff --check`: passed.
- `compileall`: passed after granting temporary bytecode-cache write access.

No full dataset, real encoder update, or PPO training step was executed.

## Remaining risks

- Five seeds support paired estimation but remain a small sample; bootstrap intervals are descriptive, not strong asymptotic guarantees.
- The pilot's asymmetric and potentially catastrophic OOD behavior may persist. The analyzer retains all such seeds, sides, distances, and angles.
- Full dataset collection and the 10 encoder plus 20 PPO runs remain computationally expensive and unexecuted.
- The frozen source commit is the repository HEAD at implementation time (`41b796cd315013c5398ff15727a05d263a7393d6`); the primary implementation itself is currently an uncommitted worktree change and should be committed without altering the frozen scientific fields before launching the real experiment.
