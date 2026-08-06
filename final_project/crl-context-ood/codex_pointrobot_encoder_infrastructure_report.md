# PointRobot matched encoder infrastructure report

## Outcome

Shared, immutable PointRobot trajectory data, exactly matched GRU history encoders, VAE and contrastive objectives, frozen-latent PPO integration, locked pilot matrices, result-only analysis, and preservation tests are implemented. Only a tiny dataset, one update per encoder, frozen diagnostics, and 64-step learned-method PPO smokes were run. No full dataset, encoder pilot, encoder pretraining, or 200,000-step PPO job was launched.

The original Dense Semi-Circle PointRobot gate v1 remains **REJECT** and the separate probe-validity audit v2 remains **PASS**. Neither finding nor any completed result was changed.

No separate uploaded proposal or ContraBAR design document was present in the repository; the implementation follows the matched-encoder requirements in this task and the repository's existing independently implemented ContraBAR-inspired PointRobot documentation.

## Files changed

- Configs: `configs/pointrobot_encoders/{primary,downstream}.yaml`.
- Dataset/encoders/runners: `src/crl_ood/pointrobot_encoders/{__init__,dataset,models,training,wrapper,downstream,cli}.py`.
- Result-only analysis: `src/crl_ood/analysis/analyze_pointrobot_encoders.py`.
- Entry points: `scripts/{run_pointrobot_encoders,analyze_pointrobot_encoders}.py` and two additions to `pyproject.toml`.
- Tests: `tests/test_pointrobot_encoder_infrastructure.py`.
- New artifacts only under `results/pointrobot_encoders/`.
- This report: `codex_pointrobot_encoder_infrastructure_report.md`.

The pre-existing untracked `codex_goal_pilot_report.md` and `codex_oracle_audit_report.md` were not touched.

## Protected-artifact hashes

The sorted before and after manifests contain 863 files and cover every configuration that existed before this work plus `results/phase0`, `results/phase0_diagnostic`, `results/phase0_audit`, `results/goal_pilot`, `results/goal_pilot_mechanistic_audit`, `results/pointrobot_gate`, and `results/pointrobot_probe_audit`.

- `results/pointrobot_encoders/protected_before.sha256`: `6bea510ecc1fb9a4a2b706c2458f5c8cc10050666a231128cd3479a45a702c70`
- `results/pointrobot_encoders/protected_after.sha256`: `6bea510ecc1fb9a4a2b706c2458f5c8cc10050666a231128cd3479a45a702c70`
- `cmp`: passed; every protected file is byte-identical.
- Source commit recorded by the dataset: `061b65769f5d25e47328ab47d7f3e5780f8fae28`.

## Dataset format, indexing, and behavior policy

An immutable dataset directory contains `dataset.json`, `dataset.sha256`, and `trajectories.npz`. The NPZ arrays persist `states [E,T+1,2]`, `actions [E,T,2]`, `rewards [E,T]`, `next_states [E,T,2]`, `terminated [E,T]`, `truncated [E,T]`, `contexts [E]`, `timesteps [E,T]`, `trajectory_seeds [E]`, `episode_ids [E]`, `assignments [E]`, and seven-dimensional normalization mean/std. JSON records the exact schema, environment specification, behavior policy, source commit, contexts, trajectory seeds, assignment encoding, counts, and canonical dataset checksum.

Only goals `[-0.6,-0.3,0.0,0.3,0.6]` are collected or used to fit normalization and encoders. Split assignment is at episode level. ID/OOD contexts occur only in later frozen diagnostics.

At decision time `t`, the history is transitions `max(0,t-H) ... t-1`, each encoded as `[s_i,a_i,r_i,s_{i+1}]`; the future target begins at `t`. Thus the current action and reward are absent. Histories are right-padded after the valid prefix with explicit mask and length. `s_t` stays outside the latent and is concatenated directly by the PPO wrapper.

The primary behavior policy uses the fixed prefix `[(1,0),(0,1),(-1,0),(0,-1)]`, followed by iid isotropic `Uniform([-1,1]^2)` actions. A trajectory seed generates one sequence reused byte-for-byte across all five matched goals. It is context-independent and has no goal-conditioned action. `random_only` is implemented but not selected or run as an ablation.

The tiny smoke dataset has 10 episodes, 500 transitions, and checksum `bd56dc244545ca20ce81a2f0e162df30706330c7227e8e6359cfe266d6001a04`.

## Shared architecture and parameter counts

Both methods use the same normalized seven-value transition construction, batches, `H=5`, one-layer GRU with hidden size 64, shared `64 -> 8` latent head, deterministic `encode` API, masks/lengths, Adam, update budget, validation schedule, and checkpoint payload. Supported unrun history ablations are `[1,3,5,10]`. Empty history maps to the exact all-zero eight-vector for both methods.

| Method | Retained backbone | Method-specific training parameters | Total training parameters |
|---|---:|---:|---:|
| VAE | 14,536 | 2,839 | 17,375 |
| Contrastive | 14,536 | 2,312 | 16,848 |

The retained GRU/latent backbone matches exactly. Method-specific log-variance, decoder, or future projection parameters are reported and discarded for PPO.

## VAE architecture and objective

The VAE adds an eight-dimensional log-variance head. Training uses `z = mean + exp(0.5 logvar) * epsilon`; deterministic frozen use returns the mean. A decoder receives only `z_t`, current state `s_t`, and the known next five actions. It predicts five two-dimensional state deltas and five rewards; true context is absent.

The fixed objective is `state_MSE + reward_MSE + 0.001 * KL(q(z|h)||N(0,I))`. State and reward validation losses are persisted separately. Best-checkpoint selection uses only the total weighted validation objective on held-out episodes from the five training contexts.

## Contrastive architecture, objective, and negatives

The contrastive method projects `[s_t, future states, known future actions, future rewards]` to an eight-dimensional normalized future embedding. InfoNCE compares the normalized history latent with its true future block and a hard-negative block at fixed temperature `0.1`.

For every primary hard negative, future states and actions are unchanged. Rewards are recomputed with the exact PointRobot equation under the next different goal in the five-goal training list. Construction raises if the reward target is not different. Context labels are used only for this relabel choice and provenance; they never enter the encoder. Persisted rows include positive/negative goals, episode/timestep, state/action preservation, and reward-difference verification. No ID/OOD goal can become a fitting negative. Ordinary in-batch negatives are supported as an unrun ablation.

## Training, checkpoints, and frozen PPO

Every run persists resolved config, source commit, dataset checksum, method/seed, parameter counts, validation and component losses, learning rate, gradient norm, negative provenance, best/final checkpoints, selection record, checkpoint hashes, and run log. Atomic nonempty directories are refused; datasets cannot be overwritten. Sequential pilot execution skips complete runs and can resume only from a provenance-matching final checkpoint.

`FrozenHistoryObservation` loads one explicit checksum-verified checkpoint, switches to eval mode, disables gradients, owns a per-environment history deque, clears it on reset, and appends only after `env.step` completes. Learned observations have dimension 10 (`state 2 + latent 8`); hidden and oracle baselines retain dimensions 2 and 4. Context labels are never exposed. PPO provenance stores both checkpoint and dataset checksums.

The integration plan is methods `[no_context,oracle,vae,contrastive]`, seeds `[0,1]`, and 200,000 steps. Learned jobs require explicit checkpoint paths and dataset checksum. The full-primary seeds `[0,1,2,3,4]` matrix is deliberately unavailable until the integration pilot is validated and the encoder specification frozen. OOD is descriptive only throughout.

## Tests and tiny smoke results

- Focused encoder suite: `18 passed`.
- Complete repository suite: `121 passed, 34 warnings`.
- `compileall`: passed.
- `git diff --check`: passed.
- Protected before/after comparison: passed for all 863 files.
- Tiny VAE: one update, validation total `1.3384074` (`state=0.0588337`, `reward=1.2795630`, `KL=0.0107400`).
- Tiny contrastive: one update, validation InfoNCE `0.7110893`, accuracy `0.42578125`.
- Frozen train/ID/OOD diagnostics were produced with `diagnostic_only` selection role.
- Frozen VAE and contrastive PPO integrations each completed only 64 training steps, plus deterministic evaluation.

These smoke numbers establish execution and artifact integrity only; they are not scientific performance evidence and were not used for tuning.

## Exact commands

Dataset dry run:

```bash
env LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib .venv/bin/python scripts/run_pointrobot_encoders.py --config configs/pointrobot_encoders/primary.yaml dataset --budget tiny --dry-run
```

Small and full datasets (full was not run):

```bash
env LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib .venv/bin/python scripts/run_pointrobot_encoders.py --config configs/pointrobot_encoders/primary.yaml dataset --budget small
env LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib .venv/bin/python scripts/run_pointrobot_encoders.py --config configs/pointrobot_encoders/primary.yaml dataset --budget full
```

VAE and contrastive pilot matrices (not run):

```bash
env LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib .venv/bin/python scripts/run_pointrobot_encoders.py --config configs/pointrobot_encoders/primary.yaml pilot --dataset results/pointrobot_encoders/datasets/small --method vae --resume
env LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib .venv/bin/python scripts/run_pointrobot_encoders.py --config configs/pointrobot_encoders/primary.yaml pilot --dataset results/pointrobot_encoders/datasets/small --method contrastive --resume
```

Frozen-checkpoint evaluation:

```bash
env LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib .venv/bin/python scripts/run_pointrobot_encoders.py --config configs/pointrobot_encoders/primary.yaml evaluate-frozen --dataset results/pointrobot_encoders/datasets/small --checkpoint /ABSOLUTE/PATH/TO/best.pt --output results/pointrobot_encoders/frozen_evaluations/METHOD_seed_SEED_all_splits
```

Two-seed 200,000-step downstream matrix (prepared, not run):

```bash
env LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib .venv/bin/python scripts/run_pointrobot_encoders.py --config configs/pointrobot_encoders/primary.yaml downstream --downstream-config configs/pointrobot_encoders/downstream.yaml --matrix integration_pilot --vae-checkpoint /ABSOLUTE/VAE/best.pt --contrastive-checkpoint /ABSOLUTE/CONTRASTIVE/best.pt --dataset-checksum DATASET_SHA256
```

Add `--dry-run` to print the exact eight-job plan without execution.

Encoder analysis:

```bash
env LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib MPLCONFIGDIR=/tmp/pointrobot_encoder_mpl .venv/bin/python scripts/analyze_pointrobot_encoders.py --results-dir results/pointrobot_encoders --config configs/pointrobot_encoders/primary.yaml
```

## Expected outputs

Dataset outputs are the NPZ, JSON, and checksum described above. Each encoder run produces `resolved_config.yaml`, `provenance.json`, `losses.csv`, `negative_pair_provenance.csv`, `checkpoint_selection.json`, `checkpoint_manifest.json`, `best.pt`, `final.pt`, and `run.log`. Frozen evaluation produces `latents.npz`, `latent_index.csv`, `reward_predictions.csv` where applicable, and `provenance.json`. PPO runs produce `model.zip`, `episode_returns.csv`, `provenance.json`, and `run.log`.

Analysis produces `encoder_training_summary.csv`, `encoder_validation_losses.csv`, `encoder_parameter_counts.csv`, `encoder_latent_statistics.csv`, `encoder_context_probe_by_seed.csv`, `encoder_context_probe_summary.csv`, `encoder_reward_prediction_by_context.csv`, `encoder_checkpoint_manifest.csv`, `encoder_findings.md`, `training_validation_curves.png`, `latent_context_scatter.png`, `probe_error_by_method.png`, and `reward_prediction_by_context.png`. Its module imports no Gym/Gymnasium, CARL, Stable-Baselines3, or Torch.

## Unresolved scientific and engineering risks

- One-update smoke losses and 64-step PPO runs say nothing about representation or control quality; the prespecified two-seed encoder and downstream pilots remain necessary.
- VAE posterior collapse and domination by easy dynamics prediction remain risks; separate reward/state reporting makes them visible but does not prevent them.
- Reward-relabel negatives may be too easy or too hard at some trajectory geometries. The in-batch ablation must remain secondary and cannot be selected using ID/OOD return.
- The linear context probe is diagnostic only and may understate nonlinear information. It must never select checkpoints.
- Torch checkpoint portability across library versions and long-run interruption/restart behavior need validation at pilot scale.
- The full-primary matrix must stay locked until pilot artifacts, checkpoint provenance, and the frozen encoder specification are reviewed.
