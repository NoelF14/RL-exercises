# Third-Party Software

This project depends on external open-source packages. Their upstream license
terms apply to those packages; they are not vendored in this repository.

## CARL

- Package: `carl-bench`
- Project: <https://github.com/automl/CARL>
- License: Apache License 2.0
- Use: contextual `CARLPendulum` environment, context definitions, and context
  selectors.

## Stable-Baselines3

- Package: `stable-baselines3`
- Project: <https://github.com/DLR-RM/stable-baselines3>
- License: MIT License
- Use: PPO implementation, policy, rollout collection, optimization, checkpoint
  serialization, and deterministic policy prediction.

No PPO algorithm code is copied or reimplemented in this repository. Training
delegates directly to `stable_baselines3.PPO`.

Transitive dependencies, including Gymnasium, PyTorch, NumPy, pandas, and
PyYAML, are installed by the Python package resolver and retain their own
licenses. Use `python -m pip licenses` with an appropriate license-reporting
plugin when a complete environment-specific inventory is required.
