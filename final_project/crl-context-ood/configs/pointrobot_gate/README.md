# Dense Semi-Circle PointRobot gate

This environment is inspired by the Semi-Circle PointRobot hidden-goal task
used in ContraBAR. This implementation was written independently for this
project; no ContraBAR source code was copied. It differs by using dense reward,
fixed-horizon single-episode interaction, state observations, explicit
continuous train/ID/OOD angle splits, and a later frozen-encoder benchmark.

It is a contextual MDP: the goal angle changes the reward function, while the
state space, action space, and transition dynamics remain fixed. Reward is
computed from the **post-transition** clipped position. OOD-left and OOD-right
remain separate and are descriptive only. The YAML files in this directory are
predeclared, immutable gate specifications.

