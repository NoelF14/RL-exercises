# PointRobot primary execution provenance

The primary experiment completed successfully without source changes during
dataset collection, encoder training, PPO training, or analysis.

## Source identity

The run artifacts record the base Git commit:

`41b796cd315013c5398ff15727a05d263a7393d6`

At execution time, the new primary source/configuration/test files were staged
but had not yet been committed. The exact staged tree used during execution was:

`cbe6d3a5cfa1de00b32e7b2cbc456b6a4b47ef27`

That exact tree was subsequently committed without modification as:

`6b9cd43da2cb9e276b6c772e1047435f142de1c5`

The authoritative exact-source tag is:

`pointrobot-primary-execution-source-v1`

The earlier tag `pointrobot-primary-spec-v1` points to the base commit and
must not be cited as the complete source snapshot.

## Experiment identity

- Full dataset checksum:
  `cb826e04b344eb875662b8775b89f9c60bdb9bae895f25a260d25ef422a589fa`
- Dataset file-manifest checksum:
  `fd8e73a8ba5b9713ccbb440028ba823ee598294f0d0c9d78427f6983edcfedf0`
- Encoder artifact-manifest checksum:
  `b38dbe98f1d7495e8e24ca79fc911f20bfa7f526950927581ab7d3e348ad2a95`
- Downstream artifact-manifest checksum:
  `b8bba7352dedc016bac7dd155a03912fd072d33fadd1536c22fe1a72120cf3e2`
- Final primary file-manifest checksum:
  `f4708c6d8f6cc06c28bee9faab8facfc6debba49385c5049ebe824adb1ca1fa4`
- Protected pre-existing artifact-manifest checksum:
  `80285a5caab208d6c7fd839e36b714b483ec144a7863ba42db67d4a39180a320`

## Validation

- Full dataset: 2,000 episodes and 100,000 transitions.
- Encoder matrix: 10 validated runs.
- Downstream matrix: 20 validated jobs.
- Requested PPO budget: 200,000.
- Actual complete-rollout budget: 200,704 for every job.
- Five paired end-to-end seeds.
- Primary analyzer technical status: PASS.
- Complete test suite: 132 passed.
