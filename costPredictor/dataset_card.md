---
pretty_name: Rendering check and cost-predictor test on SafeDreamer's released checkpoints
tags:
- safe-reinforcement-learning
- world-models
- safety-gymnasium
---

# Rendering check and cost-predictor test

Records of two experiments on the checkpoints released with SafeDreamer
(Huang et al., ICLR 2024), on five Safety-Gymnasium tasks: PointGoal1, CarGoal1,
PointButton1, PointPush1 and PointGoal2. Both ran on rented RTX 3060 machines on
21 November 2026, in an environment built to match the one used for
[safedreamer-checkpoint-evals](https://huggingface.co/datasets/Sachioster/safedreamer-checkpoint-evals).

## rendering_check/

Does it matter whether the agent's camera images are rendered on the GPU or the CPU?

The twelve actor-based checkpoints (five reward-only DreamerV3, seven BSRP-Lag) were each
played for 50 episodes (seeds 0 to 49) four times: twice with GPU rendering (`egl-1`,
`egl-2`) and twice with CPU rendering (`osmesa-1`, `osmesa-2`). Each folder has one
`.jsonl` file per checkpoint with one line per episode: seed, reward, cost, decisions,
seconds. `comparison.txt` is the analysis.

- CPU rendering replayed 600 of 600 episodes exactly; GPU rendering 568 of 600.
- The averages do not differ: no checkpoint by more than 2 standard errors in reward or cost.
- CPU rendering took 4 to 5 times longer per episode.

## cost_predictor/

Can an upcoming cost be predicted from the internal state of a world model that was
trained on reward only?

- `recordings/<checkpoint>.tar`: one archive per reward-only checkpoint, 200 episodes each
  (seeds 1000 to 1199, GPU rendering). Inside are four folders, `chunk0` to `chunk3`, of 50
  episodes each, as `evaluate.py --record --record-latents` wrote them:
  - `episodes.jsonl`: one line per episode.
  - `layouts.jsonl`: where every object starts.
  - `steps/seed_NNNN.npz`: every simulator step: action, reward, cost, positions.
  - `latents/seed_NNNN.npz`: the world model's state at every observation: `deter`
    (512 numbers, float16) and `stoch` (48 categorical choices, stored as indices 0 to 47).
- `results/<checkpoint>/results.json`: scores of every predictor.
- `results/<checkpoint>/latent_{linear,network}_h{1,5,15,25}.npz`: weights of the predictors
  that read the full state, with the mean and spread used to scale their inputs.
- `summary.txt`: all tasks side by side.

A predictor answers "will there be a cost within the next H decisions?" for H = 1, 5, 15
and 25. Whole episodes are split 120 / 30 / 50 into training, validation and test. The score
is AUROC on the test episodes: the chance that a decision followed by a cost gets a higher
score than one that is not (0.5 is chance).

AUROC of a network with 16 hidden units, on decisions where the agent was not already
being charged:

| Task | H = 5 | H = 15 | H = 25 |
| --- | --- | --- | --- |
| PointGoal1 | 0.989 | 0.955 | 0.898 |
| CarGoal1 | 0.992 | 0.947 | 0.868 |
| PointButton1 | 0.957 | 0.854 | 0.761 |
| PointPush1 | 0.998 | 0.994 | 0.976 |
| PointGoal2 | 0.976 | 0.923 | 0.850 |

The world model's encoder reads only the two 64×64 camera images; cost is not an input.

Limits: the episodes are the reward-only agent's own behaviour, with one checkpoint
(training seed 0) per task, one split of the episodes and one training run per predictor.

## Versions

SafeDreamer commit `529221f0`, Safety-Gymnasium commit `064c0c3f`, JAX 0.3.25,
MuJoCo 2.3.3, CUDA 11.8, with `--xla_gpu_autotune_level=0`. Checkpoints from
`Weidong-Huang/SafeDreamer` at revision `80ef19ee`.
