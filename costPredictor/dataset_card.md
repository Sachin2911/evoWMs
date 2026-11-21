---
pretty_name: Rendering check, cost-predictor test and adapter pilot on SafeDreamer's released checkpoints
tags:
- safe-reinforcement-learning
- world-models
- safety-gymnasium
---

# Rendering check, cost-predictor test and adapter pilot

Records of experiments on the checkpoints released with SafeDreamer
(Huang et al., ICLR 2024), on five Safety-Gymnasium tasks: PointGoal1, CarGoal1,
PointButton1, PointPush1 and PointGoal2. All ran on rented GPU machines on
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

### cost_predictor/efficiency/

How few episodes does the predictor need? The same predictors, trained on the first 10, 25,
50, 100 or 150 of the recorded episodes (in a fixed shuffled order, 80% for training and 20%
for validation) and scored on the same 50 test episodes.

- `<checkpoint>/episodes_<N>/results.json`: scores of every predictor.
- `summary.txt`: all tasks side by side.

AUROC of the network reading the 512 continuous numbers, on decisions where the agent was
not already being charged, at H = 5 / 15 / 25:

| Task | 10 episodes | 25 episodes | 150 episodes |
| --- | --- | --- | --- |
| PointGoal1 | 0.953 / 0.901 / 0.819 | 0.986 / 0.924 / 0.849 | 0.994 / 0.969 / 0.918 |
| PointButton1 | 0.880 / 0.715 / 0.642 | 0.932 / 0.790 / 0.693 | 0.971 / 0.879 / 0.789 |

`summary.txt` has every task and episode count. One episode is 1,000 simulator steps.

## adapter_pilot/

Can a small adapter, evolved on top of the frozen reward-only agent, make it safe on
PointGoal1, and how many simulator steps does that take?

At every decision the frozen actor proposes an action and the adapter changes it:
`action = proposed × (1 + gain) + shift`, where gain and shift are weighted sums of what the
adapter reads from the world model's state. Its weights start at zero and are searched with
CMA-ES. A run has a budget of 1,000,000 simulator steps, which includes the 25 episodes used
to train the small cost network the adapter reads through, every candidate's episodes (12
candidates × 2 episodes per generation, seeds from 2000) and a validation of the search's
centre every 4 generations (10 episodes, seeds 3000 to 3009). The code is in `adapterPilot/`
of the project repository; the runs here used commits `2276076`, `1c8abf1` and `ccce317`.

- `runs/<run>/settings.json`: every option of the run.
- `runs/<run>/cost_net.npz`, `cost_net.json`: the cost network and its scores.
- `runs/<run>/generations.jsonl`: one line per generation, with every candidate's place and
  its scores inside the world model and in the simulator.
- `runs/<run>/validations.jsonl`: the search's centre on the validation seeds.
- `runs/<run>/means/generation_NNN.npy`: the search's centre at each validation.
- `runs/<run>/adapter_half_budget.npz`, `adapter_full_budget.npz`: the best validated
  adapter within 500,000 steps and within the whole budget.
- `runs/<run>/evaluation_adapter_*.jsonl`: those two adapters on seeds 0 to 49, one line
  per episode.
- `logs/run-<run>.log`: what the run printed.
- `summary.txt`: all runs side by side.

A run is named `<reads>-<scoring>[-penalty<weight>]-s<evolution seed>`:

- reads: `hidden` (the cost network's 16 hidden units, 68 weights), `deter` (the 512
  continuous numbers, 2,052 weights) or `gated` (as `hidden`, with gain and shift multiplied
  by the predicted chance of a cost within 5 decisions).
- scoring: `real` (every candidate plays in the simulator) or `imagined` (48 candidates are
  scored inside the world model and the best 12 play in the simulator).
- ranking: safe-first (candidates within the cost limit of 2 first, then by reward) unless
  the name has `penalty`, which ranks by reward minus the weight times the cost over the limit.

Result, on seeds 0 to 49 with GPU rendering (adapters kept at the end of the budget,
averaged over evolution seeds):

| Agent | Evolution seeds | Reward | Cost | Episodes within the limit |
| --- | --- | --- | --- | --- |
| Reward-only agent, no adapter | | 26.9 | 54.5 | 2 of 50 |
| BSRP-Lag (three released checkpoints) | | 21.4 to 21.7 | 0.8 to 1.8 | 40 to 46 of 50 |
| hidden, imagined | 3 | 3.0 | 3.0 | 45.3 |
| hidden, real | 3 | −3.6 | 9.5 | 37.0 |
| deter, imagined | 3 | 1.8 | 8.2 | 44.0 |
| deter, real | 3 | −3.0 | 8.4 | 33.3 |
| hidden, imagined, penalty 0.5 | 1 | 17.3 | 26.7 | 14 |
| hidden, imagined, penalty 1.0 | 1 | 8.5 | 22.2 | 40 |
| gated, imagined | 1 | 11.1 | 17.7 | 18 |
| gated, imagined, penalty 0.5 | 1 | 26.8 | 50.4 | 7 |

No adapter comes near BSRP-Lag. With the safe-first ranking the adapters lower the cost by
holding the robot back in every state (the constant part of the forward gain is between
−0.5 and −1.5), which loses nearly all of the reward. The penalty and gated runs were added
afterwards as single-seed diagnostics of that outcome.

Limits: one task, one checkpoint, three evolution seeds for the four main configurations and
one for the diagnostics. Candidates are ranked on 2 episodes each, and episode cost varies
a great deal between episodes.

## Versions

SafeDreamer commit `529221f0`, Safety-Gymnasium commit `064c0c3f`, JAX 0.3.25,
MuJoCo 2.3.3, CUDA 11.8, with `--xla_gpu_autotune_level=0`. Checkpoints from
`Weidong-Huang/SafeDreamer` at revision `80ef19ee`.
