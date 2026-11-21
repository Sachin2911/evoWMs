---
license: cc-by-4.0
pretty_name: SafeDreamer released checkpoints, evaluated
task_categories:
- reinforcement-learning
tags:
- safe-reinforcement-learning
- safety-gymnasium
- world-models
- dreamerv3
- safedreamer
size_categories:
- 1M<n<10M
configs:
- config_name: checkpoints
  data_files: checkpoints.csv
  default: true
- config_name: episodes
  data_files: episodes.csv
- config_name: steps
  data_files: steps/*.parquet
- config_name: layouts
  data_files: layouts.csv
- config_name: videos
  data_dir: videos
---

# SafeDreamer released checkpoints, evaluated

Per-task results, step-by-step trajectories and videos for 31 of the checkpoints released with
[SafeDreamer](https://openreview.net/forum?id=tsE5HLYtYg) (Huang et al., ICLR 2024), on five
image-based Safety-Gymnasium tasks.

The SafeDreamer paper reports its Safety-Gymnasium results as training curves and aggregate
scores, and the [checkpoint repository](https://huggingface.co/Weidong-Huang/SafeDreamer) holds
weights without numbers. This dataset fills that gap: what each released checkpoint scores, how
long it was trained, and what it does step by step.

This is an independent evaluation. It is not affiliated with the SafeDreamer authors, and it
measures their released checkpoints under the protocol described below. It is not a
reproduction of the training runs behind the paper's figures.

## Results

Mean ± standard deviation over 50 episodes of 1,000 simulator steps. SafeDreamer's cost limit
is 2. "Trained for" is read from the checkpoint itself.

{RESULTS_TABLE}

## What to know before using these numbers

- **The checkpoints are not 2M-step agents.** The paper's main comparison is at 2M simulator
  steps. The released checkpoints were trained for 2.6M to 13.9M, and differ from each other.
  Compare them with the paper's 8M-step curves (its Figure 15), and with each other with care.
- **Most rows are one training seed.** The paper's curves average five runs. Only some
  task and agent pairs have more than one released checkpoint.
- **The reward-only agent acts twice as often.** DreamerV3 takes a decision every 2 simulator
  steps, BSRP-Lag and OSRP-Lag every 4, and OSRP every 5. These are the settings each was
  trained with.
- **Single episodes are noisy, costs especially.** The reward-only agent's cost on one world
  ranges from 0 to several hundred. Use many episodes for any comparison.
- **An episode does not always replay exactly.** Every episode was run twice, in separate jobs,
  and {REPLAYED} episodes gave the identical reward and cost both times. The `replayed_exactly`
  column marks them. How often depends on the task and the agent: on PointButton1 every episode
  replayed, while on PointPush1 the two planning agents almost never did. The count for each
  checkpoint is in `checkpoints`. For the reward-only agent the differences were traced to
  camera images that are occasionally not bit-identical between two runs; over a long episode
  a tiny difference can grow. Other causes have not been ruled out for the planning agents. On
  other hardware expect the same seed to play out differently more often. Averages should
  agree; single episodes may not.

## Files

| Config | File | Rows | One row is |
|---|---|---|---|
| `checkpoints` | `checkpoints.csv` | 31 | a checkpoint, with its averages |
| `episodes` | `episodes.csv` | 1,550 | an episode |
| `steps` | `steps/<task>.parquet` | 1,550,000 | a simulator step |
| `layouts` | `layouts.csv` | about 3,900 | an object in a world, at the start of the episode |
| `videos` | `videos/<task>/*.mp4` | 155 | a video of an episode |

```python
from datasets import load_dataset

episodes = load_dataset("Sachioster/safedreamer-checkpoint-evals", "episodes", split="train")
steps = load_dataset("Sachioster/safedreamer-checkpoint-evals", "steps", split="train")
```

Tables join on `task`, `variant`, `training_seed` and `episode_seed`.

### Columns shared by the tables

- `task`: `PointGoal1`, `CarGoal1`, `PointButton1`, `PointPush1` or `PointGoal2`. The
  Safety-Gymnasium name is `Safety<task>-v0`.
- `variant`: the kind of agent, named as in the SafeDreamer code. `dreamerv3` is the reward-only
  baseline, which never sees the cost. `bsrp_lag` trains a safe actor inside the world model.
  `osrp_lag` and `osrp` plan at every decision, with and without a Lagrange multiplier.
- `training_seed`: the seed in the checkpoint's file name.
- `episode_seed`: 0 to 49. It fixes the layout of the world and the agent's random choices.
  For a given task, seed k is the same world for every checkpoint, so episodes can be compared
  one to one.

### `checkpoints`

`checkpoint_file` and `sha256` identify the file in the checkpoint repository.
`simulator_steps_per_decision` is the action repeat. `training_simulator_steps` is the step
counter stored in the checkpoint (which counts decisions) times the action repeat.
`reward_mean`, `reward_sd`, `cost_mean` and `cost_sd` are over the 50 episodes.
`episodes_with_zero_cost` counts the episodes with no cost at all, and
`episodes_replayed_exactly` those that came out identical in a second run.

### `episodes`

`reward` and `cost` are summed over the episode. `decisions` is how many actions the agent chose.
`replayed_exactly` is true if a second, separate run of the same episode gave the identical
reward and cost. The steps and videos are from the run reported here.

### `steps`

One row per simulator step, 1,000 per episode, describing the world after that step.

- `step`: 1 to 1,000. `decision`: which of the agent's decisions the step belongs to, from 0.
  The action is held for every step of a decision.
- `action_0`, `action_1`: the action the simulator received.
- `reward`, `cost`: for this step. The cost is 1 if anything caused a cost on the step, else 0.
- `cost_hazards`, `cost_vases_contact`, `cost_vases_velocity`, `cost_gremlins`, `cost_buttons`:
  1 if that kind of object caused a cost on the step. A kind the task does not have is always 0.
- `agent_x`, `agent_y`, `agent_heading` (radians, 0 along x), `agent_vx`, `agent_vy`.
- `goal_x`, `goal_y`: the current goal. `goal_met` is true on the step a goal is reached; a new
  goal is then placed, and these columns already show it.
- `movable_xy`: x and y of each object that can move (vases, gremlins, the box), in the order
  `layouts` lists the movable objects of that world.

### `layouts`

Where everything is at the start of an episode: `object` (for example `hazards`), `index`
within its kind, `x`, `y`, `size` (radius or half-width, in metres) and whether it is `movable`.
The robot is the row with object `agent`, which also has a `heading`.

### `videos`

Episodes 0 to 4 of every checkpoint, at real speed. `videos/metadata.csv` lists them with their
reward and cost.

- Left: the world from above, x to the right and y up, as in the tables. The robot is red, the
  goal green, hazards are blue circles, vases cyan, buttons orange, gremlins purple, the box
  yellow. A red border shows while the robot is being charged a cost.
- Right: the two 64×64 camera images the agent actually receives, front and rear, enlarged.

## How the evaluation was run

- **Checkpoints:** image-input checkpoints for the five tasks, from
  `Weidong-Huang/SafeDreamer` at revision `80ef19ee7a7a8a105489100c1dc6e1681df2c97b`. Each
  file's checksum is in `checkpoints`.
- **Code:** SafeDreamer at commit `529221f02b54d5093e304d877eb2aaf6da63a9ff` and the
  PKU-Alignment fork of Safety-Gymnasium at commit `064c0c3f1dac523ffdf1e4daddba8262779344d4`,
  with JAX 0.3.25 and MuJoCo 2.3.3.
- **Agent:** built from each variant's own settings in SafeDreamer's `configs.yaml`, with the
  policy in its evaluation mode. A check confirms every array in the checkpoint has the shape
  those settings expect. For OSRP-Lag the Lagrange multiplier is not stored in the checkpoint;
  it is set with `--pid.init_penalty 0.1`, the value in SafeDreamer's own evaluation command.
- **Observations:** two 64×64 cameras, front and rear, rendered at that size as in training.
- **Episodes:** 50 per checkpoint, 1,000 simulator steps each. Episode k resets the simulator
  with seed k and seeds the agent's action sampling with k.
- **Repeatability:** run on NVIDIA RTX 3060 GPUs with CUDA 11.8, with the simulator's cameras
  rendered on the GPU. With the default settings JAX picks GPU routines by timing them at
  start-up, and the same seed then gave different episodes depending on which routine won, so
  `--xla_gpu_autotune_level=0` is set. What remains is described above.
- **Training length:** the paper's figures show the simulator steps as the checkpoint's step
  counter times the action repeat, and `training_simulator_steps` follows the same rule.

## Licence and citation

The data and videos are released under CC BY 4.0. SafeDreamer and Safety-Gymnasium are
Apache 2.0; none of their code or weights is included here.

If you use this dataset, please cite the work it measures:

```bibtex
@inproceedings{safedreamer,
  title={SafeDreamer: Safe Reinforcement Learning with World Models},
  author={Weidong Huang and Jiaming Ji and Borong Zhang and Chunhe Xia and Yaodong Yang},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=tsE5HLYtYg}
}

@article{ji2023safety,
  title={Safety-Gymnasium: A Unified Safe Reinforcement Learning Benchmark},
  author={Ji, Jiaming and Zhang, Borong and Zhou, Jiayi and Pan, Xuehai and Huang, Weidong and Sun, Ruiyang and Geng, Yiran and Zhong, Yifan and Dai, Juntao and Yang, Yaodong},
  journal={arXiv preprint arXiv:2310.12567},
  year={2023}
}
```
