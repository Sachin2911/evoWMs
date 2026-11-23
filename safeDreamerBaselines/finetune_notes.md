# Fine-tuning baseline: SafeDreamer (BSRP-Lag) started from the reward-only agent

`finetune.py` trains SafeDreamer's safe agent for a fixed number of simulator steps, starting from a released reward-only DreamerV3 checkpoint. It answers one question: does SafeDreamer's own safe training, started from the reward-only agent, get under the cost limit of 2 per episode within 250,000 simulator steps, and with what reward?

The script has never been run with SafeDreamer. It parses as Python 3.8, and its own plain-Python parts (the copying rule, the step counting, the handling of an earlier run) pass a check on made-up data. Everything that touches SafeDreamer, JAX or the simulator is untested: see [Unverified](#unverified-try-these-in-the-trial-run).

## Commands

All from `safeDreamerBaselines/`, after the two files are on the cluster (`ms sync`).

**Trial run first** (6,000 steps, perhaps 15 minutes):

```bash
../msCli/ms submit -p bigbatch -t 1h -n finetune-trial -- apptainer exec --nv --bind /datasets/smohan \
    --env MUJOCO_GL=osmesa /home-mscluster/smohan/images/safeDreamer.sif \
    python3 finetune.py --steps 6000 --checkpoint-every 2000 --seed 0 \
    --out /datasets/smohan/safedreamer_finetune/trial_PointGoal1 --checkpoint \
    /datasets/smohan/safedreamer_checkpoints/unsafe_dreamerv3/20240327-144037_dreamerv3_safetygym_SafetyPointGoal1-v0_0.ckpt
```

A 2,000-step trial would test nothing: SafeDreamer's loop spends its first 2,196 simulator steps on random actions to fill the replay buffer, and trains only after that. 6,000 steps leave about 950 training updates and three whole episodes played by the agent.

**Full run**, one per task and seed:

```bash
../msCli/ms submit -p bigbatch -t 3d -n finetune-PointGoal1 -- apptainer exec --nv --bind /datasets/smohan \
    --env MUJOCO_GL=osmesa /home-mscluster/smohan/images/safeDreamer.sif \
    python3 finetune.py --steps 250000 --seed 0 \
    --out /datasets/smohan/safedreamer_finetune/PointGoal1_seed0 --checkpoint \
    /datasets/smohan/safedreamer_checkpoints/unsafe_dreamerv3/20240327-144037_dreamerv3_safetygym_SafetyPointGoal1-v0_0.ckpt

../msCli/ms submit -p bigbatch -t 3d -n finetune-PointButton1 -- apptainer exec --nv --bind /datasets/smohan \
    --env MUJOCO_GL=osmesa /home-mscluster/smohan/images/safeDreamer.sif \
    python3 finetune.py --steps 250000 --seed 0 \
    --out /datasets/smohan/safedreamer_finetune/PointButton1_seed0 --checkpoint \
    /datasets/smohan/safedreamer_checkpoints/unsafe_dreamerv3/20240327-144037_dreamerv3_safetygym_SafetyPointButton1-v0_0.ckpt
```

For several seeds as one sweep, add `--array N` and write `--seed {i}` and `..._seed{i}` in the `--out` folder.

**Evaluating a checkpoint.** `evaluate.py` reads the variant from the file name and would play a `bsrp_lag` checkpoint at 4 simulator steps per decision. This agent was trained at 2, so the option must be given every time:

```bash
../msCli/ms submit -t 1h -n eval-finetune -- apptainer exec --nv --bind /datasets/smohan \
    --env MUJOCO_GL=osmesa /home-mscluster/smohan/images/safeDreamer.sif \
    python3 evaluate.py --env.safetygym.repeat 2 --checkpoint \
    /datasets/smohan/safedreamer_finetune/PointGoal1_seed0/checkpoints/finetune-0250000_bsrp_lag_safetygym_SafetyPointGoal1-v0_0.ckpt
```

## What a run writes

Into the `--out` folder:

| File | Holds |
|---|---|
| `run.json` | The run's settings, every SafeDreamer setting that differs from SafeDreamer's own for `bsrp_lag`, and at the end the totals: simulator steps, episodes, cost paid in training, training updates. |
| `initialisation.json` | Every array copied from the checkpoint and every array left fresh, by name and shape. |
| `episodes.jsonl` | One line per training episode: `simulator_steps` (so far, this episode included), `reward`, `cost`, `episode_steps`, `random_steps`, `seed`, `finished`, `seconds`. The cost paid in training is the sum of `cost` over all lines. |
| `checkpoints/` | `finetune-0000000_bsrp_lag_<task>_<seed>.ckpt` (the agent as initialised), one every 50,000 steps, and `finetune-0250000_...` at the end. About 0.4 GB each. |
| `safedreamer/` | SafeDreamer's own output: `config.yaml`, `metrics.jsonl` (losses, and `train/lagrange_multiplier`, `train/penalty_multiplier`, `train/penalty` every five minutes), `scores.jsonl`, and one `checkpoint.ckpt` of its own from the start of the run, which can be deleted. |

## How SafeDreamer trains `bsrp_lag`, as read from its code

- **Entry point.** `train.py` builds the settings, a logger, a replay buffer, the simulator and the agent, then calls `embodied.run.train_eval`. That is the only training loop in the repository at this commit: the `train`, `train_save` and `train_holdout` branches of `train.py` call functions that do not exist.
- **The agent.** With `use_cost` on, the world model gets a cost head, and the behaviour is `ImagSafeActorCritic`: the actor, a reward critic, a cost critic, and an augmented Lagrangian (`jaxutils.Lagrange`). The Lagrangian holds two numbers, a multiplier that starts at 1e-6 and a penalty multiplier that starts at 5e-9 and grows by a factor of 1.00001 per update. Both are variables of the agent and are saved in its checkpoints. The actor's loss is the reward-only loss plus the Lagrangian's penalty.
- **What the limit of 2 is compared with.** Not the cost of a real episode. At each update the Lagrangian takes the imagined cost returns from the batch's start states, adds them up over the 15 imagined steps, subtracts 2 and averages.
- **The PID Lagrangian** (`PIDLagrangian` in `train.py`) is a second, separate multiplier. The loop updates it from a running average of episode cost once its counter passes 5,000, and hands it to the policy with every observation. Only the planning variant `osrp_lag` reads it. For `bsrp_lag` it has no effect, and it is not saved in checkpoints.
- **The loop.** First it plays random actions, 100 observations at a time, until the replay buffer holds 1,024 sequences of length 64. That takes 1,100 observations. Then it plays the agent, with actions sampled from the actor, and trains after every observation at a rate of `train_ratio / (batch_size × batch_length)` = 512 / 1,024 = one update per two observations.
- **The step counter.** It goes up by one for every observation handed to the agent. That is one per decision, and one more for the observation at each reset. It does not count simulator steps. A decision lasts `env.safetygym.repeat` simulator steps, so a 1,000-step episode at 2 steps per decision adds 501 to the counter, not 1,000 and not 500. `run.steps`, the train ratio and the PID gate of 5,000 all use this counter. Only the logger multiplies it by the repeat, so the `step` in `metrics.jsonl` is close to simulator steps but two too many per episode.

## How the budget is counted here

A step is one call of the simulator's `step`. `finetune.py` replaces the simulator's `reset` and `step` with versions that count, and the count is the budget. Resets cost nothing. The random steps at the start are counted, because they are real simulator steps with real cost.

With 1,000-step episodes, 250,000 steps are 250 episodes, 125,000 decisions, 125,250 on SafeDreamer's counter and about 62,075 training updates. About 2,196 of the steps are random.

SafeDreamer's loop only checks its own limit every 100 observations, so it cannot stop at an exact step. The replaced `reset` and `step` refuse the first call after the budget by raising an error that `finetune.py` catches. So the 250,000th step is taken and the loop finishes with it, training included. At the next simulator call the last checkpoint is saved and the call is refused, which ends the run.

## Settings changed from SafeDreamer's own

The first five rows are entries in SafeDreamer's settings. `run.json` lists every such entry that differs, as the script found it, with SafeDreamer's value and the value used. The other rows are differences in how the pieces are put together, which no entry covers.

| Setting | SafeDreamer's for `bsrp_lag` | Here | Why |
|---|---|---|---|
| `env.safetygym.repeat` | 4 | 2 | The reward-only agent was trained at 2 simulator steps per decision. Its world model predicts two steps ahead and its actor acts at that rate. At 4 the world model would be wrong from the first step, and the run would no longer start from the reward-only agent in any useful sense. The value is read from SafeDreamer's `dreamerv3` settings, not written into the script. |
| `run.eval_every` | 10,000 | 0 (off) | SafeDreamer plays an evaluation episode every 10,000 observations in a second simulator. Those would be simulator steps outside the budget. Checkpoints are evaluated afterwards with `evaluate.py`. |
| `run.save_every` | 900 s | 0 (off) | SafeDreamer rewrites one 0.4 GB checkpoint every 15 minutes. The script saves its own by simulator step. SafeDreamer still writes one at the start. |
| `task`, `seed`, `method`, `logdir` | its defaults | from the command line | |
| `run.steps` | 1e7 | unchanged | Not what ends the run: the budget is. The script stops at the start if this limit could be reached first. |
| Simulator process (`envs.parallel`, `envs.restart`) | its own process, restarted if it fails | inside the script's process, not restarted | The script must reach the simulator to count its steps, and a silent restart would break the count. This is how `evaluate.py` builds it, which is known to work on the cluster. |
| Evaluation simulator and replay buffer | separate ones | the training ones, handed over a second time | The loop demands them. With evaluation off it never steps or fills them, and it skips its random fill of the evaluation buffer because the buffer it is shown is already full. So no simulator step is taken outside the count. Its "eval" report in `metrics.jsonl` is therefore a second report on training data. |
| Replay buffer on disk | written to `logdir/replay` | kept in memory only | Only needed to continue a run, which the script does not do. About 3 to 4 GB of memory. |
| Logger | terminal, two JSON files, TensorBoard | the same without TensorBoard | TensorBoard would store a video of an episode every five minutes. No effect on training. |
| World seeds | not set | episode k uses seed 10,000,000 × (run seed + 1) + k | The worlds of a run can be played again, differ between run seeds, and are never the evaluation worlds (seeds 0 to 49) or those of the adapter work (below a million). |
| `run.from_checkpoint` | loads a whole checkpoint of the same variant | not used | Replaced by the copying below. |

Everything else is SafeDreamer's: learning rates, train ratio 512, batch 16 × 64, imagination horizon 15, cost limit 2, the Lagrangian's starting values and schedule, and the random fill of the replay buffer.

## What is copied from the reward-only checkpoint

The rule: an array is copied if both agents have it in the same place with the same shape and type. Everything else keeps the value the safe agent was built with. "Place" goes inside an optimiser's state, which is many arrays.

**Copied:** encoder, recurrent model, decoder, reward head, continue head; the actor; the reward critic and its slow copy with the slow copy's update counter; the scale of the reward returns; the optimiser state (Adam's running averages) for all of those; each optimiser's step count and gradient scale.

**Fresh:** the cost head; the cost critic, its slow copy and its optimiser; the scale of the cost returns; the Lagrange multiplier and the penalty multiplier; inside the world model's optimiser state, the entries for the cost head.

Choices behind this:

- **A rename is needed.** SafeDreamer names the actor-critic `ac` in the reward-only agent and `safe_ac` in the safe one. Matching by name alone would leave the actor, its optimiser and the return scale fresh without any error. The script maps `agent/task_behavior/ac/` to `agent/task_behavior/safe_ac/`.
- **Two checks, or the run stops before training.** Every array in the checkpoint must find a place, and every array left fresh must belong to one of the four safe-only parts listed at the top of the script. A wrong name, a different network size or a checkpoint of another variant fails one of them. `initialisation.json` is written first, so the reason can be read.
- **Optimiser states are copied, not reset.** A reset Adam moves every weight of a trained network by the full learning rate in its first steps, however small the gradient. Keeping the state means the only thing that changes at step zero is what the safe variant adds.
- **One side effect in the world model's optimiser.** Its Adam step count is a single number shared by all weights, and it is copied. The cost head's running averages start at zero without the usual correction for a low step count, so the cost head's first few hundred updates are up to about six times larger than the learning rate of 1e-4 would give. The factor is down to about 1.3 after 1,000 updates. The other choice, a step count of zero, would instead slow the trained weights, to as little as a sixth of their rate, over the same period. Either way the cost head's new gradient also reaches the shared weights of the world model, as SafeDreamer has it. If reward drops in the first few thousand steps, look here first.
- **The slow reward critic is copied with its update counter.** With the counter at zero SafeDreamer would overwrite the slow copy with the critic at the first update. The slow cost critic starts fresh, and SafeDreamer sets it to the cost critic at the first update, as in training from nothing.
- **SafeDreamer's step counter starts at zero.** It counts this run only.

## Other choices

- **The random fill is kept.** It is part of SafeDreamer's loop and costs under 1% of the budget. It does mean the first updates train the copied world model on random-action data only. Filling with the agent's own actions would be a change to the loop.
- **Checkpoints.** Named so that `evaluate.py` reads variant `bsrp_lag`, the task and the run seed from them. The number in the name is the simulator steps taken. The `step` stored inside is simulator steps divided by steps per decision, which `evaluate.py` reads as decisions, so its `training_simulator_steps` comes out exact. Each is saved at the first simulator call after its step, so all are taken at the same point of the loop. Each is written under a `.part` name first.
- **A run cannot be continued.** The replay buffer is not on disk, and steps redone after a restart would break the count. Started again in the same folder (the cluster does this after a node failure), the script moves what the unfinished run wrote into `interrupted-N/` and starts from the reward-only checkpoint. A finished run is left alone and the script exits at once.
- **Cameras drawn on the CPU** (`MUJOCO_GL=osmesa` in the commands). It works on every node, including the ones whose GPU cannot draw, and the checks in `adapterRecipe/` use it. Drawing is a small part of the time. Evaluate with the same setting as whatever the result is compared with.
- **Output under `/datasets`.** A run writes about 2.8 GB of checkpoints. `ms fetch` on the folder brings back the small files and skips the checkpoints.
- **Any other option is passed on to SafeDreamer's settings**, as in `evaluate.py`, and is then listed in `run.json`.

## Running time

A guess, not a measurement: **2 to 5 hours** for 250,000 steps on a `bigbatch` node. The 3-day limit leaves a wide margin.

- Training updates dominate. There are about 62,075 of them. Each trains the world model (about 26 million weights) on 16 × 64 pairs of images and imagines 1,024 × 15 steps for the actor and two critics. Nobody has timed one on this cluster. At a guessed 5 to 10 updates per second on the node's 24 GB GPU, that is 1.7 to 3.5 hours.
- Playing is measured: about 3.6 s per 1,000-step episode on a `batch` node, so about 15 minutes for 250 episodes, a little more with the cameras on the CPU.
- Start-up and compiling take 5 to 10 minutes. Six checkpoints and a report every five minutes add a few more.

The trial gives the real number. In its `episodes.jsonl`, the difference in `seconds` between the last two lines is the time for 1,000 steps with training. Multiply by 248.

## Unverified: try these in the trial run

In the order they would show up.

1. **The names.** The rename and the four safe-only parts come from reading SafeDreamer's code, not from a built agent. If they are wrong the run stops before any training and prints which arrays have no place or would be left fresh. Expected in the log: every variable `copied` except those under `agent/wm/cost/`, `agent/task_behavior/cost_critic/`, `.../safe_ac/costnorm_extr/` and `.../safe_ac/lagrange/`, which are `fresh`, and `agent/wm/model_opt/state`, which is `partly`.
2. **The optimiser states.** That the checkpoint holds them in the nested form the script walks through, and that SafeDreamer's compiled training step accepts the merged ones.
3. **The settings.** That `--run.eval_every 0` and `--run.save_every 0` are accepted and switch both off. The log's "Settings that differ" should show only `env.safetygym.repeat`, `run.eval_every`, `run.save_every`, `method` and `logdir` (and `task` for a task other than PointGoal1, `seed` for a seed other than 0).
4. **The status line.** That `ms_status.py` can be imported inside the image. If not, the run fails at once with an import error.
5. **The counting.** `episodes.jsonl` should show `random_steps` of 1000, 1000, 196 and then 0, and `simulator_steps` of 1000, 2000, ... 6000. If the total were higher than the lines add up to, the loop would be stepping the simulator for evaluation after all.
6. **Saving in the middle of the loop.** Checkpoints at 0, 2,000, 4,000 and 6,000 steps, saved from inside the replaced simulator calls while SafeDreamer's loop is running.
7. **The end.** That the error which ends the loop passes through SafeDreamer's code untouched, `run.json` says `finished: true` with about 950 training updates, and the job exits with code 0 and does not hang on background threads.
8. **The copy, by behaviour.** Evaluate `finetune-0000000_...ckpt` and the reward-only checkpoint on the same few seeds, both with the cameras on the CPU (`--episodes 3`, and `--env.safetygym.repeat 2` for the first). The initialised safe agent has the reward-only actor and world model, so reward and cost should be the same episode by episode. This is the strongest check that the copy is right.
9. **`evaluate.py` on a trained checkpoint.** That it loads `finetune-0006000_...ckpt` and plays it.
10. **Time and memory.** Seconds per 1,000 steps as above, and that training fits in the node's GPU memory.
11. **Python 3.8 and NumPy 1.23.** The made-up-data check ran on a newer Python and NumPy. Only the syntax was checked against 3.8.

Not something a trial can settle:

- **Whether every episode is 1,000 steps.** If one ends early, the budget ends in the middle of the last episode, and that line has `finished: false`.
- **Whether the Lagrangian gets going in time.** Its multipliers start at SafeDreamer's values for training from nothing, where the actor is almost unconstrained at first, and the pressure builds only as the cost critic learns. If the agent is not under the limit after 250,000 steps, `train/lagrange_multiplier` in `metrics.jsonl` shows whether the constraint was pressing at all.
