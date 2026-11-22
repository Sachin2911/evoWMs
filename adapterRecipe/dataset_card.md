# Adapter recipe search

Records of the search for a way to evolve a small safety adapter on a frozen, reward-only
DreamerV3 agent (the checkpoints released with SafeDreamer, Huang et al., ICLR 2024), on
Safety-Gymnasium image tasks. The runs were made on the Wits mscluster on 21 and 22
November 2026 (RTX 3060 and RTX 3090 nodes, GPU rendering).

Nothing in the agent is retrained. At every decision the frozen actor proposes an action
and the adapter changes it:

    action = proposed * (1 + gate * gain) + gate * shift

The gate is a small cost network's chance of a cost within the next 15 decisions; the
network is trained on the unadapted agent's first 25 episodes. Gain and shift are weighted
sums of what the adapter reads. The weights start at zero and are found by CMA-ES.

What the adapter reads is the main difference between the rounds:

- `const`, `probs`, `hidden`, `deter`: nothing, the cost network's three chances, its hidden
  units, or the world model's whole state.
- `look`: look-ahead inputs. The frozen world model is rolled forward a few decisions with
  the proposed action held, and again with each part of the action pushed up and down. The
  adapter reads the chance of a cost for the proposed action, how each push changes that
  chance, and how each push changes the predicted reward (52 weights for a 2-part action).

## Folders

One folder per sweep, and inside it one folder per run:

    <task>-<round>-<how candidates are scored>-<what the adapter reads>-<gate>-<ranking>[-variant]-s<search seed>

- `round1/`: adapters that read the cost network only. `round1_other/`: the same on CarGoal1
  and PointPush1.
- `check_look/`, `check_look2/`: adapters with hand-set weights on the look-ahead inputs,
  10 episodes each. A check of the inputs, not a way of making adapters.
- `round3/`, `round3_more/`, `round3b/`: look-ahead adapters on PointGoal1, by ranking, by how
  candidates are scored, by gate and by look-ahead length. `round3_other/`: the same recipe
  on CarGoal1, PointPush1, PointGoal2 and PointButton1.
- `round4/`, `round4_more/`: gate on, three ways of steering the search inside the world
  model (`feedback`, `hold`, `priced`), each with a final re-check of the best centres on
  fresh episodes. `round4_none/`: the same with no gate. `round4_danger/`,
  `round4_none_extra/`: variants of where imagined roll-outs start and of which network the
  look-ahead asks (cancelled part-way). `round4_other/`: no gate on the other four tasks.
- `round5/`: no gate, with a limit that relaxes slowly or not at all.
- `round6/`: a gate that is fully open once the chance of a cost reaches 5, 10 or 20%.
- `round7/`, `round7b/`, `round7c/`: the two recipes that did best (`priced` with no gate,
  and with the 10% gate) on all five tasks with more seeds. `round7b` has 200,000 steps.
- `round8_no_look/`: the 10%-gate priced recipe without the look-ahead inputs.
  `round8_170k/`: the same recipe with 170,000 steps.

The recipe that got under the cost limit: look-ahead inputs, no gate, the `priced`
ranking, search inside the world model, final re-check (`*-imagined-look-none-priced-*`).

In `round4*` the adapter kept as `final` is the one with the best reward minus cost above
the limit on the fresh episodes. From `round5` on `final` is the best safe-first (within
the limit first, then reward) and the other is kept as `final_penalty`.
- `code/`: the code each run used, one folder per version. Every run's `CODE_VERSION`
  names its version and the git commit it sits on.

Each sweep folder has `summary.txt` (the table printed by `summarise.py`) and `runs.csv`
(one line per kept adapter). Runs that were cancelled or that failed are included with
whatever they had written. A run with no `finished.json` did not reach its evaluation; a
run that was stopped during its evaluations has a `finished.json` that lists only the
adapters whose evaluation was complete.

## Files of a run

- `run.txt`: the label, the task and the options the run was started with.
- `settings.json`: every setting, and the unadapted agent's reward and cost over its first
  25 episodes (seeds 1000 to 1024).
- `cost_net.npz`: the cost network the adapter reads through.
- `generations.jsonl`: one line per generation (or per round, for searches inside the world
  model): simulator steps used so far, every candidate's score inside the world model, and
  the simulator score of those that played.
- `validations.jsonl`: the centre of the search on the validation seeds (3000 to 3009)
  every few generations, with the steps used by then.
- `means/generation_NNNN.npy`: the centre's weights at each validation.
- `finals.json` (runs with a final re-check): the best validated centres on the same fresh
  episodes (seeds 4000 and up). These episodes are counted in the steps used.
- `adapter_final.npz` (and `adapter_final_penalty.npz`): the centre that was best on those
  fresh episodes. `adapter_penalty.npz`, `adapter_safe_first.npz`: the validated centre that is best by
  reward minus cost above the limit, and the one that is best safe-first, judged by the
  validation episodes alone. Each file holds the weights and the cost network.
- `evaluation_final.jsonl`, `evaluation_penalty.jsonl`, `evaluation_safe_first.jsonl`: the
  kept adapters on the evaluation seeds 0 to 49, the seeds the released checkpoints were
  measured on in `safedreamer-checkpoint-evals`. One line per episode: seed, reward, cost.
- `finished.json`: steps and episodes used, time taken, and the kept adapters' averages.
- `task.log`: the run's output.

For the hand-set checks a run has `adapter.npz` and `episodes.jsonl` only.

## Counting steps

A run's simulator steps are the unadapted agent's first 25 episodes, every candidate's
episodes, every validation episode and the fresh episodes of the final re-check: 1,000
steps per episode. The final 50 evaluation episodes are not counted, as for the baselines.
The cost limit is SafeDreamer's: 2 per episode.
