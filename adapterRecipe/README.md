# Adapter recipe

Evolves a small adapter that makes a frozen, reward-only DreamerV3 agent safe. Nothing in
the agent is retrained. Results are in the Lab NoteBook and in the Hugging Face records.

## The adapter

At every decision the frozen actor proposes an action and the adapter changes it:

    action = proposed * (1 + gate * gain) + gate * shift

Gain and shift are weighted sums of what the adapter reads, squashed to stay within
`--output-bound`. The weights start at zero, so the first adapter changes nothing.

What it reads (`--reads`):

- `look`: look-ahead inputs. The frozen world model is rolled forward `--look-horizon`
  decisions with the proposed action held, and again with each part of the action pushed
  up and down by `--look-push`. A small cost network gives the chance of a cost along each
  roll-out, and the world model's reward predictor gives the reward. The adapter reads the
  chance for the proposed action, how each push changes that chance, and how each push
  changes the reward. Away from hazards a push changes nothing, so these inputs are zero
  there by themselves.
- `const`, `probs`, `hidden`, `deter`: no inputs, the cost network's three chances, its
  hidden units, or the world model's whole state. With these the search can only slow the
  agent down.

The gate (`--gate`) is the cost network's chance of a cost soon, or `none` for no gate.
The cost network is trained on the unadapted agent's first 25 episodes.

## The search

`evolve.py` runs one search from start to finish with CMA-ES (diagonal covariance).

Scoring (`--scoring`):

- `imagined`: candidates are scored inside the world model, from states the agent really
  visited, with the same states and random numbers for every candidate. Only the centre of
  the search plays real episodes, `--round-episodes` per round. Those episodes steer the
  search and train the cost model that scores the next round.
- `real`: every candidate plays episodes in the simulator. `prescreen`: the best imagined
  candidates do.

Ranking (`--ranking`), for `imagined` scoring:

- `feedback`: safe-first against a limit that is a share of what the unadapted agent would
  cost from the same states. The share starts at 1. After a round whose real episodes cost
  more than `--feedback-target` it is multiplied by `--feedback-step`; after a round that
  does not, by `--feedback-relax`. With `--feedback-hold` it is only cut once the search
  has met it.
- `priced`: reward kept minus a price times cost kept, both as shares of the unadapted
  agent's. The price is steered the same way.

`safe-first`, `penalty`, `eps` and `floor` rank against the real limit and are described
in `evolve.py`.

`--danger-share` takes that share of the imagined starting states from moments shortly
before a real cost.

## What a run keeps

Every `--validate-every` rounds the centre plays the validation seeds. At the end, with
`--final-centres`, the best validated centres play the same fresh episodes and the one
that is best there, safe-first, is kept as `adapter_final.npz`. It is then played once on
the evaluation seeds 0 to 49, which are not counted as steps used. Everything else is:
the first episodes, the centre's episodes, the validations and the fresh episodes.

## Running

From this folder, one task per line of a list of runs, five to a batch node:

    ../msCli/ms submit --array N --per-node 5 -t 4h -- bash run.sh configs/FILE.txt {i}

Each line of the list is `label task options`. `summarise.py` prints the table for a
fetched sweep, `collect.sh` gathers sweeps on the cluster and uploads them,
`evaluate_adapter.py` replays a kept adapter, and `evaluate_weights.py` plays an adapter
with hand-set weights (a check of the inputs, not a way of making adapters).
