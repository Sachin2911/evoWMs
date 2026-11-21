"""Trains the small cost network the adapter pilot uses, from a few recorded episodes.

  python3 cost_features.py --recordings FOLDER [FOLDER ...] --episodes 50 --out cost_net.npz

The network reads the world model's 512 continuous numbers and has one hidden layer of 16
units. It answers three questions at once about the moment an observation arrives:

  arrived   did the decision that led here cost?
  ahead5    will there be a cost within the next 5 decisions?
  ahead15   ... within the next 15?

It is used in two ways. Its 16 hidden units are what the `hidden` adapter reads. And its
`arrived` answer stands in for the simulator's cost signal when candidates are scored
inside the world model, where every imagined step arrives somewhere.

FOLDER is what evaluate.py --record --record-latents wrote for the unadapted agent. The
first --episodes of them (by seed) are used, four fifths for training and one fifth for
validation; the rest are only used to report how good the network is.
"""

import argparse
import json
import pathlib

import numpy as np

OUTPUTS = ('arrived', 'ahead5', 'ahead15')
AHEAD = {'ahead5': 5, 'ahead15': 15}
HIDDEN = 16
EPOCHS = 30
BATCH = 2048
UPDATES = 29      # weight updates per epoch at least, however few episodes there are


def load_recorded(folders):
    """Recorded episodes, ordered by seed. Each is a dict with, for D decisions:

    deter (D + 1, 512) and stoch (D + 1, 48): the world model's state at every observation
    cost (D,): the cost each decision incurred
    reward, total_cost: the episode's totals
    """
    episodes = []
    for folder in map(pathlib.Path, folders):
        repeat = json.loads((folder / 'run.json').read_text())['simulator_steps_per_decision']
        for line in (folder / 'episodes.jsonl').read_text().splitlines():
            result = json.loads(line)
            steps = np.load(folder / 'steps' / f"seed_{result['seed']:04d}.npz")
            latents = np.load(folder / 'latents' / f"seed_{result['seed']:04d}.npz")
            episodes.append({
                'seed': result['seed'], 'reward': result['reward'], 'total_cost': result['cost'],
                'deter': latents['deter'], 'stoch': latents['stoch'],
                'cost': steps['cost'].reshape(result['decisions'], repeat).sum(1)})
    return sorted(episodes, key=lambda episode: episode['seed'])


def labels(episode):
    """For every observation of an episode: the answers to the three questions, and which are known."""
    cost = episode['cost'] > 0
    rows = len(cost) + 1
    answer = np.zeros((rows, len(OUTPUTS)), np.float32)
    known = np.zeros((rows, len(OUTPUTS)), np.float32)
    answer[1:, 0] = cost
    known[:, 0] = 1
    for column, name in enumerate(OUTPUTS[1:], 1):
        horizon = AHEAD[name]
        # Observations too close to the end of the episode to know the answer are left out.
        for row in range(rows - horizon):
            answer[row, column] = cost[row:row + horizon].any()
            known[row, column] = 1
    return answer, known


def auroc(scores, answers):
    """Chance that a case with answer yes gets a higher score than one with answer no. 0.5 is chance."""
    from scipy.stats import rankdata
    answers = answers > 0.5
    yes = answers.sum()
    if yes == 0 or yes == len(answers):
        return float('nan')
    ranks = rankdata(scores)
    return float((ranks[answers].sum() - yes * (yes + 1) / 2) / (yes * (len(answers) - yes)))


def score_net(net, episodes):
    """The network's AUROC for each question on the given episodes."""
    from adapter import cost_logits
    logits = np.concatenate([cost_logits(net, episode['deter'], np) for episode in episodes])
    answer, known = (np.concatenate(part) for part in zip(*map(labels, episodes)))
    return {name: auroc(logits[known[:, column] > 0, column], answer[known[:, column] > 0, column])
            for column, name in enumerate(OUTPUTS)}


def train(episodes, seed=0):
    """Trains the network on the given episodes (the last fifth is held out to choose the epoch)."""
    import jax
    import jax.numpy as jnp
    import optax

    held_out = max(1, round(len(episodes) / 5))
    training, validation = episodes[:-held_out], episodes[-held_out:]
    deter = np.concatenate([episode['deter'] for episode in training]).astype(np.float32)
    answer, known = (np.concatenate(part) for part in zip(*map(labels, training)))
    mean = deter.mean(0)
    # Inputs that barely vary are not blown up: their spread is treated as at least 0.1.
    spread = np.maximum(deter.std(0), 0.1)
    data, answer, known = jnp.asarray((deter - mean) / spread), jnp.asarray(answer), jnp.asarray(known)

    first, second = jax.random.split(jax.random.PRNGKey(seed))
    width = deter.shape[1]
    params = {'w1': jax.random.normal(first, (width, HIDDEN)) / np.sqrt(width), 'b1': jnp.zeros(HIDDEN),
              'w2': jax.random.normal(second, (HIDDEN, len(OUTPUTS))) / np.sqrt(HIDDEN), 'b2': jnp.zeros(len(OUTPUTS))}

    def loss(params, x, y, known):
        logits = jnp.tanh(x @ params['w1'] + params['b1']) @ params['w2'] + params['b2']
        return (optax.sigmoid_binary_cross_entropy(logits, y) * known).sum() / known.sum()

    optimiser = optax.adamw(1e-3, weight_decay=1e-4)
    state = optimiser.init(params)

    @jax.jit
    def update(params, state, data, answer, known, rows):
        gradients = jax.grad(loss)(params, data[rows], answer[rows], known[rows])
        updates, state = optimiser.update(gradients, state, params)
        return optax.apply_updates(params, updates), state

    rng = np.random.default_rng(seed)
    batches = max(len(deter) // BATCH, UPDATES)
    passes = -(-batches * BATCH // len(deter))
    best = (-1.0, None, 0)
    for epoch in range(1, EPOCHS + 1):
        order = np.concatenate([rng.permutation(len(deter)) for _ in range(passes)])
        for start in range(0, batches * BATCH, BATCH):
            params, state = update(params, state, data, answer, known, jnp.asarray(order[start:start + BATCH]))
        net = {**{name: np.asarray(value) for name, value in params.items()}, 'mean': mean, 'spread': spread}
        scores = [value for value in score_net(net, validation).values() if not np.isnan(value)]
        # With no cost at all in the validation episodes there is nothing to choose by: keep the latest.
        quality = float(np.mean(scores)) if scores else -1.0
        if quality >= best[0]:
            best = (quality, net, epoch)
    _, net, epoch = best

    costly = np.concatenate([episode['cost'] for episode in training])
    # What a decision that costs anything costs on average: turns "chance of a cost" into "expected cost".
    net['cost_when_costly'] = np.float32(costly[costly > 0].mean() if (costly > 0).any() else 1.0)
    return net, epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--recordings', required=True, nargs='+')
    parser.add_argument('--episodes', type=int, required=True, help='how many recorded episodes to learn from')
    parser.add_argument('--out', required=True, type=pathlib.Path)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    recorded = load_recorded(args.recordings)
    used, unused = recorded[:args.episodes], recorded[args.episodes:]
    net, epoch = train(used, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **net)
    report = {
        'episodes_used': len(used), 'seeds_used': [used[0]['seed'], used[-1]['seed']],
        'best_epoch': epoch,
        'unadapted_reward': float(np.mean([episode['reward'] for episode in used])),
        'unadapted_cost': float(np.mean([episode['total_cost'] for episode in used])),
        'cost_when_costly': float(net['cost_when_costly']),
        'auroc_on_unused_episodes': score_net(net, unused) if unused else None,
        'unused_episodes': len(unused),
    }
    args.out.with_suffix('.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
