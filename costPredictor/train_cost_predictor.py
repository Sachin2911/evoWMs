"""Tests whether cost can be predicted from the frozen world model's internal state.

  python3 train_cost_predictor.py --recordings FOLDER [FOLDER ...] --out FOLDER

Each FOLDER is what evaluate.py --record --record-latents wrote for one checkpoint; several
folders of the same checkpoint are joined. The script asks, for several look-ahead lengths H:
"from what is known at this decision, will there be a cost within the next H decisions?"

The same question is put to predictors that read different things:

  latent      the world model's internal state (what an adapter would read)
  deter       only its 512 continuous numbers
  stoch       only its 48 categorical choices
  privileged  the true positions and speeds from the simulator: the best one could hope for
  shuffled    the internal state with the answers shuffled in training: what chance looks like

Episodes are split into training, validation and test sets as whole episodes, because
neighbouring decisions of one episode are nearly identical. Results go to results.json.

With --labelled-episodes N only N of the episodes outside the test set are used, four fifths
of them for training and one fifth for validation. The test episodes stay the same, so runs
with different N show how few episodes a predictor needs.
"""

import argparse
import json
import pathlib

import numpy as np

HORIZONS = [1, 5, 15, 25]     # look-ahead lengths, in decisions
HIDDEN = 16                   # size of the small network's one hidden layer
EPOCHS = 30
BATCH = 2048
UPDATES = 29                  # weight updates per epoch at least: what one pass over 120 episodes gives


def load_episode(folder, seed, repeat):
    """One episode as per-decision arrays: what each predictor reads, and whether each decision cost."""
    steps = np.load(folder / 'steps' / f'seed_{seed:04d}.npz')
    latents = np.load(folder / 'latents' / f'seed_{seed:04d}.npz')
    decisions = len(steps['cost']) // repeat
    # A decision costs if any of its simulator steps did.
    cost = steps['cost'][:decisions * repeat].reshape(decisions, repeat).max(1) > 0
    # Row k of the latents is the state after observation k, before decision k.
    deter = latents['deter'][:decisions].astype(np.float32)
    stoch = np.eye(48, dtype=np.float32)[latents['stoch'][:decisions]].reshape(decisions, -1)
    return {'cost': cost, 'deter': deter, 'stoch': stoch, 'steps': steps, 'decisions': decisions}


def privileged_features(steps, layout, repeat, decisions):
    """What perfect knowledge of the simulator gives, at each decision.

    The robot's speed and, for every kind of object, how far the nearest two are, which way
    the nearest lies relative to where the robot is facing, and how fast the robot is closing on it.
    """
    # Observation k shows the world after simulator step k * repeat; observation 0 is the start.
    rows = np.arange(1, decisions) * repeat - 1
    agent = np.vstack([np.array(layout['agent'] + [0.0, 0.0])[None], steps['agent'][rows]])   # x, y, heading, vx, vy
    start = [xy for item in layout['objects'] if item['movable'] for xy in (item['x'], item['y'])]
    movable = np.vstack([np.array(start).reshape(1, -1), steps['movable'][rows]])
    goal_start = next(item for item in layout['objects'] if item['object'] == 'goal')
    goal = np.vstack([[goal_start['x'], goal_start['y']], steps['goal'][rows]])

    kinds, moved = {}, 0
    for item in layout['objects']:
        if item['object'] == 'goal':
            position = goal
        elif item['movable']:
            position = movable[:, 2 * moved:2 * moved + 2]
            moved += 1
        else:
            position = np.broadcast_to([item['x'], item['y']], (decisions, 2))
        kinds.setdefault(item['object'], []).append((position, item['size']))

    features = [np.hypot(agent[:, 3], agent[:, 4])]
    for kind in sorted(kinds):
        positions = np.stack([position for position, _ in kinds[kind]], 1)   # decisions x objects x 2
        size = kinds[kind][0][1]
        offset = positions - agent[:, None, :2]
        distance = np.linalg.norm(offset, axis=-1)
        order = np.argsort(distance, axis=1)
        nearest = np.take_along_axis(distance, order[:, :1], 1)[:, 0]
        second = np.take_along_axis(distance, order[:, 1:2], 1)[:, 0] if distance.shape[1] > 1 else nearest
        towards = offset[np.arange(decisions), order[:, 0]] / np.maximum(nearest, 1e-6)[:, None]
        bearing = np.arctan2(towards[:, 1], towards[:, 0]) - agent[:, 2]
        closing = (agent[:, 3:5] * towards).sum(1)
        features += [nearest - size, second - size, np.cos(bearing), np.sin(bearing), closing]
    return np.stack(features, 1).astype(np.float32)


def auroc(scores, labels):
    """Chance that a decision followed by a cost gets a higher score than one that is not. 0.5 is chance."""
    from scipy.stats import rankdata
    positives = labels.sum()
    if positives == 0 or positives == len(labels):
        return float('nan')
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * (len(labels) - positives)))


def average_precision(scores, labels):
    """Of the decisions the predictor ranks as most dangerous, the share that really are, averaged down the ranking."""
    if labels.sum() == 0:
        return float('nan')
    ranked = labels[np.argsort(-scores)]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].mean())


def auroc_interval(scores, labels, episode_of, rng, resamples=200):
    """A 95% range for the AUROC, from redrawing whole test episodes with replacement."""
    episodes = np.unique(episode_of)
    members = {episode: np.nonzero(episode_of == episode)[0] for episode in episodes}
    values = []
    for _ in range(resamples):
        chosen = np.concatenate([members[episode] for episode in rng.choice(episodes, len(episodes))])
        values.append(auroc(scores[chosen], labels[chosen]))
    low, high = np.nanpercentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def train(inputs, labels, parts, hidden, seed):
    """Trains one predictor and returns its scores for every decision, with the weights it ended on.

    hidden = 0 is a plain weighted sum of the inputs (logistic regression); otherwise there is
    one hidden layer of that many units. The weights kept are those of the epoch that did best
    on the validation episodes.
    """
    import jax
    import jax.numpy as jnp
    import optax

    mean = inputs[parts['train']].mean(0)
    # Inputs that barely vary are not blown up: their spread is treated as at least 0.1.
    spread = np.maximum(inputs[parts['train']].std(0), 0.1)
    data = jnp.asarray((inputs - mean) / spread)
    target = jnp.asarray(labels.astype(np.float32))

    key = jax.random.PRNGKey(seed)
    width = inputs.shape[1]
    if hidden:
        first, second = jax.random.split(key)
        params = {'w1': jax.random.normal(first, (width, hidden)) / np.sqrt(width), 'b1': jnp.zeros(hidden),
                  'w2': jax.random.normal(second, (hidden,)) / np.sqrt(hidden), 'b2': jnp.zeros(())}
    else:
        params = {'w': jnp.zeros(width), 'b': jnp.zeros(())}

    def score(params, x):
        if hidden:
            return jnp.tanh(x @ params['w1'] + params['b1']) @ params['w2'] + params['b2']
        return x @ params['w'] + params['b']

    def loss(params, x, y):
        return optax.sigmoid_binary_cross_entropy(score(params, x), y).mean()

    optimiser = optax.adamw(1e-3, weight_decay=1e-4)
    state = optimiser.init(params)

    # The data is handed to the compiled functions as an argument. Left as an outside variable,
    # the compiler would treat a gigabyte of numbers as a constant and take minutes over it.
    @jax.jit
    def update(params, state, data, target, rows):
        gradients = jax.grad(loss)(params, data[rows], target[rows])
        updates, state = optimiser.update(gradients, state, params)
        return optax.apply_updates(params, updates), state

    all_scores = jax.jit(score)
    order_rng = np.random.default_rng(seed)
    train_rows = parts['train']
    best = (-1.0, None, None, 0)
    # With few training episodes one pass is only a handful of updates, so the rows are gone
    # through more than once per epoch: every training-set size then gets as many updates.
    batches = max(len(train_rows) // BATCH, UPDATES)
    passes = -(-batches * BATCH // len(train_rows))
    for epoch in range(1, EPOCHS + 1):
        shuffled = np.concatenate([order_rng.permutation(train_rows) for _ in range(passes)])
        for start in range(0, batches * BATCH, BATCH):
            params, state = update(params, state, data, target, jnp.asarray(shuffled[start:start + BATCH]))
        scores = np.asarray(all_scores(params, data))
        validation = auroc(scores[parts['validation']], labels[parts['validation']])
        weights = {name: np.asarray(value) for name, value in params.items()}
        if np.isnan(validation):
            # No cost at all in the validation episodes: nothing to choose by, so keep the latest.
            best = (-1.0, scores, weights, epoch)
        elif validation > best[0]:
            best = (validation, scores, weights, epoch)
    _, scores, weights, epoch = best
    weights.update(mean=mean, spread=spread)
    return scores, weights, epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--recordings', required=True, type=pathlib.Path, nargs='+')
    parser.add_argument('--out', required=True, type=pathlib.Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--labelled-episodes', type=int,
                        help='use only this many of the episodes outside the test set')
    parser.add_argument('--predictors', nargs='+', help='only these predictors (default: all five)')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    run = json.loads((args.recordings[0] / 'run.json').read_text())
    repeat = run['simulator_steps_per_decision']
    episodes = []
    for folder in args.recordings:
        layouts = {layout['seed']: layout for layout in map(json.loads, (folder / 'layouts.jsonl').read_text().splitlines())}
        for line in (folder / 'episodes.jsonl').read_text().splitlines():
            seed = json.loads(line)['seed']
            episode = load_episode(folder, seed, repeat)
            episode['privileged'] = privileged_features(episode.pop('steps'), layouts[seed], repeat, episode['decisions'])
            episode['seed'] = seed
            episodes.append(episode)
    episodes.sort(key=lambda episode: episode['seed'])

    # Whole episodes go to training (60%), validation (15%) or test (25%).
    order = rng.permutation(len(episodes))
    cut_train, cut_validation = int(0.6 * len(episodes)), int(0.75 * len(episodes))
    part_of_episode = np.full(len(episodes), 'unused', dtype=object)
    part_of_episode[order[:cut_train]] = 'train'
    part_of_episode[order[cut_train:cut_validation]] = 'validation'
    part_of_episode[order[cut_validation:]] = 'test'
    if args.labelled_episodes:
        # The first N in the shuffled order, so a smaller N is always a subset of a larger one.
        labelled = order[:cut_validation][:args.labelled_episodes]
        for_validation = max(1, round(len(labelled) / 5))
        part_of_episode[order[:cut_validation]] = 'unused'
        part_of_episode[labelled[:-for_validation]] = 'train'
        part_of_episode[labelled[-for_validation:]] = 'validation'

    episode_of = np.concatenate([np.full(episode['decisions'], index) for index, episode in enumerate(episodes)])
    position = np.concatenate([np.arange(episode['decisions']) for episode in episodes])
    length = np.concatenate([np.full(episode['decisions'], episode['decisions']) for episode in episodes])
    cost = np.concatenate([episode['cost'] for episode in episodes])
    # "Not being charged right now": the decision before this one had no cost. Predicting a cost
    # from here means seeing it coming, not just noticing that it is already happening.
    calm = np.concatenate([np.r_[True, ~episode['cost'][:-1]] for episode in episodes])
    deter = np.concatenate([episode['deter'] for episode in episodes])
    stoch = np.concatenate([episode['stoch'] for episode in episodes])
    latent = np.concatenate([stoch, deter], 1)
    inputs = {'latent': latent, 'deter': deter, 'stoch': stoch,
              'privileged': np.concatenate([episode['privileged'] for episode in episodes]), 'shuffled': latent}
    if args.predictors:
        inputs = {name: inputs[name] for name in args.predictors}

    results = {
        'checkpoint': run['checkpoint'], 'task': run['task'], 'variant': run['variant'],
        'episodes': len(episodes), 'episode_seeds': [episodes[0]['seed'], episodes[-1]['seed']],
        'simulator_steps_per_decision': repeat,
        # Steps of the episodes the predictors learn from (training and validation): what
        # gathering them costs in interaction with the simulator.
        'simulator_steps_labelled': int(sum(
            episode['decisions'] for episode, part in zip(episodes, part_of_episode) if part in ('train', 'validation')) * repeat),
        'episodes_in': {part: int((part_of_episode == part).sum()) for part in ('train', 'validation', 'test')},
        'decisions_with_cost': float(cost.mean()),
        'horizons': {},
    }
    print(f"{run['task']}: {len(episodes)} episodes, {len(cost)} decisions, {100 * cost.mean():.1f}% of them with a cost")

    for horizon in HORIZONS:
        # Will there be a cost in this decision or the next horizon - 1? Decisions too close to
        # the end of the episode to know are left out.
        ahead = np.concatenate([
            np.array([episode['cost'][k:k + horizon].any() for k in range(episode['decisions'])]) for episode in episodes])
        known = position + horizon <= length
        parts = {part: np.nonzero(known & (part_of_episode[episode_of] == part))[0] for part in ('train', 'validation', 'test')}
        test = parts['test']
        entry = {'share_followed_by_cost': float(ahead[test].mean()),
                 'share_followed_by_cost_when_calm': float(ahead[test][calm[test]].mean()),
                 'test_decisions': int(len(test)), 'test_decisions_calm': int(calm[test].sum()), 'predictors': {}}
        for name, values in inputs.items():
            labels = ahead
            if name == 'shuffled':
                # Shuffled in validation too: choosing the best epoch against true answers would
                # itself be a little learning, and lift this above chance.
                labels = ahead.copy()
                for part in ('train', 'validation'):
                    labels[parts[part]] = rng.permutation(ahead[parts[part]])
            entry['predictors'][name] = {}
            for model, hidden in (('linear', 0), ('network', HIDDEN)):
                scores, weights, epoch = train(values, labels, parts, hidden, args.seed)
                quiet = test[calm[test]]
                entry['predictors'][name][model] = {
                    'auroc': auroc(scores[test], ahead[test]),
                    'auroc_range': auroc_interval(scores[test], ahead[test], episode_of[test], rng),
                    'average_precision': average_precision(scores[test], ahead[test]),
                    'auroc_when_calm': auroc(scores[quiet], ahead[quiet]),
                    'auroc_when_calm_range': auroc_interval(scores[quiet], ahead[quiet], episode_of[quiet], rng),
                    'average_precision_when_calm': average_precision(scores[quiet], ahead[quiet]),
                    'best_epoch': epoch,
                }
                if name == 'latent':
                    # Kept for later: the network's hidden layer is a candidate compressor for the adapter.
                    np.savez(args.out / f'latent_{model}_h{horizon}.npz', **weights)
            row = entry['predictors'][name]
            print(f"  H={horizon:2d} {name:10s} AUROC linear {row['linear']['auroc']:.3f} network {row['network']['auroc']:.3f}"
                  f" | when calm: linear {row['linear']['auroc_when_calm']:.3f} network {row['network']['auroc_when_calm']:.3f}", flush=True)
        results['horizons'][str(horizon)] = entry

    (args.out / 'results.json').write_text(json.dumps(results, indent=1) + '\n')
    print(f'wrote {args.out / "results.json"}')


if __name__ == '__main__':
    main()
