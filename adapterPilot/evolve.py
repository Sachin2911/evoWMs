"""Evolves an adapter for a frozen agent, counting every simulator step it uses.

  python3 evolve.py --checkpoint X.ckpt --recordings FOLDER [FOLDER ...] --cost-net cost_net.npz \\
      --initial-episodes 50 --reads hidden --scoring imagined --out FOLDER

The search is CMA-ES (the `cma` package, with a diagonal covariance). Each generation it
proposes a set of candidate adapters, they are ranked, and the ranking moves the search.

Ranking (--ranking):

  safe-first  a candidate whose average episode cost is within the limit beats any that is
              over it; among those within the limit the higher reward wins, and among those
              over it the lower cost wins.
  penalty     candidates are ranked by reward minus --penalty-weight times the cost above the
              limit. Unlike safe-first, a candidate that keeps most of the reward and cuts
              most of the cost beats one that is safe only because it stands still.

How candidates get their scores (--scoring):

  real      every candidate plays episodes in the simulator.
  imagined  many more candidates are proposed and all of them are scored inside the world
            model, which costs no simulator steps. Only the best few of those then play in
            the simulator. Their simulator ranking comes first; the rest follow in their
            imagined order.

Scoring inside the world model: starting from states the agent really visited, the world
model is rolled forward for a few decisions with the candidate's actions. Its reward
predictor gives the reward and the cost network (cost_features.py) the cost.

Every few generations the centre of the search is played on a fixed set of validation
seeds. The adapter that is finally kept is the validated centre that ranks best.

The step count includes the episodes the cost network learnt from, every candidate's
episodes and the validation episodes. Seeds: candidates 2000 and up, validation 3000 and
up; both apart from the evaluation seeds (0 to 49) and the recorded episodes (1000 to 1199).
"""

import argparse
import collections
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'safeDreamerBaselines'))

import adapter  # noqa: E402
import cost_features  # noqa: E402

COST_LIMIT = 2.0            # SafeDreamer's limit on the cost of an episode
IMAGINED_DECISIONS = 15     # how far the world model is rolled forward, as in the agent's own training
IMAGINED_STARTS = 1024      # how many starting states each candidate is imagined from
BUFFER_EPISODES = 100       # starting states are drawn from the latest this many real episodes
CANDIDATE_SEED, VALIDATION_SEED = 2000, 3000


def make_rank_key(ranking, penalty_weight):
    """Returns rank_key(cost, reward, limit); sorting by it puts the best candidate first."""
    def rank_key(cost, reward, limit):
        over = max(0.0, cost - limit)
        if ranking == 'penalty':
            return (penalty_weight * over - reward,)
        return (over, -reward)
    return rank_key


def make_imagination(built, net, kind):
    """Returns imagine(starts, weights, seed) -> (reward, cost) per decision, averaged over the roll-out.

    It uses the agent's own actor, world model and reward predictor, with their loaded weights.
    """
    import jax
    import jax.numpy as jnp
    from SafeDreamer import ninjax as nj

    inner = built.agent.agent
    actor, world = inner.task_behavior.ac.actor, inner.wm
    device = built.agent.policy_devices[0]

    def roll_out(start, weights, net):
        state, rewards, arrivals = start, [], []
        for _ in range(IMAGINED_DECISIONS):
            action = actor(state).sample(seed=nj.rng())
            action = adapter.adapt(weights, adapter.features(net, state['deter'], kind, jnp), action, jnp)
            state = world.rssm.img_step(state, action)
            rewards.append(world.heads['reward'](state).mean())
            arrivals.append(jax.nn.sigmoid(adapter.cost_logits(net, state['deter'], jnp)[..., 0]))
        return jnp.stack(rewards).mean(), jnp.stack(arrivals).mean()

    compiled = nj.jit(nj.pure(roll_out), device=device)
    on_device = jax.device_put({name: np.asarray(value, np.float32) for name, value in net.items()}, device)
    varibs = built.agent.varibs

    def imagine(starts, weights, seed):
        (reward, arrival), _ = compiled(
            varibs, jax.device_put(np.int64(seed), device), jax.device_put(starts, device),
            jax.device_put(np.asarray(weights, np.float32), device), on_device)
        # The roll-out must only read networks that came with the checkpoint. A misspelt path
        # would quietly create a fresh, untrained one instead.
        created = compiled.keys - set(varibs)
        assert not created, f'imagination created new variables: {sorted(created)[:5]}'
        reward, arrival = jax.device_get((reward, arrival))
        return float(reward), float(arrival)

    return imagine


class Buffer:
    """States the agent really visited, to start imagined roll-outs from."""

    def __init__(self, classes):
        self.classes = classes      # how many values each of the world model's categorical choices can take
        self.episodes = collections.deque(maxlen=BUFFER_EPISODES)

    def add(self, episode):
        # Only moments when the agent was not being charged: from there, a good candidate
        # can still avoid the cost, which is what the roll-out should measure.
        calm = np.r_[True, episode['cost'] == 0]
        self.episodes.append((episode['deter'][calm], episode['stoch'][calm]))

    def sample(self, rng, count):
        deter = np.concatenate([deter for deter, _ in self.episodes])
        stoch = np.concatenate([stoch for _, stoch in self.episodes])
        rows = rng.choice(len(deter), count, replace=len(deter) < count)
        # The choices are stored as numbers; the world model wants each as a row of zeros with a single one.
        return {'deter': deter[rows].astype(np.float16),
                'stoch': np.eye(self.classes, dtype=np.float16)[stoch[rows]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--recordings', required=True, nargs='+',
                        help='recorded episodes of the unadapted agent (evaluate.py --record --record-latents)')
    parser.add_argument('--cost-net', required=True, help='from cost_features.py, trained on the same episodes')
    parser.add_argument('--initial-episodes', type=int, required=True,
                        help='how many recorded episodes the cost network learnt from; they count as steps used')
    parser.add_argument('--reads', choices=adapter.KINDS, required=True, help='what the adapter reads')
    parser.add_argument('--scoring', choices=('real', 'imagined'), required=True)
    parser.add_argument('--ranking', choices=('safe-first', 'penalty'), default='safe-first')
    parser.add_argument('--penalty-weight', type=float, default=1.0,
                        help='with --ranking penalty: reward given up per unit of cost above the limit')
    parser.add_argument('--out', required=True, type=pathlib.Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--budget', type=int, default=1_000_000, help='simulator steps, everything included')
    parser.add_argument('--real-candidates', type=int, default=12, help='candidates per generation that play in the simulator')
    parser.add_argument('--imagined-candidates', type=int, default=48, help='candidates per generation with --scoring imagined')
    parser.add_argument('--episodes-per-candidate', type=int, default=2)
    parser.add_argument('--validate-every', type=int, default=4, help='generations between validations')
    parser.add_argument('--validation-episodes', type=int, default=10)
    parser.add_argument('--first-step-size', type=float, default=0.2, help='how far the first candidates are from zero')
    args, safedreamer_options = parser.parse_known_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'means').mkdir(exist_ok=True)
    rank_key = make_rank_key(args.ranking, args.penalty_weight)

    net = dict(np.load(args.cost_net))
    recorded = cost_features.load_recorded(args.recordings)[:args.initial_episodes]
    unadapted_reward = float(np.mean([episode['reward'] for episode in recorded]))
    unadapted_cost = float(np.mean([episode['total_cost'] for episode in recorded]))

    import cma
    import evaluate
    built = evaluate.build(args.checkpoint, safedreamer_options)
    import embodied  # importable once evaluate.build has imported SafeDreamer
    repeat = built.steps_per_decision
    used = {'steps': int(sum(len(episode['cost']) for episode in recorded) * repeat), 'episodes': len(recorded)}
    action_size = built.env.act_space['action'].shape[-1]
    size = adapter.count_weights(net, args.reads, action_size)
    settings = {**{name: str(value) if isinstance(value, pathlib.Path) else value for name, value in vars(args).items()},
                'task': built.task, 'weights': size, 'cost_limit': COST_LIMIT,
                'unadapted_reward': unadapted_reward, 'unadapted_cost': unadapted_cost,
                'steps_used_by_cost_net': used['steps']}
    (args.out / 'settings.json').write_text(json.dumps(settings, indent=1) + '\n')
    print(json.dumps(settings), flush=True)

    # Playing one episode in the simulator with a given adapter.
    waiting_seeds = collections.deque()
    evaluate.seed_episodes(built.simulator, built.agent, iter(waiting_seeds.popleft, None))
    now = {'weights': np.zeros(size), 'latents': [], 'finished': []}

    def on_decision(latent):
        now['latents'].append((latent['deter'][0].astype(np.float16), latent['stoch'][0].argmax(-1).astype(np.uint8)))

    policy = adapter.adapted_policy(built.agent, net, args.reads, lambda: now['weights'], on_decision)
    driver = embodied.Driver(built.env)
    driver.on_episode(lambda episode, _worker: now['finished'].append(episode))

    def play(weights, seed):
        waiting_seeds.append(seed)
        now.update(weights=np.asarray(weights), latents=[], finished=[])
        driver(policy, episodes=1, lag=built.penalty)
        episode, = now['finished']
        # The first entry belongs to the observation at reset, before any decision.
        cost = episode['cost'][1:].astype(np.float64)
        deter, stoch = map(np.stack, zip(*now['latents']))
        used['steps'] += len(cost) * repeat
        used['episodes'] += 1
        return {'seed': seed, 'reward': float(episode['reward'].astype(np.float64).sum()),
                'total_cost': float(cost.sum()), 'cost': cost, 'deter': deter, 'stoch': stoch}

    def summary(episodes):
        return {'reward': float(np.mean([episode['reward'] for episode in episodes])),
                'cost': float(np.mean([episode['total_cost'] for episode in episodes])),
                'episodes': [[episode['seed'], round(episode['reward'], 3), episode['total_cost']] for episode in episodes]}

    buffer = Buffer(built.config.rssm.classes)
    for episode in recorded:
        buffer.add(episode)
    imagine = make_imagination(built, net, args.reads)
    decisions = len(recorded[0]['cost'])

    def imagined_score(starts, weights, seed):
        """What the world model expects of a whole episode, in the simulator's units."""
        reward, arrival = imagine(starts, weights, seed)
        return {'reward': reward * decisions, 'cost': arrival * decisions * float(net['cost_when_costly'])}

    proposed = args.imagined_candidates if args.scoring == 'imagined' else args.real_candidates
    search = cma.CMAEvolutionStrategy(np.zeros(size), args.first_step_size, {
        'popsize': proposed, 'seed': args.seed + 1, 'CMA_diagonal': True, 'verbose': -9})
    rng = np.random.default_rng(args.seed)
    steps_per_episode = decisions * repeat
    generation_steps = args.real_candidates * args.episodes_per_candidate * steps_per_episode
    validation_steps = args.validation_episodes * steps_per_episode
    validations = []

    def validate(generation):
        weights = np.array(search.mean)
        result = summary([play(weights, VALIDATION_SEED + k) for k in range(args.validation_episodes)])
        result.update(generation=generation, steps_used=used['steps'], step_size=float(search.sigma))
        np.save(args.out / 'means' / f'generation_{generation:03d}.npy', weights.astype(np.float32))
        validations.append(result)
        with (args.out / 'validations.jsonl').open('a') as file:
            file.write(json.dumps(result) + '\n')
        print(f"validation, generation {generation}: reward {result['reward']:.2f}, cost {result['cost']:.2f}, "
              f"{used['steps']} steps used", flush=True)

    generation, started = 0, time.perf_counter()
    # A generation is only started if it and one last validation still fit in the budget.
    while used['steps'] + generation_steps + validation_steps <= args.budget:
        candidates = search.ask()
        starts, imagination_seed = buffer.sample(rng, IMAGINED_STARTS), int(rng.integers(2 ** 31))
        # Every candidate is imagined from the same states with the same random numbers, so
        # differences between their scores come from the candidates and not from luck.
        unadapted = imagined_score(starts, np.zeros(size), imagination_seed)
        imagined = [imagined_score(starts, weights, imagination_seed) for weights in candidates]
        # The world model's cost is not in the simulator's units for certain. So its limit is
        # set relative to the unadapted agent: the same share of its cost as in the simulator.
        imagined_limit = unadapted['cost'] * COST_LIMIT / unadapted_cost
        imagined_order = sorted(range(proposed), key=lambda i: rank_key(imagined[i]['cost'], imagined[i]['reward'], imagined_limit))
        chosen = imagined_order[:args.real_candidates] if args.scoring == 'imagined' else list(range(proposed))

        seeds = [CANDIDATE_SEED + generation * args.episodes_per_candidate + k for k in range(args.episodes_per_candidate)]
        # All candidates of a generation play the same worlds.
        real = {i: [play(candidates[i], seed) for seed in seeds] for i in chosen}
        scores = {i: summary(episodes) for i, episodes in real.items()}
        order = sorted(chosen, key=lambda i: rank_key(scores[i]['cost'], scores[i]['reward'], COST_LIMIT))
        order += [i for i in imagined_order if i not in real]
        # CMA-ES only uses the order of the values it is told, so each candidate's value is its place.
        places = np.empty(proposed)
        places[order] = np.arange(proposed)
        search.tell(candidates, places.tolist())
        for episodes in real.values():
            for episode in episodes:
                buffer.add(episode)

        record = {'generation': generation, 'steps_used': used['steps'], 'step_size': float(search.sigma),
                  'imagined_unadapted': unadapted, 'imagined_limit': imagined_limit,
                  'candidates': [{'place': int(places[i]), 'imagined': imagined[i], 'real': scores.get(i)} for i in range(proposed)]}
        with (args.out / 'generations.jsonl').open('a') as file:
            file.write(json.dumps(record) + '\n')
        best = scores[order[0]]
        print(f"generation {generation}: best reward {best['reward']:.2f}, cost {best['cost']:.2f} | "
              f"median cost {np.median([score['cost'] for score in scores.values()]):.1f} | "
              f"{used['steps']} steps used, {time.perf_counter() - started:.0f} s", flush=True)
        generation += 1
        if generation % args.validate_every == 0:
            validate(generation)
    if not validations or validations[-1]['generation'] != generation:
        validate(generation)

    # The adapter kept: the validated centre that ranks best, over the whole budget and over its first half.
    for name, limit in (('adapter_full_budget', args.budget), ('adapter_half_budget', args.budget // 2)):
        within = [result for result in validations if result['steps_used'] <= limit]
        if not within:
            continue
        kept = min(within, key=lambda result: rank_key(result['cost'], result['reward'], COST_LIMIT))
        weights = np.load(args.out / 'means' / f"generation_{kept['generation']:03d}.npy")
        adapter.save(args.out / f'{name}.npz', weights, args.reads, net,
                     generation=kept['generation'], steps_used=kept['steps_used'])
        print(f"{name}: generation {kept['generation']}, {kept['steps_used']} steps, "
              f"validation reward {kept['reward']:.2f}, cost {kept['cost']:.2f}", flush=True)
    (args.out / 'finished.json').write_text(json.dumps({**used, 'generations': generation, 'seconds': round(time.perf_counter() - started)}) + '\n')


if __name__ == '__main__':
    main()
