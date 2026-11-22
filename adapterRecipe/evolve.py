"""Evolves a gated adapter for a frozen agent, counting every simulator step it uses.

  python3 evolve.py --checkpoint X.ckpt --out FOLDER --reads hidden --gate ahead15 \\
      --ranking penalty --scoring imagined

One run does everything: it plays the unadapted agent for a few episodes, trains the small
cost network on them (adapterPilot/cost_features.py), evolves the adapter (adapter.py), and
at the end plays the adapters it keeps on the 50 evaluation seeds.

The search is CMA-ES (the `cma` package, diagonal covariance), starting from all weights
zero. Each generation it proposes candidate adapters, they are ranked, and the ranking
moves the search.

Ranking (--ranking). cost is the average episode cost, limit is SafeDreamer's 2:

  safe-first  a candidate within the limit beats any that is over it; within the limit the
              higher reward wins, over it the lower cost wins. (The pilot's ranking.)
  penalty     by reward minus --penalty-weight times the cost above the limit.
  eps         safe-first, but the limit starts at the unadapted agent's cost and comes down
              to 2 over the first --eps-fraction of the budget. The search is then never
              asked to cut the whole cost at once.
  feedback    for --scoring imagined. Safe-first inside the world model, against a limit
              that is a share of what the unadapted agent would cost from the same states.
              The share starts at 1 and is steered by the centre's real episodes: cut by a
              fifth after a round while they cost more than --feedback-target, raised again
              when they cost less. Reward is then only given up while the real cost asks for it.
              With --feedback-hold the limit is only cut once the search has met it.
  priced      for --scoring imagined. By the share of the unadapted agent's reward a candidate
              keeps, minus a price times the share of its cost it keeps, both inside the world
              model. The price starts at 1 and is steered like the limit above: raised by a
              quarter after a round while the centre's real episodes cost more than
              --feedback-target, lowered again when they cost less.
  floor       reward first: a candidate that keeps at least --reward-floor of the unadapted
              agent's reward beats any that does not. Among those that keep it, the one
              closer to the cost limit wins, then the higher reward. The search is never
              allowed to buy safety by giving up the job.

How candidates get their scores (--scoring):

  real       every candidate plays episodes in the simulator.
  prescreen  --imagined-candidates are proposed and scored inside the world model, which
             costs no simulator steps. The best --real-candidates of them then play in the
             simulator; their simulator ranking comes first, the rest follow in imagined order.
  imagined   the search runs inside the world model only. After every --imagined-generations
             generations the centre of the search plays --round-episodes episodes in the
             simulator. Those episodes are the only simulator steps the search uses; they
             show the world model where the adapter takes the agent.

Scoring inside the world model: starting from states the agent really visited, the world
model is rolled forward --imagined-decisions decisions with the candidate's actions. Its reward predictor
gives the reward. The cost comes from a second copy of the cost network, the cost model,
which is retrained on every real episode played so far (--retrain-every), so it also knows
the states the adapted agent reaches. The network the adapter reads through is trained once,
on the unadapted agent's episodes, and then left alone.

Every few generations (or rounds) the centre of the search is played on a fixed set of
validation seeds. Two adapters are kept: the validated centre that ranks best safe-first,
and the one that ranks best by reward minus cost above the limit.

The step count includes the unadapted agent's first episodes, every candidate's episodes
and the validation episodes. Seeds: first episodes 1000 and up, candidates 2000 and up,
validation 3000 and up; all apart from the evaluation seeds (0 to 49).
"""

import argparse
import collections
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'adapterPilot'))
sys.path.insert(0, str(HERE.parent / 'safeDreamerBaselines'))
sys.path.insert(0, str(HERE))       # this folder's adapter.py, not the pilot's

import adapter  # noqa: E402
import cost_features  # noqa: E402

COST_LIMIT = 2.0            # SafeDreamer's limit on the cost of an episode
IMAGINED_STARTS = 1024      # how many starting states each candidate is imagined from
BUFFER_EPISODES = 200       # the latest this many real episodes give starting states and train the cost model
FIRST_SEED, CANDIDATE_SEED, VALIDATION_SEED, FINAL_SEED = 1000, 2000, 3000, 4000
FEEDBACK_STEP = 0.8         # with --ranking feedback or priced: how much the limit (or the price) moves after a round


def make_rank_key(ranking, penalty_weight=1.0, reward_floor=0.8):
    """Returns rank_key(cost, reward, limit, unadapted_reward); sorting by it puts the best candidate first."""
    def rank_key(cost, reward, limit, unadapted_reward):
        over = max(0.0, cost - limit)
        if ranking == 'penalty':
            return (penalty_weight * over - reward,)
        if ranking == 'floor':
            short = max(0.0, reward_floor * unadapted_reward - reward)
            return (short, over, -reward)
        return (over, -reward)
    return rank_key


def make_imagination(built, reads, gate, imagined_decisions, bound, look_horizon, look_push, look_with_cost_model=False):
    """Returns imagine(starts, weights, seed, feature_net, cost_model) -> (reward, cost chance) per decision.

    It uses the agent's own actor, world model and reward predictor, with their loaded weights.
    """
    import jax
    import jax.numpy as jnp
    from SafeDreamer import ninjax as nj

    inner = built.agent.agent
    actor, world = inner.task_behavior.ac.actor, inner.wm
    device = built.agent.policy_devices[0]

    def roll_out(start, weights, feature_net, cost_model):
        def decide(state, _):
            action = actor(state).sample(seed=nj.rng())
            looked = None
            if reads in adapter.LOOKS:
                # One sample per variation here: there are already many starting states.
                looked = adapter.look_ahead(world, state, action, cost_model if look_with_cost_model else feature_net, look_horizon, 1, jnp, look_push, nj.scan)
            feats, opening = adapter.read(feature_net, state['deter'], reads, gate, jnp, looked)
            action = adapter.adapt(weights, feats, opening, action, jnp, bound)
            following = world.rssm.img_step(state, action)
            reward = world.heads['reward'](following).mean().astype(jnp.float32).mean()
            arrival = jax.nn.sigmoid(adapter.cost_logits(cost_model, following['deter'], jnp)[..., 0]).mean()
            return {name: following[name] for name in ('deter', 'stoch')}, (reward, arrival)

        # The first decision is taken outside the loop: its result has exactly the form the loop
        # must keep. The rest is one compiled loop, not a copy of the world model per decision.
        state, (reward, arrival) = decide(start, 0)
        if imagined_decisions > 1:
            _, (rewards, arrivals) = nj.scan(decide, state, jnp.arange(imagined_decisions - 1))
            reward, arrival = reward + rewards.sum(), arrival + arrivals.sum()
        return reward / imagined_decisions, arrival / imagined_decisions

    compiled = nj.jit(nj.pure(roll_out), device=device)
    varibs = built.agent.varibs

    def on_device(net):
        return jax.device_put({name: np.asarray(value, np.float32) for name, value in net.items()}, device)

    def imagine(starts, weights, seed, feature_net, cost_model):
        (reward, arrival), _ = compiled(
            varibs, jax.device_put(np.int64(seed), device), starts,
            jax.device_put(np.asarray(weights, np.float32), device), feature_net, cost_model)
        # The roll-out must only read networks that came with the checkpoint. A misspelt path
        # would quietly create a fresh, untrained one instead.
        created = compiled.keys - set(varibs)
        assert not created, f'imagination created new variables: {sorted(created)[:5]}'
        reward, arrival = jax.device_get((reward, arrival))
        return float(reward), float(arrival)

    return imagine, on_device, lambda starts: jax.device_put(starts, device)


def make_look(built, horizon, samples, push=1.0):
    """Returns look(latent, action, net, seed): adapter.look_ahead for the agent's current state, as a numpy array."""
    import jax
    import jax.numpy as jnp
    from SafeDreamer import ninjax as nj

    world = built.agent.agent.wm
    device = built.agent.policy_devices[0]

    def looking(state, action, net):
        return adapter.look_ahead(world, state, action, net, horizon, samples, jnp, push, nj.scan)

    compiled = nj.jit(nj.pure(looking), device=device)
    varibs = built.agent.varibs

    def look(latent, action, net, seed):
        state = {name: np.asarray(latent[name]) for name in ('deter', 'stoch')}
        looked, _ = compiled(varibs, jax.device_put(np.int64(seed), device), jax.device_put(state, device),
                             jax.device_put(np.asarray(action, np.float32), device), net)
        created = compiled.keys - set(varibs)
        assert not created, f'look-ahead created new variables: {sorted(created)[:5]}'
        return np.asarray(jax.device_get(looked))

    return look


def train_cost_network(episodes, seed):
    """cost_features.train, allowed to move its arrays to the GPU.

    SafeDreamer, once loaded, makes JAX refuse any move to the GPU that is not asked for by
    name; the training code was written to run on its own and moves arrays freely.
    """
    import jax
    with jax.transfer_guard('allow'):
        return cost_features.train(episodes, seed)


class Buffer:
    """The latest real episodes: starting states for imagined roll-outs, and what the cost model learns from."""

    def __init__(self, classes):
        self.classes = classes      # how many values each of the world model's categorical choices can take
        self.episodes = collections.deque(maxlen=BUFFER_EPISODES)

    def add(self, episode):
        self.episodes.append(episode)

    def sample(self, rng, count, latest=0, danger_share=0.0, danger_latest=0, window=15, charged_too=False):
        """Starting states from the buffer's episodes, or from only the latest few of them.

        With danger_share, that share of the states are moments at most `window` decisions
        before a real cost, taken from the latest danger_latest episodes. Once an adapter is
        mostly safe such moments are rare, and a roll-out from an ordinary moment says little
        about the costs that are left.
        """
        def moments(episodes, before_cost):
            keeps = []
            for episode in episodes:
                # Only moments when the agent was not being charged: from there, a good candidate
                # can still avoid the cost, which is what the roll-out should measure.
                # With charged_too those moments are kept as well, so that a candidate is also
                # scored on how quickly it gets out of a hazard it is already in.
                charged = np.r_[False, episode['cost'] > 0]
                keep = np.ones_like(charged) if charged_too and not before_cost else ~charged
                if before_cost:
                    total = np.cumsum(np.r_[0, charged])
                    at = np.arange(len(charged))
                    keep &= total[np.minimum(at + window + 1, len(charged))] - total[at + 1] > 0
                keeps.append(keep)
            return (np.concatenate([episode[name][keep] for episode, keep in zip(episodes, keeps)]) for name in ('deter', 'stoch'))

        episodes = list(self.episodes)[-latest:] if latest else list(self.episodes)
        deter, stoch = moments(episodes, False)
        rows = rng.choice(len(deter), count, replace=len(deter) < count)
        deter, stoch = deter[rows], stoch[rows]
        if danger_share:
            near_deter, near_stoch = moments(list(self.episodes)[-danger_latest:] if danger_latest else list(self.episodes), True)
            if not len(near_deter):
                near_deter, near_stoch = moments(list(self.episodes), True)
            if len(near_deter):
                swapped = int(round(count * danger_share))
                near = rng.choice(len(near_deter), swapped, replace=len(near_deter) < swapped)
                deter, stoch = np.concatenate([near_deter[near], deter[swapped:]]), np.concatenate([near_stoch[near], stoch[swapped:]])
        return {'deter': deter.astype(np.float16), 'stoch': np.eye(self.classes, dtype=np.float16)[stoch]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True, type=pathlib.Path)
    parser.add_argument('--reads', choices=adapter.READS, required=True, help='what the adapter reads')
    parser.add_argument('--gate', choices=adapter.GATES, required=True, help='what switches the adapter on')
    parser.add_argument('--ranking', choices=('safe-first', 'penalty', 'eps', 'floor', 'feedback', 'priced'), required=True)
    parser.add_argument('--scoring', choices=('real', 'prescreen', 'imagined'), required=True)
    parser.add_argument('--penalty-weight', type=float, default=1.0,
                        help='with --ranking penalty: reward given up per unit of cost above the limit')
    parser.add_argument('--reward-floor', type=float, default=0.8,
                        help="with --ranking floor: the share of the unadapted agent's reward a candidate must keep")
    parser.add_argument('--feedback-target', type=float, default=1.0,
                        help="with --ranking feedback: the limit is tightened while the centre's real episodes cost more than this")
    parser.add_argument('--eps-fraction', type=float, default=0.5,
                        help='with --ranking eps: the share of the budget over which the limit comes down to 2')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--budget', type=int, default=300_000, help='simulator steps, everything included')
    parser.add_argument('--initial-episodes', type=int, default=25,
                        help='episodes of the unadapted agent played first; the cost network learns from them')
    parser.add_argument('--real-candidates', type=int, default=12, help='candidates per generation that play in the simulator')
    parser.add_argument('--imagined-candidates', type=int, default=48,
                        help='candidates per generation scored inside the world model (prescreen and imagined)')
    parser.add_argument('--episodes-per-candidate', type=int, default=2)
    parser.add_argument('--imagined-decisions', type=int, default=15,
                        help="how far the world model is rolled forward; 15 is what the agent's own training used")
    parser.add_argument('--imagined-generations', type=int, default=10,
                        help='with --scoring imagined: generations inside the world model per round')
    parser.add_argument('--round-episodes', type=int, default=4,
                        help='with --scoring imagined: simulator episodes the centre plays after each round')
    parser.add_argument('--start-episodes', type=int, default=0,
                        help='imagined roll-outs start from states of only the latest this many real episodes; 0 uses the whole buffer')
    parser.add_argument('--weight-bound', type=float, default=0.0,
                        help='every weight is squashed to stay between -bound and bound; 0 leaves weights free')
    parser.add_argument('--output-bound', type=float, default=0.0,
                        help='gain and shift are squashed to stay between -bound and bound before the gate; 0 leaves them free')
    parser.add_argument('--look-horizon', type=int, default=15,
                        help='with --reads look: decisions each variation of the action is held for inside the world model')
    parser.add_argument('--look-push', type=float, default=1.0,
                        help='with --reads look: how far each variation moves one action dimension')
    parser.add_argument('--look-samples', type=int, default=4,
                        help='with --reads look: times each variation is tried at every real decision')
    parser.add_argument('--imagined-starts', type=int, default=IMAGINED_STARTS,
                        help='how many starting states each candidate is imagined from')
    parser.add_argument('--skip-imagined', action='store_true',
                        help='with --scoring real: do not also score candidates inside the world model (it is only logged there)')
    parser.add_argument('--retrain-every', type=int, default=1,
                        help='generations (or rounds) between retrainings of the cost model; 0 never retrains it')
    parser.add_argument('--validate-every', type=int, default=4, help='generations (or rounds) between validations')
    parser.add_argument('--validation-episodes', type=int, default=10)
    parser.add_argument('--evaluation-episodes', type=int, default=50, help='episodes each kept adapter plays at the end; not counted')
    parser.add_argument('--first-step-size', type=float, default=0.5, help='how far the first candidates are from zero')
    parser.add_argument('--danger-share', type=float, default=0.0,
                        help='this share of the imagined starting states are moments shortly before a real cost; 0 takes all of them at random')
    parser.add_argument('--starts-charged', action='store_true',
                        help='imagined roll-outs also start from moments when the agent was being charged')
    parser.add_argument('--danger-episodes', type=int, default=48, help='the latest this many real episodes give those moments')
    parser.add_argument('--feedback-step', type=float, default=FEEDBACK_STEP,
                        help='with --ranking feedback or priced: the limit is multiplied by this (the price divided by it) after a round that cost too much')
    parser.add_argument('--feedback-relax', type=float, default=0.0,
                        help='the limit is multiplied by this (the price divided by it) after a round that did not; 1 never relaxes, 0 undoes one step')
    parser.add_argument('--look-with-cost-model', action='store_true',
                        help='the look-ahead asks the cost model that is retrained on every real episode so far, not the cost network from the first episodes')
    parser.add_argument('--feedback-hold', action='store_true',
                        help='with --ranking feedback: only tighten the limit once the best imagined candidate has met it')
    parser.add_argument('--final-centres', type=int, default=0,
                        help='at the end, this many of the best validated centres play fresh episodes and the best of them is kept as "final"; 0 skips this')
    parser.add_argument('--final-episodes', type=int, default=20, help='fresh episodes each of those centres plays; counted in the budget')
    args, safedreamer_options = parser.parse_known_args()
    if args.ranking in ('feedback', 'priced') and args.scoring != 'imagined':
        parser.error(f'--ranking {args.ranking} needs --scoring imagined')
    if args.skip_imagined and args.scoring != 'real':
        parser.error('--skip-imagined needs --scoring real')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'means').mkdir(exist_ok=True)
    rank_key = make_rank_key(args.ranking, args.penalty_weight, args.reward_floor)
    started = time.perf_counter()

    import cma
    import evaluate
    built = evaluate.build(args.checkpoint, safedreamer_options)
    import embodied  # importable once evaluate.build has imported SafeDreamer
    repeat = built.steps_per_decision
    action_size = built.env.act_space['action'].shape[-1]
    used = {'steps': 0, 'episodes': 0}

    # Playing one episode in the simulator with a given adapter. The adapter's network is
    # only known after the first episodes, which are played with no adapter at all.
    waiting_seeds = collections.deque()
    evaluate.seed_episodes(built.simulator, built.agent, iter(waiting_seeds.popleft, None))
    now = {'weights': None, 'latents': [], 'finished': [], 'policy': None, 'look_rng': None}

    def on_decision(latent):
        now['latents'].append((latent['deter'][0].astype(np.float16), latent['stoch'][0].argmax(-1).astype(np.uint8)))

    def unadapted_policy(*inputs):
        actions, state = built.agent.policy(*inputs, mode='eval')
        on_decision(state[0][0])
        return actions, state

    now['policy'] = unadapted_policy
    driver = embodied.Driver(built.env)
    driver.on_episode(lambda episode, _worker: now['finished'].append(episode))

    def play(weights, seed, counted=True):
        waiting_seeds.append(seed)
        now.update(weights=None if weights is None else np.asarray(weights, np.float32), latents=[], finished=[],
                   look_rng=np.random.default_rng(seed + 7919))
        driver(lambda *inputs: now['policy'](*inputs), episodes=1, lag=built.penalty)
        episode, = now['finished']
        # The first entry belongs to the observation at reset, before any decision.
        cost = episode['cost'][1:].astype(np.float64)
        deter, stoch = map(np.stack, zip(*now['latents']))
        assert len(deter) == len(cost) + 1, (len(deter), len(cost))
        if counted:
            used['steps'] += len(cost) * repeat
            used['episodes'] += 1
        return {'seed': seed, 'reward': float(episode['reward'].astype(np.float64).sum()),
                'total_cost': float(cost.sum()), 'cost': cost, 'deter': deter, 'stoch': stoch}

    def summary(episodes):
        return {'reward': float(np.mean([episode['reward'] for episode in episodes])),
                'cost': float(np.mean([episode['total_cost'] for episode in episodes])),
                'episodes': [[episode['seed'], round(episode['reward'], 3), episode['total_cost']] for episode in episodes]}

    # The unadapted agent's first episodes, and the cost network trained on them.
    first = [play(None, FIRST_SEED + k) for k in range(args.initial_episodes)]
    unadapted_reward = float(np.mean([episode['reward'] for episode in first]))
    unadapted_cost = float(np.mean([episode['total_cost'] for episode in first]))
    feature_net, epoch = train_cost_network(first, args.seed)
    np.savez(args.out / 'cost_net.npz', **feature_net)
    cost_model = feature_net
    size = adapter.count_weights(feature_net, args.reads, action_size)
    imagine, on_device, starts_on_device = make_imagination(built, args.reads, args.gate, args.imagined_decisions, args.output_bound, args.look_horizon, args.look_push, args.look_with_cost_model)
    models = {'feature': on_device(feature_net), 'cost': on_device(cost_model), 'cost_when_costly': float(cost_model['cost_when_costly']),
              'cost_arrays': cost_model}
    look = None
    if args.reads in adapter.LOOKS:
        looking = make_look(built, args.look_horizon, args.look_samples, args.look_push)

        def look(latent, action):
            return looking(latent, action, models['cost' if args.look_with_cost_model else 'feature'], int(now['look_rng'].integers(2 ** 31)))

    now['policy'] = adapter.adapted_policy(built.agent, feature_net, args.reads, args.gate, lambda: now['weights'], on_decision,
                                           args.output_bound, look)
    decisions = len(first[0]['cost'])
    steps_per_episode = decisions * repeat
    settings = {**{name: str(value) if isinstance(value, pathlib.Path) else value for name, value in vars(args).items()},
                'task': built.task, 'weights': size, 'cost_limit': COST_LIMIT,
                'unadapted_reward': unadapted_reward, 'unadapted_cost': unadapted_cost,
                'steps_used_by_first_episodes': used['steps'], 'cost_net_epoch': epoch,
                'cost_when_costly': float(feature_net['cost_when_costly'])}
    (args.out / 'settings.json').write_text(json.dumps(settings, indent=1) + '\n')
    print(json.dumps(settings), flush=True)
    print(f"first episodes: reward {unadapted_reward:.2f}, cost {unadapted_cost:.2f}, {time.perf_counter() - started:.0f} s", flush=True)

    buffer = Buffer(built.config.rssm.classes)
    for episode in first:
        buffer.add(episode)

    def imagined_score(starts, weights, seed):
        """What the world model expects of a whole episode, in the simulator's units."""
        reward, arrival = imagine(starts, weights, seed, models['feature'], models['cost'])
        return {'reward': reward * decisions, 'cost': arrival * decisions * models['cost_when_costly']}

    def retrain_cost_model():
        # The first episodes are always kept, so the model never forgets what the unadapted agent does.
        episodes = first + [episode for episode in buffer.episodes if episode['seed'] >= CANDIDATE_SEED]
        order = np.random.default_rng(len(episodes)).permutation(len(episodes))
        cost_model, _ = train_cost_network([episodes[i] for i in order], args.seed)
        models.update(cost=on_device(cost_model), cost_when_costly=float(cost_model['cost_when_costly']), cost_arrays=cost_model)

    def limit_now():
        """The cost limit candidates are ranked against at this point of the run."""
        if args.ranking != 'eps':
            return COST_LIMIT
        first_steps = settings['steps_used_by_first_episodes']
        share = (used['steps'] - first_steps) / (args.eps_fraction * (args.budget - first_steps))
        return COST_LIMIT + max(0.0, unadapted_cost - COST_LIMIT) * max(0.0, 1 - share) ** 2

    proposed = args.real_candidates if args.scoring == 'real' else args.imagined_candidates
    search = cma.CMAEvolutionStrategy(np.zeros(size), args.first_step_size, {
        'popsize': proposed, 'seed': args.seed + 1, 'CMA_diagonal': True, 'verbose': -9})
    rng = np.random.default_rng(args.seed)
    validations = []
    # With --ranking feedback: the share of the unadapted agent's imagined cost that candidates
    # are allowed, and what the centre's real episodes cost after each round.
    # With --ranking priced: the price of cost, steered the same way.
    feedback = {'share': 1.0, 'price': 1.0, 'costs': []}
    relax = args.feedback_relax or 1 / args.feedback_step

    def shape(weights):
        """The adapter's weights for the numbers the search works with: squashed to stay within the bound."""
        weights = np.asarray(weights, np.float64)
        return args.weight_bound * np.tanh(weights / args.weight_bound) if args.weight_bound else weights

    def validate(generation):
        weights = shape(search.mean)
        imagined = imagined_unadapted = None
        if not args.skip_imagined:
            starts, imagination_seed = starts_on_device(buffer.sample(rng, args.imagined_starts, args.start_episodes, args.danger_share, args.danger_episodes, args.imagined_decisions, args.starts_charged)), int(rng.integers(2 ** 31))
            imagined = imagined_score(starts, weights, imagination_seed)
            imagined_unadapted = imagined_score(starts, np.zeros(size), imagination_seed)
        episodes = [play(weights, VALIDATION_SEED + k) for k in range(args.validation_episodes)]
        result = summary(episodes)
        result.update(generation=generation, steps_used=used['steps'], step_size=float(search.sigma),
                      imagined=imagined, imagined_unadapted=imagined_unadapted, seconds=round(time.perf_counter() - started))
        np.save(args.out / 'means' / f'generation_{generation:04d}.npy', weights.astype(np.float32))
        validations.append(result)
        with (args.out / 'validations.jsonl').open('a') as file:
            file.write(json.dumps(result) + '\n')
        for episode in episodes:
            buffer.add(episode)
        inside = f" (imagined {imagined['reward']:.1f} / {imagined['cost']:.1f})" if imagined else ''
        print(f"validation, generation {generation}: reward {result['reward']:.2f}, cost {result['cost']:.2f}{inside}, "
              f"{used['steps']} steps used", flush=True)

    def imagined_generation(candidates):
        """Scores the candidates inside the world model. Returns their scores, the order (best first) and what it was judged against."""
        if args.skip_imagined:
            return [None] * len(candidates), list(range(len(candidates))), {}
        starts, imagination_seed = starts_on_device(buffer.sample(rng, args.imagined_starts, args.start_episodes, args.danger_share, args.danger_episodes, args.imagined_decisions, args.starts_charged)), int(rng.integers(2 ** 31))
        # Every candidate is imagined from the same states with the same random numbers, so
        # differences between their scores come from the candidates and not from luck.
        unadapted = imagined_score(starts, np.zeros(size), imagination_seed)
        imagined = [imagined_score(starts, weights, imagination_seed) for weights in candidates]
        # The world model's cost is not in the simulator's units for certain. So its limit is
        # set relative to the unadapted agent: the same share of its cost as in the simulator.
        limit = unadapted['cost'] * limit_now() / max(unadapted_cost, COST_LIMIT)
        if args.ranking == 'feedback':
            limit = unadapted['cost'] * feedback['share']
        order = sorted(range(len(candidates)), key=lambda i: rank_key(imagined[i]['cost'], imagined[i]['reward'], limit, unadapted['reward']))
        if args.ranking == 'priced':
            # The share of the unadapted agent's reward a candidate keeps, minus the price times
            # the share of its cost it keeps. At a price of 1 the unadapted agent and standing
            # still both score 0, and so does any adapter that only slows the agent down.
            reward_scale, cost_scale = max(abs(unadapted['reward']), 1.0), max(unadapted['cost'], 1e-6)
            order = sorted(range(len(candidates)), key=lambda i: feedback['price'] * imagined[i]['cost'] / cost_scale - imagined[i]['reward'] / reward_scale)
        return imagined, order, {'imagined_unadapted': unadapted, 'imagined_limit': limit}

    def tell(asked, order):
        # CMA-ES only uses the order of the values it is told, so each candidate's value is its place.
        places = np.empty(len(asked))
        places[order] = np.arange(len(asked))
        search.tell(asked, places.tolist())
        if args.weight_bound:
            # Beyond a few bounds the squashing makes every value look the same, and the search
            # would drift there for good. It is kept where a change still changes the adapter.
            search.mean[:] = np.clip(search.mean, -2.5 * args.weight_bound, 2.5 * args.weight_bound)
            search.sigma = min(search.sigma, args.weight_bound)
        return places

    def log(record):
        with (args.out / 'generations.jsonl').open('a') as file:
            file.write(json.dumps(record) + '\n')

    generation, candidate_episodes = 0, 0
    validation_steps = args.validation_episodes * steps_per_episode + args.final_centres * args.final_episodes * steps_per_episode
    if args.scoring == 'imagined':
        between = args.round_episodes * steps_per_episode
    else:
        between = args.real_candidates * args.episodes_per_candidate * steps_per_episode
    # A generation (or round) is only started if it, one last validation and the final episodes still fit in the budget.
    while used['steps'] + between + validation_steps <= args.budget:
        limit = limit_now()
        if args.scoring == 'imagined':
            for _ in range(args.imagined_generations):
                asked = search.ask()
                imagined, order, judged = imagined_generation([shape(weights) for weights in asked])
                tell(asked, order)
            best = imagined[order[0]]
            centre = shape(search.mean)
            episodes = [play(centre, CANDIDATE_SEED + candidate_episodes + k) for k in range(args.round_episodes)]
            candidate_episodes += args.round_episodes
            for episode in episodes:
                buffer.add(episode)
            real = summary(episodes)
            log({'generation': generation, 'steps_used': used['steps'], 'step_size': float(search.sigma), 'limit': limit,
                 **judged, 'imagined_best': best, 'centre_real': real, 'share': feedback['share'], 'price': feedback['price']})
            # The centre's real episodes steer the search inside the world model.
            feedback['costs'].append(real['cost'])
            over = np.mean(feedback['costs'][-2:]) > args.feedback_target
            if args.ranking == 'feedback':
                # The limit is tightened by a fifth while the last two rounds cost more than the
                # target, and relaxed again otherwise. With --feedback-hold it is only tightened
                # once the search has met it, so it cannot run ahead of what the search can reach.
                if not over:
                    feedback['share'] = min(1.0, feedback['share'] * relax)
                elif not args.feedback_hold or best['cost'] <= judged['imagined_limit']:
                    feedback['share'] *= args.feedback_step
            if args.ranking == 'priced':
                feedback['price'] = feedback['price'] / args.feedback_step if over else feedback['price'] / relax
            print(f"round {generation}: centre reward {real['reward']:.2f}, cost {real['cost']:.2f} | imagined best "
                  f"{best['reward']:.1f} / {best['cost']:.1f} (unadapted {judged['imagined_unadapted']['reward']:.1f} / "
                  f"{judged['imagined_unadapted']['cost']:.1f}) | {used['steps']} steps used, {time.perf_counter() - started:.0f} s", flush=True)
        else:
            asked = search.ask()
            candidates = [shape(weights) for weights in asked]
            imagined, imagined_order, judged = imagined_generation(candidates)
            chosen = imagined_order[:args.real_candidates] if args.scoring == 'prescreen' else list(range(proposed))
            seeds = [CANDIDATE_SEED + candidate_episodes + k for k in range(args.episodes_per_candidate)]
            candidate_episodes += args.episodes_per_candidate
            # All candidates of a generation play the same worlds.
            real = {i: [play(candidates[i], seed) for seed in seeds] for i in chosen}
            scores = {i: summary(episodes) for i, episodes in real.items()}
            order = sorted(chosen, key=lambda i: rank_key(scores[i]['cost'], scores[i]['reward'], limit, unadapted_reward))
            order += [i for i in imagined_order if i not in real]
            places = tell(asked, order)
            for episodes in real.values():
                for episode in episodes:
                    buffer.add(episode)
            log({'generation': generation, 'steps_used': used['steps'], 'step_size': float(search.sigma), 'limit': limit, **judged,
                 'candidates': [{'place': int(places[i]), 'imagined': imagined[i], 'real': scores.get(i)} for i in range(proposed)]})
            best = scores[order[0]]
            print(f"generation {generation}: best reward {best['reward']:.2f}, cost {best['cost']:.2f} | "
                  f"median reward {np.median([score['reward'] for score in scores.values()]):.1f}, "
                  f"cost {np.median([score['cost'] for score in scores.values()]):.1f} | limit {limit:.1f} | "
                  f"{used['steps']} steps used, {time.perf_counter() - started:.0f} s", flush=True)
        generation += 1
        if args.retrain_every and not args.skip_imagined and generation % args.retrain_every == 0:
            retrain_cost_model()
        if generation % args.validate_every == 0:
            validate(generation)
    if not validations or validations[-1]['generation'] != generation:
        validate(generation)
    search_seconds = round(time.perf_counter() - started)

    # The adapters kept: the validated centre that ranks best, by each of two rules.
    kept = {}
    penalty_rule = make_rank_key('penalty')

    def keep(name, best, **notes):
        weights = np.load(args.out / 'means' / f"generation_{best['generation']:04d}.npy")
        # With --look-with-cost-model the kept adapter looks ahead with the cost model as it is at
        # the end of the run; the final episodes and the evaluation below are played with that one too.
        look_net = {f'looknet_{key}': value for key, value in models['cost_arrays'].items()} if args.look_with_cost_model else {}
        adapter.save(args.out / f'adapter_{name}.npz', weights, args.reads, args.gate, feature_net, args.output_bound,
                     generation=best['generation'], steps_used=best['steps_used'],
                     look_horizon=args.look_horizon, look_samples=args.look_samples, look_push=args.look_push, **look_net)
        kept[name] = {'generation': best['generation'], 'steps_used': best['steps_used'],
                      'validation_reward': best['reward'], 'validation_cost': best['cost'], **notes}
        print(f"kept by {name}: generation {best['generation']}, {best['steps_used']} steps, "
              f"validation reward {best['reward']:.2f}, cost {best['cost']:.2f}", flush=True)

    safe_rule = make_rank_key('safe-first')

    def best_first(results, rule):
        return sorted(results, key=lambda result: rule(result['cost'], result['reward'], COST_LIMIT, unadapted_reward))

    if args.final_centres:
        # Ten validation episodes are few: one costly episode decides the average. So the best
        # few validated centres each play the same fresh episodes, and the best there is kept.
        # The short list takes turns between the two rules, so it holds both the safest centres
        # and those that give up least reward.
        short_list = []
        for pair in zip(best_first(validations, safe_rule), best_first(validations, penalty_rule)):
            for result in pair:
                if result not in short_list and len(short_list) < args.final_centres:
                    short_list.append(result)
        finals = []
        for result in short_list:
            weights = np.load(args.out / 'means' / f"generation_{result['generation']:04d}.npy")
            fresh = summary([play(weights, FINAL_SEED + k) for k in range(args.final_episodes)])
            finals.append({'generation': result['generation'], 'validation_reward': result['reward'], 'validation_cost': result['cost'], **fresh})
            print(f"final episodes, generation {result['generation']}: reward {fresh['reward']:.2f}, cost {fresh['cost']:.2f}, {used['steps']} steps used", flush=True)
        (args.out / 'finals.json').write_text(json.dumps(finals, indent=1) + '\n')
        # "final" is safe-first on the fresh episodes: within the limit comes first, then reward.
        for name, rule in (('final', safe_rule), ('final_penalty', penalty_rule)):
            chosen = best_first(finals, rule)[0]
            best = next(result for result in validations if result['generation'] == chosen['generation'])
            keep(name, {**best, 'steps_used': used['steps']}, final_reward=chosen['reward'], final_cost=chosen['cost'])
    for name, rule in (('penalty', penalty_rule), ('safe_first', safe_rule)):
        keep(name, best_first(validations, rule)[0])

    # The kept adapters on the evaluation seeds the baselines were measured on. Not counted as steps used.
    # The record of the run is written after each, so an adapter's evaluation is not lost if a later one is cut short.
    evaluated, done = {}, {}
    for name, note in kept.items():
        if note['generation'] not in evaluated:
            weights = np.load(args.out / 'means' / f"generation_{note['generation']:04d}.npy")
            episodes = []
            for seed in range(args.evaluation_episodes):
                episode = play(weights, seed, counted=False)
                episodes.append({'seed': seed, 'reward': episode['reward'], 'cost': episode['total_cost']})
            evaluated[note['generation']] = episodes
        episodes = evaluated[note['generation']]
        (args.out / f'evaluation_{name}.jsonl').write_text(''.join(json.dumps(episode) + '\n' for episode in episodes))
        reward = np.array([episode['reward'] for episode in episodes])
        cost = np.array([episode['cost'] for episode in episodes])
        note.update(evaluation_reward=float(reward.mean()), evaluation_cost=float(cost.mean()),
                    evaluation_within_limit=int((cost <= COST_LIMIT).sum()), evaluation_episodes=len(episodes))
        print(f"evaluation of {name}: reward {reward.mean():.2f} (sd {reward.std():.2f}), cost {cost.mean():.2f} "
              f"(sd {cost.std():.2f}), {int((cost <= COST_LIMIT).sum())} of {len(cost)} within the limit", flush=True)
        done[name] = note
        (args.out / 'finished.json').write_text(json.dumps({
            **used, 'generations': generation, 'search_seconds': search_seconds,
            'seconds': round(time.perf_counter() - started), 'kept': done}, indent=1) + '\n')


if __name__ == '__main__':
    main()
