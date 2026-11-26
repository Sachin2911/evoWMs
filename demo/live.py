"""Plays the frozen TD-MPC2 agent and the same agent with its evolved adapter side by side, live, in a browser.

Two arms play the same Meta-World episode from the same seed: one as the released agent
plays it, one with the adapter between the agent and the simulator. Each arm is a process
of its own, so that the planner's random numbers in one are not used up by the other, and
each plays exactly as frozen_tdmpc2.Player.play does, one decision at a time.

It is a demonstration and measures nothing: the numbers in the paper come from
scoreboard.py. It runs on the machine in front of the viewer, inside the TD-MPC2 image
(run.sh starts it), and is opened at the address it prints.

What the viewer can do:

    choose the task        pressing a button from above, or sliding a window open
    change the region      to another size or place, from those listed in regions.json.
                           The adapter's weights stay as they are; the cost predictor
                           it reads is the one fitted for that region
    turn the strength      what the adapter's gains and shifts are multiplied by before
                           they are applied to the action. 0 is the unadapted agent, and
                           1 is the adapter as the search left it
    push the arm           for the next PUSHED decisions the action both arms take is
                           replaced by a full move in one direction, as if someone had
                           shoved the arm. The agent and the adapter are not told
    pause, step, replay    and choose the seed

What is drawn that the simulator does not have: the region the gripper must not enter, as
a red ball, and the path the gripper has taken, red where it was charged. Both are markers
in the picture alone. The cost is ours (frozen_tdmpc2.py) and the agent never sees it.

The adapter of each task is the one the first search of that task chose, among the
searches with the hold and the small first step (implementation/adapterArchitecture.md,
decision 13). They are in adapters/, one file per task, as adapter.save wrote them.

The totals under the pictures count only episodes that were played whole at one strength
and with no push.
"""

import argparse
import base64
import io
import json
import multiprocessing
import pathlib
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'adapterRecipe'))

TASKS = ('mw-button-press-topdown', 'mw-window-open')
# The camera TD-MPC2 itself draws a Meta-World task from.
CAMERA = 'corner2'
# How many decisions a push lasts.
PUSHED = 5
# The fresh episodes of the scoreboards begin at this seed.
FIRST_SEED = 5000
# How long a finished episode stays on the screen before the next begins by itself, in seconds.
SHOWN_AFTER = 2.5


class Arm:
    """One frozen agent with its simulator, played a decision at a time, with the adapter or without."""

    def __init__(self, checkpoints, adapter_folder, adapted, size):
        self.checkpoints, self.adapter_folder = pathlib.Path(checkpoints), pathlib.Path(adapter_folder)
        self.adapted, self.size = adapted, size
        self.loaded = {}

    def choose(self, task):
        """Makes `task` the one this arm plays, loading its agent and simulator the first time, with the region it was searched with."""
        if task not in self.loaded:
            import frozen_tdmpc2

            player = frozen_tdmpc2.Player(self.checkpoints / f'{task}-1.pt')
            sim = player._env.unwrapped.sim
            # MuJoCo draws into a buffer whose size the task's model sets, 640 by 480 unless it is made larger here.
            shown = player._env.unwrapped.model.vis.global_
            shown.offwidth, shown.offheight = max(shown.offwidth, self.size), max(shown.offheight, self.size)
            # The first picture makes the renderer, which the markers are then added to.
            sim.render(self.size, self.size, camera_name=CAMERA, mode='offscreen')
            # The image draws on the processor, where the table's reflections take longer than all the rest of a picture.
            from mujoco_py.generated import const
            sim._render_context_offscreen.scn.flags[const.RND_REFLECTION] = 0
            self.loaded[task] = {'player': player, 'sim': sim, 'regions': regions_of(self.adapter_folder, task, player.radius),
                                 'adapters': {}}
        self.task = task
        self.now = self.loaded[task]
        self.place(self.now['regions'][0]['name'])

    def place(self, name):
        """Makes the region called `name` the one this arm plays with, loading its adapter the first time.

        Every region has an adapter file of its own: the same weights, with the cost
        predictor of that region and the sizes measured with it (adapterRecipe/check_region.py).
        """
        import adapter as adapters

        region = next(region for region in self.now['regions'] if region['name'] == name)
        if name not in self.now['adapters']:
            file = self.adapter_folder / region['file']
            state_numbers = int(np.load(file)['predictor_weights'].shape[0])
            adapter, weights, fixed = adapters.load(file, self.now['player'].action_numbers, state_numbers)
            self.now['adapters'][name] = {'adapter': adapter, 'weights': np.asarray(weights, np.float32), 'fixed': fixed,
                                          'at strength': {}}
        self.region = region
        self.now.update(self.now['adapters'][name])

    def reset(self, seed):
        """Begins the episode of `seed`, as Player.play begins one."""
        from collect import GOAL, GRIPPER, OBJECT

        player = self.now['player']
        # Reset twice, so that the half of the observation that is the scene one step
        # earlier is the same scene standing still (see tdmpc2Probe/collect.py).
        for _ in range(2):
            player._set_seed(seed)
            self.obs = player._env.reset()
        player.frozen.reseed(seed)
        self.memory = self.now['adapter'].start()
        self.gripper = GRIPPER
        observation = self.obs.numpy()
        start, target = observation[GRIPPER], observation[GOAL if player.toward == 'goal' else OBJECT]
        # The region the gripper must not enter. The adapters were searched with it halfway between where the gripper
        # starts and the first thing it has to reach; that one is worked out as frozen_tdmpc2.py works it out.
        along, up = self.region['along'], self.region['up']
        self.centre = (start + target) / 2 if (along, up) == (0.5, 0.0) else (
            start + along * (target - start) + np.array([0.0, 0.0, up])).astype(np.float32)
        self.decision, self.done, self.reward, self.success, self.output = 0, False, 0.0, False, None
        self.path, self.charged, self.chance = [], [], []
        self._arrived(observation)
        return self.described()

    def _arrived(self, observation):
        """Notes where the gripper is, whether that is charged, and what the cost predictor makes of the agent's state."""
        from look_ahead import expected

        player, torch = self.now['player'], self.now['player']._torch
        gripper = observation[self.gripper]
        self.path.append(gripper.copy())
        self.charged.append(bool(np.linalg.norm(gripper - self.centre) < self.region['radius']))
        with torch.no_grad():
            self.latent = player.agent.model.encode(self.obs.to(player.agent.device).unsqueeze(0), None).cpu().numpy()
        predictor = self.now['fixed']['predictor']
        self.chance.append(float(expected(np, predictor, self.latent)[0] / predictor['mean_cost']))

    def step(self, strength, push):
        """Plays one decision: the lines of Player.play's loop, with the strength and the push added."""
        player, torch = self.now['player'], self.now['player']._torch
        if self.done:
            return self.described()
        # The agent's own way of acting: its encoder, then its planner, without exploration noise.
        proposed = player.agent.act(self.obs, t0=self.decision == 0, eval_mode=True).numpy()[None]
        action, self.output = proposed, None
        if self.adapted and strength > 0:
            acts = self.now['at strength']
            if strength not in acts:
                acts[strength] = self.now['adapter'].compiled(player.frozen, everything=True, strength=strength)
            worked_out = acts[strength]({'z': self.latent}, proposed, self.memory, self.now['fixed'], self.now['weights'])
            action, self.memory, self.output = worked_out['action'], worked_out['memory'], worked_out['output'][0]
        action = np.array(action, np.float32)
        if push is not None:
            action[0, :3] = self._direction(push)
        self.obs, reward, self.done, info = player._env.step(torch.from_numpy(action[0]))
        self.decision += 1
        self.reward += float(reward)
        self.success = self.success or bool(info['success'])
        self._arrived(self.obs.numpy())
        return self.described()

    def _direction(self, push):
        """The three position numbers of an action that moves the gripper at full speed the way of `push`."""
        toward = self.centre - self.path[-1]
        toward = toward / max(float(np.linalg.norm(toward)), 1e-6)
        if push == 'toward':
            return toward
        if push == 'away':
            return -toward
        if push == 'up':
            return np.array([0.0, 0.0, 1.0], np.float32)
        # Sideways: level, and square to the line from the gripper to the region.
        sideways = np.array([-toward[1], toward[0], 0.0], np.float32)
        return sideways / max(float(np.linalg.norm(sideways)), 1e-6)

    def picture(self):
        """The scene as a JPEG, with the region and the gripper's path drawn in."""
        from mujoco_py.generated import const
        from PIL import Image

        sim, radius = self.now['sim'], self.region['radius']
        renderer = sim._render_context_offscreen
        inside = self.charged[-1]
        renderer.add_marker(pos=self.centre.astype(np.float64), size=np.full(3, radius), type=const.GEOM_SPHERE, label='',
                            rgba=np.array([1.0, 0.1, 0.1, 0.55 if inside else 0.3]))
        # The path is drawn as small boxes: a hundred balls take as long to draw as the scene.
        for point, charged in zip(self.path, self.charged):
            renderer.add_marker(pos=point.astype(np.float64), size=np.full(3, 0.005), type=const.GEOM_BOX, label='',
                                rgba=np.array([1.0, 0.15, 0.1, 1.0] if charged else [0.1, 0.75, 1.0, 1.0]))
        image = sim.render(self.size, self.size, camera_name=CAMERA, mode='offscreen')
        # The renderer keeps its markers from one picture to the next unless they are taken out.
        del renderer._markers[:]
        packed = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(image)).save(packed, 'JPEG', quality=88)
        return packed.getvalue()

    def described(self):
        return {'decision': self.decision, 'done': bool(self.done), 'reward': self.reward, 'success': self.success,
                'cost': int(sum(self.charged)), 'inside': self.charged[-1], 'charged': list(self.charged),
                'chance': [round(chance, 4) for chance in self.chance],
                'output': None if self.output is None else [round(float(number), 4) for number in self.output],
                'picture': self.picture()}


def regions_of(adapter_folder, task, radius):
    """The regions `task` can be shown with. The first is the one its adapter was searched with.

    The others are listed in regions.json in the adapter folder, if there is one: for each
    task, a list of regions with a `name`, the adapter `file` of that region, `along`,
    `up` and `radius` as check_region.py takes them, and what check_region.py `measured`
    there. An entry called 'as searched' only adds what was measured to the first region.
    """
    regions = [{'name': 'as searched', 'file': f'{task}.npz', 'along': 0.5, 'up': 0.0, 'radius': radius}]
    listed = pathlib.Path(adapter_folder) / 'regions.json'
    if listed.exists():
        for region in json.loads(listed.read_text()).get(task, []):
            if region['name'] == 'as searched':
                regions[0]['measured'] = region.get('measured')
            else:
                regions.append(region)
    return regions


def arm(pipe, checkpoints, adapter_folder, adapted, size):
    """An arm in a process of its own. It answers every order it is sent with what described() gives, or with the error."""
    one = Arm(checkpoints, adapter_folder, adapted, size)
    while True:
        order = pipe.recv()
        try:
            if order[0] == 'task':
                one.choose(order[1])
                pipe.send({'regions': one.now['regions'], 'limit': one.now['player'].cost_limit})
            elif order[0] == 'region':
                one.place(order[1])
                pipe.send({})
            elif order[0] == 'reset':
                pipe.send(one.reset(order[1]))
            elif order[0] == 'step':
                pipe.send(one.step(order[1], order[2]))
        except Exception as problem:  # sent back, so that the page can show it
            import traceback
            traceback.print_exc()
            pipe.send({'error': repr(problem)})


class Show:
    """The two arms, what the viewer has asked for, and the latest state for the page."""

    def __init__(self, args):
        context = multiprocessing.get_context('spawn')
        self.pipes = []
        for adapted in (False, True):
            ours, theirs = context.Pipe()
            context.Process(target=arm, args=(theirs, args.checkpoints, args.adapters, adapted, args.size), daemon=True).start()
            self.pipes.append(ours)
        self.changed = threading.Condition()
        self.wanted = {'task': args.task, 'region': None, 'strength': 1.0, 'paused': False, 'auto': True, 'rate': args.rate}
        self.orders = []
        self.version, self.latest = 0, json.dumps({'loading': f'loading {args.task}'})

    def ask(self, *order):
        """Sends both arms the same order at once and waits for both answers."""
        for pipe in self.pipes:
            pipe.send(order)
        answers = [pipe.recv() for pipe in self.pipes]
        for answer in answers:
            if 'error' in answer:
                raise RuntimeError(answer['error'])
        return answers

    def publish(self, state):
        with self.changed:
            self.latest = json.dumps(state)
            self.version += 1
            self.changed.notify_all()

    def control(self, asked):
        """Takes what the page asks for. Anything it does not know is left out."""
        with self.changed:
            if asked.get('task') in TASKS:
                self.wanted['task'] = asked['task']
            if isinstance(asked.get('region'), str):
                self.wanted['region'] = asked['region']
            if 'strength' in asked:
                self.wanted['strength'] = round(min(max(float(asked['strength']), 0.0), 2.0), 2)
            if 'rate' in asked:
                self.wanted['rate'] = min(max(float(asked['rate']), 1.0), 40.0)
            for name in ('paused', 'auto'):
                if name in asked:
                    self.wanted[name] = bool(asked[name])
            if asked.get('order') in ('next', 'replay', 'step', 'clear'):
                self.orders.append((asked['order'],))
            if asked.get('push') in ('toward', 'away', 'sideways', 'up'):
                self.orders.append(('push', asked['push']))
            if 'seed' in asked:
                self.orders.append(('seed', int(asked['seed'])))
            self.changed.notify_all()

    def run(self):
        task, region, strength, seed, details = None, None, None, FIRST_SEED, {}
        arms, begin, whole, push, pushes_left, over_since, finished, shown = None, True, True, None, 0, None, [], None
        while True:
            with self.changed:
                wanted, orders, self.orders = dict(self.wanted), self.orders, []
            step_once = False
            if wanted['task'] != task:
                task = wanted['task']
                self.publish({'loading': f'loading {task}'})
                details = self.ask('task', task)[0]
                # A task begins with the region its adapter was searched with.
                region = details['regions'][0]
                with self.changed:
                    self.wanted['region'] = wanted['region'] = region['name']
                seed, begin, finished = FIRST_SEED, True, []
            if wanted['region'] != region['name']:
                chosen = [one for one in details['regions'] if one['name'] == wanted['region']]
                if chosen:
                    region = chosen[0]
                    self.ask('region', region['name'])
                    begin, finished = True, []
                else:
                    with self.changed:
                        self.wanted['region'] = wanted['region'] = region['name']
            if wanted['strength'] != strength:
                # An episode played at two strengths is not counted, and the totals are of one strength.
                strength, whole, finished = wanted['strength'], False, []
            for order in orders:
                if order[0] == 'next':
                    seed, begin = seed + 1, True
                elif order[0] == 'replay':
                    begin = True
                elif order[0] == 'seed':
                    seed, begin = order[1], True
                elif order[0] == 'step':
                    step_once = True
                elif order[0] == 'clear':
                    finished = []
                elif order[0] == 'push' and arms is not None and not arms[0]['done']:
                    push, pushes_left, whole = order[1], PUSHED, False
            over = arms is not None and all(one['done'] for one in arms)
            if over and not begin and wanted['auto'] and not wanted['paused'] and time.monotonic() - over_since > SHOWN_AFTER:
                seed, begin = seed + 1, True

            started, played = time.monotonic(), True
            if begin:
                arms, begin, whole, pushes_left, over_since = self.ask('reset', seed), False, True, 0, None
            elif not over and (step_once or not wanted['paused']):
                arms = self.ask('step', strength, push if pushes_left else None)
                pushes_left = max(pushes_left - 1, 0)
                if all(one['done'] for one in arms):
                    over_since = time.monotonic()
                    if whole:
                        finished.append([(one['reward'], one['cost'], one['success']) for one in arms])
            else:
                played = False

            # The page is sent every decision played, and anything else that changes what it shows.
            showing = (wanted, whole, pushes_left > 0, len(finished))
            if played or showing != shown:
                shown = showing
                self.publish({
                    'task': task, 'tasks': TASKS, 'seed': seed, 'strength': strength, 'paused': wanted['paused'],
                    'auto': wanted['auto'], 'rate': wanted['rate'], 'whole': whole, 'pushing': pushes_left > 0,
                    'limit': details['limit'], 'length': 100, 'region': region,
                    'regions': [one['name'] for one in details['regions']],
                    'arms': [{**{name: value for name, value in one.items() if name != 'picture'},
                              'picture': base64.b64encode(one['picture']).decode()} for one in arms],
                    'totals': self.totals(finished)})
            if played:
                # No faster than the rate asked for.
                time.sleep(max(0.0, 1.0 / wanted['rate'] - (time.monotonic() - started)))
            else:
                # Nothing to play: wait for the viewer, or for the finished episode to have been shown long enough.
                with self.changed:
                    if not self.orders and self.wanted == wanted:
                        self.changed.wait(0.05)

    @staticmethod
    def totals(finished):
        if not finished:
            return {'episodes': 0}
        plain, adapted = (np.array([episode[arm] for episode in finished], float) for arm in (0, 1))
        return {'episodes': len(finished),
                'reward': [float(plain[:, 0].mean()), float(adapted[:, 0].mean())],
                'cost': [float(plain[:, 1].mean()), float(adapted[:, 1].mean())],
                'succeeded': [float(plain[:, 2].mean()), float(adapted[:, 2].mean())],
                'reward kept': float(adapted[:, 0].mean() / plain[:, 0].mean()) if plain[:, 0].mean() else None}


class Page(BaseHTTPRequestHandler):
    show = None

    def log_message(self, *arguments):
        pass

    def do_GET(self):
        if self.path == '/':
            page = (HERE / 'page.html').read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(page)))
            self.end_headers()
            self.wfile.write(page)
        elif self.path == '/events':
            # One long answer that the browser reads as it comes: a line of state for every decision played.
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            seen = -1
            try:
                while True:
                    with self.show.changed:
                        if self.show.version == seen:
                            self.show.changed.wait(10)
                        fresh, seen, state = self.show.version != seen, self.show.version, self.show.latest
                    # A line that says nothing keeps the connection open while the show is paused.
                    self.wfile.write(f'data: {state}\n\n'.encode() if fresh else b': waiting\n\n')
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != '/control':
            self.send_error(404)
            return
        try:
            self.show.control(json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0)))))
            self.send_response(204)
        except (ValueError, TypeError):
            self.send_response(400)
        self.end_headers()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoints', required=True, help='the folder of the released agents, mw-TASK-1.pt')
    parser.add_argument('--adapters', default=str(HERE / 'adapters'), help='the folder of the adapters, TASK.npz')
    parser.add_argument('--task', default=TASKS[0], choices=TASKS, help='the task shown first')
    parser.add_argument('--size', type=int, default=900, help='the side of each picture, in pixels')
    parser.add_argument('--rate', type=float, default=10.0, help='decisions a second, at most')
    parser.add_argument('--host', default='127.0.0.1', help='the address to listen on; this machine alone by default')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    show = Show(args)
    Page.show = show
    server = ThreadingHTTPServer((args.host, args.port), Page)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f'open http://{args.host}:{args.port}', flush=True)
    show.run()


if __name__ == '__main__':
    main()
