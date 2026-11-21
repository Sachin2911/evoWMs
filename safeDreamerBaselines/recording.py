"""Records what happens inside an episode. Used by evaluate.py when it is given --record.

For every episode it saves one row per simulator step (the action, the reward, the cost and
what caused it, where the robot and the goal are) and where every object started. For chosen
episodes it also saves a video: a view from above, next to the two camera images the agent sees.
It can also save the world model's internal state at every decision.

Recording only reads from the simulator, so a recorded episode plays out exactly like an
unrecorded one.
"""

import json
import time

import numpy as np

VIDEO_SIZE = 480    # width and height of the view from above, in pixels
VIDEO_EVERY = 2     # one video frame per this many simulator steps
VIDEO_FPS = 25      # 50 simulator steps are one second, so the video plays at real speed
PANEL = VIDEO_SIZE // 2   # the agent's two camera images are shown at this size, one above the other


def make_folder(path):
    """Creates a folder, and any missing folders above it.

    On the cluster's shared storage a folder that was only just created can, for a moment,
    be reported as "exists" and "is not a folder" at once. Waiting a second clears it.
    """
    for _ in range(10):
        try:
            path.mkdir(parents=True, exist_ok=True)
            return
        except FileExistsError:
            time.sleep(1)
    path.mkdir(parents=True, exist_ok=True)


class Recorder:

    def __init__(self, simulator, folder, steps_per_decision, video_seeds=()):
        self.simulator = simulator
        self.folder = folder
        self.steps_per_decision = steps_per_decision
        self.video_seeds = set(video_seeds)
        self.camera = None
        make_folder(folder / 'steps')
        make_folder(folder / 'videos')

    @property
    def task(self):
        # The part of the simulator that holds the robot, the objects and the physics.
        return self.simulator.unwrapped.task

    def start_episode(self, seed):
        """Call right after the simulator's reset."""
        self.seed = seed
        self.filming = seed in self.video_seeds
        self.rows = []
        self.frames = []
        self.states = []
        self.layout = {'seed': seed, 'agent': self._agent()[:3], 'objects': self._objects()}

    def after_step(self, action, result):
        """Call after every simulator step, with the action taken and what the step returned."""
        _, reward, cost, _, _, info = result
        movable = [xy for obj in self._objects() if obj['movable'] for xy in (obj['x'], obj['y'])]
        self.rows.append({
            'action': np.asarray(action, np.float32),
            'reward': reward,
            'cost': cost,
            # The simulator reports the cost of each kind of object separately.
            'costs': {name: value for name, value in info.items()
                      if name.startswith('cost_') and name != 'cost_sum'},
            'agent': self._agent(),
            'goal': [float(v) for v in self.task.goal.pos[:2]],
            'goal_met': bool(info.get('goal_met', False)),
            'movable': movable,
        })
        if self.filming and len(self.rows) % VIDEO_EVERY == 0:
            self.frames.append(self._view_from_above())

    def after_decision(self, state):
        """Call after each of the agent's decisions, with the state its policy returned.

        The state holds the world model's summary of everything seen so far, which is what the
        actor reads to pick an action. It has two parts: 512 numbers ("deter"), and 48 choices
        each among 48 options ("stoch"), which we store as the 48 chosen option numbers.
        """
        latent = state[0][0]
        self.states.append((np.asarray(latent['deter'][0], np.float16),
                            np.asarray(latent['stoch'][0]).argmax(-1).astype(np.uint8)))

    def finish_episode(self, episode):
        """Call at the end of the episode, with everything the agent saw. Writes the files."""
        rows = self.rows
        cost_names = sorted({name for row in rows for name in row['costs']})
        arrays = {
            'action': np.array([row['action'] for row in rows], np.float32),
            'reward': np.array([row['reward'] for row in rows], np.float32),
            'cost': np.array([row['cost'] for row in rows], np.float32),
            # Which of the agent's decisions each simulator step belongs to.
            'decision': np.arange(len(rows)) // self.steps_per_decision,
            'agent': np.array([row['agent'] for row in rows], np.float32),   # x, y, heading, vx, vy
            'goal': np.array([row['goal'] for row in rows], np.float32),     # x, y
            'goal_met': np.array([row['goal_met'] for row in rows], bool),
            'movable': np.array([row['movable'] for row in rows], np.float32),   # x, y of each movable object
        }
        for name in cost_names:
            arrays[name] = np.array([row['costs'].get(name, 0.0) for row in rows], np.float32)
        np.savez_compressed(self.folder / 'steps' / f'seed_{self.seed:04d}.npz', **arrays)

        if self.states:
            # One row per observation: the state after seeing it, before acting on it.
            make_folder(self.folder / 'latents')
            np.savez_compressed(
                self.folder / 'latents' / f'seed_{self.seed:04d}.npz',
                deter=np.array([deter for deter, _ in self.states]),
                stoch=np.array([stoch for _, stoch in self.states]))

        if self.filming:
            write_video(
                self.folder / 'videos' / f'seed_{self.seed:04d}.mp4', self.frames,
                episode['image'], episode['image2'], arrays['reward'], arrays['cost'],
                self.steps_per_decision)

        with (self.folder / 'layouts.jsonl').open('a') as file:
            file.write(json.dumps(self.layout) + '\n')

    def _agent(self):
        """The robot's x, y, heading (radians, 0 is along x), and its speed along x and y."""
        agent = self.task.agent
        rotation = np.asarray(agent.mat)
        heading = np.arctan2(rotation[1, 0], rotation[0, 0])
        return [float(v) for v in (*agent.pos[:2], heading, *agent.vel[:2])]

    def _objects(self):
        """Every object in the world, one entry each: its kind, where it is, how big, whether it can move."""
        task = self.task
        objects = []
        for group in task._obstacles:
            # A group is e.g. all the hazards. Its positions are one row each, or a single row.
            positions = np.asarray(group.pos, np.float64).reshape(-1, 3)
            movable = group.name in task._free_geoms or group.name in task._mocaps
            for index, position in enumerate(positions):
                objects.append({
                    'object': group.name, 'index': index, 'movable': movable,
                    'size': float(group.size), 'x': float(position[0]), 'y': float(position[1])})
        return objects

    def _view_from_above(self):
        task = self.task
        if self.camera is None:
            # A camera of our own, of the same kind as the simulator's. The simulator's is fixed
            # at the 64x64 size the agent sees. Ours is made once and kept for the whole run:
            # closing a camera part-way through disturbs the simulator's one.
            from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer
            task.model.vis.global_.offwidth = VIDEO_SIZE
            task.model.vis.global_.offheight = VIDEO_SIZE
            self.camera = OffScreenViewer(task.model, task.data)
            self.camera.vopt.geomgroup[:] = 1   # every kind of object visible
        camera = self.camera
        # Every episode builds a new world, so point the camera at the current one.
        camera.model, camera.data = task.model, task.data
        camera.cam.lookat[:] = 0
        camera.cam.elevation = -90    # straight down
        camera.cam.azimuth = 90       # x to the right, y up, like a map
        # High enough to see the area the objects are placed in, plus a margin.
        half_width = max(abs(edge) for edge in task.placements_conf.extents) + 0.75
        camera.cam.distance = half_width / np.tan(np.radians(task.model.vis.global_.fovy / 2))
        camera.make_context_current()
        image = camera.render('rgb_array', camera_id=-1).copy()
        self._give_graphics_back()
        return image

    def _give_graphics_back(self):
        # Only one camera can use the graphics card at a time. Ours has just taken its turn,
        # so hand it back to the simulator's camera, or the agent's next image would be wrong.
        viewer = self.task._viewers.get('rgb_array')
        if viewer is not None:
            viewer.make_context_current()


def write_video(path, frames, front, rear, rewards, costs, steps_per_decision):
    """Writes the video: the view from above on the left, the agent's two cameras on the right."""
    import cv2
    import imageio

    def label(image, text, position):
        # White text with a dark outline, so it shows on any background.
        for colour, thickness in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, thickness, cv2.LINE_AA)

    writer = imageio.get_writer(path, fps=VIDEO_FPS, codec='libx264', quality=8, macro_block_size=16)
    for number, above in enumerate(frames, start=1):
        step = number * VIDEO_EVERY
        # The newest image the agent had seen by this step. It gets a new one after each decision.
        seen = min(step // steps_per_decision, len(front) - 1)
        frame = np.zeros((VIDEO_SIZE, VIDEO_SIZE + PANEL, 3), np.uint8)
        frame[:, :VIDEO_SIZE] = above
        frame[:PANEL, VIDEO_SIZE:] = cv2.resize(front[seen], (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
        frame[PANEL:, VIDEO_SIZE:] = cv2.resize(rear[seen], (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
        if costs[step - VIDEO_EVERY:step].sum() > 0:
            # A red border while the robot is being charged a cost.
            cv2.rectangle(frame, (2, 2), (VIDEO_SIZE - 3, VIDEO_SIZE - 3), (230, 0, 0), 5)
        label(frame, f'step {step}   reward {rewards[:step].sum():.1f}   cost {costs[:step].sum():.0f}', (10, 22))
        label(frame, 'front camera', (VIDEO_SIZE + 8, 18))
        label(frame, 'rear camera', (VIDEO_SIZE + 8, PANEL + 18))
        writer.append_data(frame)
    writer.close()
