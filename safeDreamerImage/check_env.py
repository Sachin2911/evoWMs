"""Checks that the SafeDreamer image works on my laptop.

Four checks: the GPU is visible, JAX computes on it, the simulator draws its
camera images, and how fast the simulator runs. Run it inside the image:

  apptainer exec --nv safeDreamer.sif python3 check_env.py --out some_folder

It prints one PASS or FAIL line per check and exits with an error if any failed.
"""

import argparse
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

# Where SafeDreamer's code is. The image has it at /opt/SafeDreamer.
SAFEDREAMER_DIR = os.environ.get('SAFEDREAMER_DIR', '/opt/SafeDreamer')

TASK = 'SafetyPointGoal1-v0'
ACTION_REPEAT = 4      # simulator steps per agent decision, as in the paper
WARMUP_DECISIONS = 20  # move the robot a little before saving the camera images
SPEED_DECISIONS = 200


def check_gpu_visible():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
        capture_output=True, text=True, check=True)
    return result.stdout.strip()


def check_jax_on_gpu():
    import jax
    import jax.numpy as jnp

    device = jax.devices()[0]
    if device.platform != 'gpu':
        raise RuntimeError(f'JAX is using the {device.platform}, not the GPU')

    # A compiled calculation with a known answer: a 512x512 matrix of ones times itself sums to 512^3.
    total = float(jax.jit(lambda a: (a @ a).sum())(jnp.ones((512, 512))))
    if total != 512.0 ** 3:
        raise RuntimeError(f'wrong answer from the GPU: {total}')
    return f'{device}, compiled calculation correct'


def make_env():
    # Built through SafeDreamer's own wrapper, so this exercises the same code the agents will use.
    sys.path.insert(0, os.path.join(SAFEDREAMER_DIR, 'SafeDreamer'))
    from embodied.envs.safetygym import SafetyGym

    env = SafetyGym(
        TASK, platform='gpu', repeat=ACTION_REPEAT, obs_key='image', render=True,
        size=(64, 64), camera_name='vision_front_back')
    env.step({'action': np.zeros(env.act_space['action'].shape, np.float32), 'reset': True})
    return env


def random_step(env, rng):
    space = env.act_space['action']
    action = rng.uniform(space.low, space.high).astype(np.float32)
    return env.step({'action': action, 'reset': False})


def check_rendering(env, rng, out_dir):
    import imageio

    for _ in range(WARMUP_DECISIONS):
        obs = random_step(env, rng)

    cameras = {'front': obs['image'], 'rear': obs['image2']}
    for name, image in cameras.items():
        if image.shape != (64, 64, 3) or image.dtype != np.uint8:
            raise RuntimeError(f'{name} camera has shape {image.shape} and type {image.dtype}')
        # A failed render is usually one flat colour, which has no spread in its pixel values.
        if image.std() < 1.0:
            raise RuntimeError(f'{name} camera image is blank')
        imageio.imwrite(out_dir / f'{name}.png', image)

    # Both cameras side by side, enlarged 6x so they are easy to look at.
    side_by_side = np.concatenate([cameras['front'], cameras['rear']], axis=1)
    enlarged = side_by_side.repeat(6, axis=0).repeat(6, axis=1)
    imageio.imwrite(out_dir / 'cameras.png', enlarged)
    return f'front and rear images saved in {out_dir} (MUJOCO_GL={os.environ.get("MUJOCO_GL")})'


def check_speed(env, rng):
    start = time.perf_counter()
    for _ in range(SPEED_DECISIONS):
        random_step(env, rng)
    per_second = SPEED_DECISIONS / (time.perf_counter() - start)
    episode_seconds = (1000 / ACTION_REPEAT) / per_second
    return (f'{per_second:.0f} decisions per second with random actions '
            f'({ACTION_REPEAT} simulator steps and 2 camera images each), '
            f'so a 1000-step episode takes about {episode_seconds:.1f} s before the agent is added')


def run(name, check, *args):
    try:
        print(f'PASS  {name}: {check(*args)}', flush=True)
        return True
    except Exception as error:
        print(f'FAIL  {name}: {type(error).__name__}: {error}', flush=True)
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', default='check_env_out', help='folder for the camera images')
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = [
        run('GPU visible', check_gpu_visible),
        run('JAX on the GPU', check_jax_on_gpu),
    ]

    # The last two checks share one simulator, so they are skipped if it cannot start.
    rng = np.random.default_rng(0)
    try:
        env = make_env()
    except Exception as error:
        print(f'FAIL  simulator start: {type(error).__name__}: {error}', flush=True)
        sys.exit(1)
    results.append(run('rendering', check_rendering, env, rng, out_dir))
    results.append(run('speed', check_speed, env, rng))

    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
