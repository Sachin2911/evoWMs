"""Checks that the TD-MPC2 image works on a cluster node.

Five checks: the GPU is visible, PyTorch computes on it, the packages are the versions
requirements.txt lists, a Meta-World task starts and steps, and a TD-MPC2 agent encodes an
observation and plans. The agent is an untrained one, so no checkpoint is needed and
nothing is downloaded. Run it inside the image, from this folder:

  ../msCli/ms submit -t 30m -n check-tdmpc2-image -- apptainer exec --nv \\
      /home-mscluster/smohan/images/tdmpc2.sif python3 check_env.py

It prints one PASS or FAIL line per check and exits with an error if any failed.
"""

import importlib
import json
import pathlib
import re
import subprocess
import sys
import time
from importlib import metadata

import numpy as np

from collect import GOAL, GRIPPER, OBJECT, latent, prepare, settings

TASK = 'mw-reach'
SPEED_DECISIONS = 50    # fewer than the 100 of an episode, so the simulator is not reset while timing


def check_gpu_visible():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
        capture_output=True, text=True, check=True)
    return result.stdout.strip()


def check_torch_on_gpu():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f'PyTorch {torch.__version__} cannot use the GPU')

    # A calculation with a known answer: a 512x512 matrix of ones times itself sums to 512^3.
    ones = torch.ones((512, 512), device='cuda')
    total = float((ones @ ones).sum())
    if total != 512.0 ** 3:
        raise RuntimeError(f'wrong answer from the GPU: {total}')
    return (f'PyTorch {torch.__version__}, built for CUDA {torch.version.cuda}, '
            f'on {torch.cuda.get_device_name(0)}, calculation correct')


def installed_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        # A package installed by conda can lack the record pip leaves, so ask the package itself.
        return importlib.import_module(name.replace('-', '_')).__version__


def check_versions():
    wrong, found = [], [f'Python {sys.version.split()[0]}']
    if sys.version_info[:2] != (3, 9):
        wrong.append('Python is not 3.9')
    for line in (pathlib.Path(__file__).resolve().parent / 'requirements.txt').read_text().splitlines():
        line = line.split('#')[0].strip()
        if not line:
            continue
        if ' @ ' in line:
            # Installed from a commit of a git repository, which pip records beside the package.
            name, source = line.split(' @ ')
            wanted = source.rsplit('@', 1)[1]
            record = metadata.distribution(name).read_text('direct_url.json')
            have = json.loads(record).get('vcs_info', {}).get('commit_id') if record else None
            if have is None:
                found.append(f'{name} (its commit is not recorded in the image)')
                continue
            right = have == wanted
        else:
            name, wanted = re.match(r'([\w.-]+)==(\S+)$', line).groups()
            have = installed_version(name)
            # What follows a + names the build, such as +cu121, and is not part of the version.
            right = have.split('+')[0] == wanted
        found.append(f'{name} {have}')
        if not right:
            wrong.append(f'{name} is {have} and not {wanted}')
    if wrong:
        raise RuntimeError('; '.join(wrong) + '. Found: ' + ', '.join(found))
    return ', '.join(found)


def make_simulator():
    # Built through TD-MPC2's own code and in the same order as in collect.py, so this
    # exercises what collect.py will use.
    prepare()
    # Imported by name first, so that a failure shows its own message. TD-MPC2 would
    # carry on without Meta-World and only say later that it cannot make the task.
    import metaworld.envs  # noqa: F401
    from envs import make_env

    cfg = settings(TASK)
    return make_env(cfg), cfg


def check_simulator(env, cfg):
    import torch
    from common.seed import set_seed

    set_seed(0)
    first = env.reset().numpy()
    obs, reward, done, info = env.step(torch.zeros(cfg.action_dim))
    obs = obs.numpy()
    if obs.shape != (39,) or cfg.action_dim != 4 or cfg.episode_length != 100:
        raise RuntimeError(f'observations of {obs.shape}, actions of {cfg.action_dim} numbers and '
                           f'{cfg.episode_length} decisions per episode, not (39,), 4 and 100')
    if 'success' not in info:
        raise RuntimeError('the simulator does not say whether the task is done')

    # collect.py reads the positions from the observation. After a step the observation
    # and the simulator describe the same moment, so they can be compared here.
    sawyer = env.unwrapped
    in_simulator = {
        'gripper': (GRIPPER, sawyer.get_endeff_pos()),
        'object': (OBJECT, sawyer._get_pos_objects()[:3]),
        'goal': (GOAL, sawyer._target_pos),
    }
    for name, (where, position) in in_simulator.items():
        if not np.allclose(obs[where], position, atol=1e-5):
            raise RuntimeError(f'the simulator has the {name} at {position}, but numbers {where.start} '
                               f'to {where.stop - 1} of the observation are {obs[where]}')
    away = np.linalg.norm(first[GOAL] - first[GRIPPER])
    return (f'{TASK}: 39 numbers observed, 4 per action, 100 decisions per episode; the gripper, object '
            f'and goal in the observation are where the simulator has them; the gripper starts at '
            f'{np.round(first[GRIPPER], 3)}, {away:.3f} m from the goal')


def check_agent(env, cfg):
    from tdmpc2 import TDMPC2

    # Untrained: nothing is loaded. It plans exactly as much as a trained one.
    agent = TDMPC2(cfg)
    obs = env.reset()
    z = latent(agent, obs)
    groups = z.reshape(-1, cfg.simnorm_dim).sum(-1)
    if z.shape != (512,) or z.min() < 0 or not np.allclose(groups, 1, atol=1e-4):
        raise RuntimeError(f'the latent state has shape {z.shape}, and its groups of {cfg.simnorm_dim} add up to '
                           f'{groups.min():.4f} at least and {groups.max():.4f} at most, not 1')

    # The first decision also starts up the GPU, so it is not timed.
    action = agent.act(obs, t0=True, eval_mode=True)
    obs, _, _, _ = env.step(action)
    start = time.perf_counter()
    for _ in range(SPEED_DECISIONS):
        action = agent.act(obs, t0=False, eval_mode=True)
        obs, _, _, _ = env.step(action)
    per_second = SPEED_DECISIONS / (time.perf_counter() - start)
    if action.shape != (4,) or float(action.abs().max()) > 1:
        raise RuntimeError(f'the planner gave {action}, not 4 numbers between -1 and 1')
    return (f'latent state of 512 numbers in groups that add up to 1; {per_second:.0f} decisions per second '
            f'with the planner, so an episode takes about {100 / per_second:.0f} s')


def run(name, check, *args):
    try:
        print(f'PASS  {name}: {check(*args)}', flush=True)
        return True
    except Exception as error:
        print(f'FAIL  {name}: {type(error).__name__}: {error}', flush=True)
        return False


def main():
    results = [
        run('GPU visible', check_gpu_visible),
        run('PyTorch on the GPU', check_torch_on_gpu),
        run('versions', check_versions),
    ]

    # The last two checks share one simulator, so they are skipped if it cannot start.
    try:
        env, cfg = make_simulator()
    except Exception as error:
        print(f'FAIL  simulator start: {type(error).__name__}: {error}', flush=True)
        sys.exit(1)
    results.append(run('simulator', check_simulator, env, cfg))
    results.append(run('agent', check_agent, env, cfg))

    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
