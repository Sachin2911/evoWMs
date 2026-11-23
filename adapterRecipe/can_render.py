"""Tests whether this node can draw the cameras on its GPU, and holds the node if it cannot.

A few nodes fail with "Cannot initialize a EGL device display". A job that fails frees its
node at once, so the next job lands on the same node and fails too. Run this first: on a
node that cannot draw it sleeps for --hold hours, which keeps the node away from the jobs
submitted after it. On a node that can draw it ends at once.

  ../msCli/ms submit -t 8h -n can-render -- apptainer exec --nv --env MUJOCO_GL=egl \\
      /home-mscluster/smohan/images/safeDreamer.sif python3 can_render.py --hold 6
"""

import argparse
import socket
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--hold', type=float, default=6, help='hours to hold a node that cannot draw')
    args = parser.parse_args()
    node = socket.gethostname()
    try:
        import mujoco.egl  # noqa: F401  (importing it sets up the display, which is the step that fails)
    except Exception as problem:
        print(f'{node} cannot draw: {str(problem).splitlines()[0]}. Holding it for {args.hold:g} hours.', flush=True)
        time.sleep(args.hold * 3600)
        return
    print(f'{node} can draw', flush=True)


if __name__ == '__main__':
    main()
