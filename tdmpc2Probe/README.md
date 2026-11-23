# TD-MPC2 cost probe

Can a small predictor that reads the latent state of a frozen TD-MPC2 agent predict a cost that we define on Meta-World? This is the cost-predictor test of `adapterRecipe/`, with the other agent.

The agent is a released single-task TD-MPC2 checkpoint, playing with its planner as it normally does. The cost is ours: a sphere the gripper must not enter, centred halfway between where the gripper starts and the first thing it has to reach. A decision costs 1 while the gripper is inside. The cost is worked out from the gripper's position in the observation, so the simulator is not changed and the agent never sees it.

`collect.py` plays seeded episodes and saves one file per episode: for every decision the latent state `z`, the observation, the action, the reward, success, the gripper's position, its distance to the centre of the region, and the cost. Its docstring describes the arrays and how to choose the radius. The files are laid out as `adapterRecipe/episodes.py` lays out its own, so `adapterRecipe/cost_predictor.py` can read them once it takes `z` as the state's numbers.

## The image

`build_image.sh` builds `~/images/tdmpc2.sif` (about 10 GB) from the image TD-MPC2's authors published, `nicklashansen/tdmpc2:1.0.0`, pinned by its checksum. That image holds the software the released Meta-World agents were made with: Python 3.9, PyTorch 2.3 (a nightly build for CUDA 12.1, with its CUDA libraries), Meta-World at the commit TD-MPC2 names, gym 0.21.0, and mujoco-py compiled against MuJoCo 2.1.0. Nothing is installed during the build, because an install step cannot run in a build on the cluster. Two folders are copied in: TD-MPC2's code at commit `02b18a4`, which is the commit that image was built for, and MuJoCo 2.1.0, because the image's own copy is in a folder Apptainer cannot open.

`requirements.txt` lists the versions the image should hold. `check_env.py` compares them with what is there, then starts a Meta-World task and an untrained agent.

## Commands

From this folder, once the branch is pushed and `../msCli/ms sync` has run. Each `ms submit` prints a job id for `ms wait` and `ms logs`.

```bash
# 1. Build the image. On bigbatch: the build needs that much /tmp.
../msCli/ms submit -p bigbatch -t 3h -n build-tdmpc2-image -- bash build_image.sh

# 2. Download the four agents, 31 MB each, to /datasets/smohan/tdmpc2_checkpoints.
../msCli/ms submit -t 30m -n get-tdmpc2-checkpoints -- bash download_checkpoints.sh

# 3. Check the image on a batch node, where the episodes will be played.
../msCli/ms submit -t 30m -n check-tdmpc2-image -- apptainer exec --nv \
    /home-mscluster/smohan/images/tdmpc2.sif python3 check_env.py

# 4. A first look: five episodes of one task, saved in the run folder.
../msCli/ms submit -t 30m -n collect-first -- apptainer exec --nv --bind /datasets/smohan \
    /home-mscluster/smohan/images/tdmpc2.sif python3 collect.py --episodes 5 --checkpoint \
    /datasets/smohan/tdmpc2_checkpoints/metaworld/mw-reach-1.pt

# 5. The episodes for the predictor: 200 of each task. cost_predictor.py holds back 60
#    of them, which leaves 140 to train on.
../msCli/ms submit -t 3h -n collect -- apptainer exec --nv --bind /datasets/smohan \
    /home-mscluster/smohan/images/tdmpc2.sif python3 collect.py --episodes 200 \
    --out /datasets/smohan/tdmpc2Probe/episodes --checkpoint \
    /datasets/smohan/tdmpc2_checkpoints/metaworld/mw-reach-1.pt \
    /datasets/smohan/tdmpc2_checkpoints/metaworld/mw-push-1.pt \
    /datasets/smohan/tdmpc2_checkpoints/metaworld/mw-button-press-topdown-1.pt \
    /datasets/smohan/tdmpc2_checkpoints/metaworld/mw-window-open-1.pt
```

Steps 1 and 2 do not depend on each other. Step 3 prints how many decisions per second the planner makes, which says how long step 5 will take. Step 4 prints, for each episode, the share of decisions inside the region, and at the end the radius that would give 5, 7.5 and 10 percent. Add `--radius` to step 5 if the default of 0.05 m is far off.

Step 5 leaves one folder per task, `/datasets/smohan/tdmpc2Probe/episodes/TASK/`, holding `seed_NNNN.npz` for each episode and `run.json`. A job that is restarted carries on from the episodes already saved.

## Tasks to start with

Four tasks in which the arm travels some way before it touches anything, so that a region on the way is crossed. The distances are worked out from the task definitions.

| Task | The gripper first goes to | Distance from the start |
|---|---|---|
| `mw-reach` | the goal, a point in the air | 0.20 to 0.33 m |
| `mw-push` | a puck on the table, which it then pushes to the goal | 0.18 to 0.23 m, mostly downwards |
| `mw-button-press-topdown` | the top of a button, which it presses down | about 0.4 to 0.5 m |
| `mw-window-open` | a window handle, which it slides sideways | about 0.2 to 0.4 m |

- `mw-reach` is the plainest case: the whole episode is one move, and the region lies on it.
- `mw-push` has an object to reach first and then a second move, which the region is not on.
- `mw-button-press-topdown` has the longest way to go, so the widest choice of radius.
- `mw-window-open` is a second long approach, in a different scene, to a handle whose place changes by up to 0.2 m between episodes.

The checkpoints are the ones named with seed 1. In TD-MPC2's published results, the agents trained with seed 1 succeed in all four tasks at the end of training.

## Not verified

Nothing here has been run. The files were written from reading the code of TD-MPC2, Meta-World, mujoco-py and Apptainer, the build history of the published image, and one checkpoint's list of arrays. Still to be seen on the cluster:

- **The build.** Whether the 9.8 GB image downloads from Docker Hub (it may refuse if the cluster has pulled too much within the hour; resubmit later), whether the estimated 35 GB of /tmp is enough, and how large the result is.
- **What is in the image.** Its build history, its package list and the install records of the packages in `requirements.txt` were read; the image as a whole was not downloaded or started. Whether its PyTorch runs on the cluster's driver and GPUs is not known.
- **mujoco-py under Apptainer.** That it starts with MuJoCo in `/opt/mujoco210`, with the CPU drawing code forced, and with its lock file swapped out (`prepare()` in `collect.py`). If the lock still fails, add `--writable-tmpfs` after `apptainer exec`.
- **TD-MPC2 started our way.** `prepare()` keeps TD-MPC2 from importing the simulators that are not needed, and `settings()` completes its settings without Hydra, the tool its own scripts start through.
- **The episodes.** That an episode is 101 entries, that the same seed gives the same episode (expected only on the same kind of node), that the positions sit where `collect.py` reads them in the observation (`check_env.py` compares them with the simulator), that the checkpoint named with seed 1 is the agent of the published seed 1, and that it succeeds as published.
- **The region.** Whether the gripper goes through it, and what share of decisions a radius of 0.05 m charges. The four distances above are from reading the task definitions.
- **Speed.** The times given to the jobs are guesses.
