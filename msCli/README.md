# ms

A command-line tool for running jobs on mscluster from the laptop. Each command opens one short SSH session to the login node and exits; nothing is installed or left running on the cluster.

It needs Python 3 and the `msCluster` alias in `~/.ssh/config`. Run it in place as `msCli/ms`, or put it on the PATH:

```bash
ln -s "$PWD/msCli/ms" ~/.local/bin/ms
```

## Commands

| Command | Does |
|---|---|
| `ms nodes` | Idle, busy, and down nodes per partition, and my jobs against the per-user caps |
| `ms status [JOBID...]` | My jobs, one line each: queued, running, and finished in the last day |
| `ms logs JOBID [-n N] [--task K]` | Last lines of a job's log (40 by default) |
| `ms sync` | Clone or update the cluster's copy of the current git repo from `origin` |
| `ms submit [options] -- COMMAND` | Submit a job and print its id |
| `ms wait JOBID...` | Block until the jobs finish; exit code 0 only if all succeeded |
| `ms cancel JOBID...` | Cancel jobs |
| `ms fetch JOBID_OR_PATH [DEST]` | Copy a run folder to the laptop (`results/` by default) |
| `ms ls [PATH]` | List a folder on the cluster |
| `ms rm PATH...` | Delete on the cluster; there is no undo |
| `ms mv SRC DST` | Move or rename on the cluster |

`nodes`, `status`, `ls`, and `wait` take `--json`.

## A run, start to finish

```bash
git push                                   # the cluster pulls from GitHub
ms sync                                    # -> /home-mscluster/smohan/<repo>
ms submit -t 2h -- python train.py data=pusht optimizer.lr=0.0003
ms wait 41235                              # returns when it ends
ms logs 41235
ms fetch 41235                             # -> results/<repo>-train-41235/
```

## What `submit` does

- Runs the command on one whole node, in the cluster's copy of the repo (same subfolder as the one `ms` was called from). `--workdir` overrides this.
- Partition `batch` unless `-p bigbatch`. Time limit one hour unless `-t` says otherwise (`02:00:00`, `90m`, `1d`); a limit over the partition's cap is refused.
- Creates the run folder `~/runs/<repo>-<label>-<jobid>/` before the job starts and exports it as `MS_RUN_DIR`. The folder holds `slurm.out`, `job.sbatch`, `command.txt`, and `commit.txt` (the commit the job ran from).
- Activates `.venv` in the working folder if there is one (`--env DIR` for another). `--module NAME` loads a module.
- Sets `HF_TOKEN_PATH` to `~/credentials/hf_token` if that file exists, so Hugging Face downloads and uploads in a job are authenticated. The token stays in that one file on the cluster; it is never written into a script, a log, or this repo.
- Passes everything after `--` to the node unchanged. No shell on the way interprets it, so quotes, brackets, `$`, and `*` arrive as typed, and shell features such as `&&` or `|` are not available; put those in a script and run the script.
- Refuses to queue past the per-user cap for the partition.

A requeued job keeps its job id, so it restarts in the same `MS_RUN_DIR` and its log is appended to.

## Sweeps

A sweep is N independent tasks submitted as one job array. Each task gets its own folder, `task-0000/`, `task-0001/`, ..., as its `MS_RUN_DIR`, with its output in `task.log`.

```bash
ms submit --array 100 --per-node 4 -t 2h -- python eval.py task={i} ntasks={n}
ms submit --grid "seed=0,1,2 optimizer.lr=1e-3,3e-4" -t 4h -- python train.py data=pusht
```

- `--array N`: `{i}` becomes the task number (0 to N-1) and `{n}` becomes N. `MS_TASK_ID` and `MS_NUM_TASKS` are also set.
- `--grid`: every combination of the comma lists becomes a task, and its settings are appended to the command. The list is saved as `tasks.txt`. Values with commas, brackets, or quotes cannot go in `--grid`.
- `--per-node P` runs P tasks side by side on each node.
- A task that finished is never run again, so a requeue repeats only unfinished work.
- `ms status` shows a sweep as one line with the numbers of the failed tasks. `ms logs JOBID --task K` shows one task.
- `ms submit --retry JOBID` resubmits the unfinished tasks into the same folder.

## Status file

A job that writes `$MS_RUN_DIR/status.json` has its latest values shown by `ms status`. [ms_status.py](ms_status.py) is a small helper for the research code: `write_status(step=..., of=..., metric=...)`, and `read_status()` for a restarted job to find where to resume.

## Hydra

Point Hydra at the run folder once, in the root config:

```yaml
hydra:
  run:
    dir: ${oc.env:MS_RUN_DIR,outputs/local}
```

Do not use `--multirun` on the cluster; use `--grid`. Hydra does not start on Python 3.14, the compute nodes' system Python, so the research code needs its own environment with an older one.

## Limits built in

- **Two folders only:** `/home-mscluster/smohan` and `/datasets/smohan`. A path is resolved on the cluster first, so `..` and symlinks cannot lead outside. The two folders themselves, and `~/.ssh`, cannot be deleted or moved.
- **Two partitions only:** `batch` and `bigbatch`.
- **Heavy file work runs as a job:** deleting a folder with more than 2000 entries, and moving anything between the two folders. The command prints the job id.
- **Nodes that cannot draw are skipped:** every job excludes the nodes in `config.EXCLUDED_NODES`, whose GPUs fail with "Cannot initialize a EGL device display". Add a node there when a job dies on it with that message.
- **`fetch` skips files over 50 MB** and names them; `--all` includes them.
- **One SSH attempt, key only.** A failed login is never retried, because the login node bans addresses after a few failures.

## Tests

```bash
python3 -m unittest discover -s msCli
```

They need no cluster: the login-node side runs locally against a fake Slurm.
