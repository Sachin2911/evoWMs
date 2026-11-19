# Plan: `ms`, a command-line tool for running jobs on mscluster

**Status: built.** The tool is in `msCli/`, and `msCli/README.md` is the usage reference. This file records the design and the reasons for it.

## Why

All compute for this research runs on mscluster, and agents (Claude Code, Cursor) will submit, monitor, and collect jobs hundreds of times. Done by hand each time, an agent rereads `msClusterUserGuide.md`, composes an `sbatch` script, polls `squeue`, and reads raw logs. That is slow, costs tokens on every run, and each improvised SSH session is a chance to break a login-node rule.

The tool moves that knowledge into code so a run is one or two short commands with compact output.

## Decisions

- **A CLI in this repo, not an MCP server or a Claude skill.**
  - Every agent and the human have a shell. MCP needs separate configuration per tool; a skill is Claude-only.
  - A CLI costs no tokens until it is called. MCP tool schemas load into every session.
  - Each call is one short SSH command that exits, which is what the login node allows. A long-lived server holding a connection is what MSS kills.
  - An MCP wrapper can be added later on top of the same commands if it turns out to be wanted.
- **Runs on the laptop.** Nothing is installed or left running on the login node.
- **Python, standard library only.** Shells out to the system `ssh` and `rsync`, so it uses the existing `msCluster` alias and key.
- **The research code is configured with Hydra, and the tool does not know it.** The tool runs a command and provides a run directory; Hydra settings are part of the command. The few points where the two meet are in the Hydra section.
- **The guide stays** as the reference for humans and as the fallback when the tool fails. `AGENTS.md` tells agents to use `ms` first.

## Constraints the tool must respect

From `msClusterUserGuide.md`; these are the rules that shaped the design.

- Login node has one core and a 10% per-user cap. One SSH session per command, all remote work batched into it, nothing heavy (`du`, builds, installs) ever run there.
- No persistent processes or connections on the login node.
- The login node bans IPs after a few failed logins. Always `ssh -o BatchMode=yes`; never fall back to a password.
- SSH can take over a minute when the node is loaded. Generous timeouts, and one retry at most.
- Slurm allocates whole nodes and has no GRES. Never emit `--gres`; pack small tasks into one node.
- Partitions have no default time limit. Always set `--time`.
- Jobs are requeued after the frequent power, cooling, and network failures. Everything submitted must be restartable.
- Per-user QoS caps (for example 13 running jobs on `batch`, 6 on `bigbatch`). Refuse to submit past them with a clear message rather than letting jobs pile up pending.

## Commands

| Command | Does |
|---|---|
| `ms nodes` | Partition summary: idle / busy / down per partition, one line each |
| `ms status [JOBID]` | My jobs, one line each: id, name, partition, state, elapsed, node, last status-file line |
| `ms logs JOBID [-n N] [--task K]` | Tail of the job's log, default 40 lines. Output and errors share one file, so a traceback is at the end of it |
| `ms ls [PATH]` | List a directory, one line per entry |
| `ms rm PATH...` | Delete files or directories |
| `ms mv SRC DST` | Move or rename |
| `ms sync` | Clone or update the cluster's copy of the repo from GitHub. Refuses if the local commit has not been pushed |
| `ms submit [opts] -- CMD` | Generate the `sbatch` script from a template, submit, print the job id |
| `ms wait JOBID...` | Block locally, checking every few minutes, exit when the jobs finish; exit code reflects success |
| `ms cancel JOBID` | `scancel` |
| `ms fetch JOBID [PATH]` | Pull results back to the laptop. Files over 50 MB are skipped and named unless `--all` is given |

Output rules: plain text, one line per item, no banners, no raw Slurm tables. Errors say what failed and what to do next. `--json` on the read commands for scripts.

`ms wait` is what removes polling: an agent starts it in the background and is only woken when the job ends.

## Defaults baked into `submit`

So that an agent passes only the command and any overrides:

- Partition `batch`; `--partition bigbatch` when asked for. No other partition is accepted.
- One node per job. `--time` is set by the caller and rejected if it exceeds the partition's limit (1 day on `batch`, 3 days on `bigbatch`). One hour when omitted.
- Job name from the repo and a short label; log at `~/runs/<job-name>-<jobid>/slurm.out`. The job is submitted held, its run directory is created, and it is then released, so the directory exists before the job starts.
- Script header loads Lmod (`source /etc/profile.d/lmod.sh`) and activates `.venv` in the working directory if there is one. It does not set CUDA paths: PyTorch installed with pip carries its own CUDA libraries.
- Records the git commit the job ran from in the run directory.
- Exports `MS_RUN_DIR`, the run directory, to the job.
- Everything after `--` is passed through untouched.

## Allowed folders

The tool has access to two folders on the cluster, hardcoded:

```
/home-mscluster/smohan
/datasets/smohan
```

- Every command that takes a remote path (`ls`, `rm`, `mv`, `fetch`, and the run directory of `submit`) refuses a path outside them.
- The check is made on the cluster against the resolved path (`realpath`), so `..` and symlinks cannot lead outside.
- `rm` and `mv` act on things inside the two folders, never on the two folders themselves, and never on `~/.ssh`, which holds the login keys.
- `rm` deletes outright, with no trash folder. Code is backed up on GitHub and data and checkpoints on Hugging Face.
- A move between the two folders, or the deletion of a large tree, is heavy I/O on NFS and runs as a job, not on the login node. The move is a copy followed by a delete, so a job requeued halfway can run again safely.

## Hydra

The tool never reads a config file and never checks that an override names a real setting. A wrong override fails within seconds of the job starting, and `ms logs` shows the error.

```
ms submit --time 02:00:00 -- python train.py data=pusht optimizer.lr=0.0003
```

**Run directory.** Hydra's default is a new timestamped directory on every start, so a requeued job would lose its own checkpoints. The research code points Hydra at the tool's directory instead, once, in its root config:

```yaml
hydra:
  run:
    dir: ${oc.env:MS_RUN_DIR,outputs/local}
```

- On the cluster, Hydra's saved config (`.hydra/config.yaml`), the Slurm log, the status file, and the checkpoints share one directory that survives a requeue.
- On the laptop the variable is unset, so runs go to `outputs/local`.
- The code writes every output under that directory, taking the path from `HydraConfig.get().runtime.output_dir` and not from the current working directory, which Hydra may or may not change depending on its version.
- `ms fetch` brings `.hydra/config.yaml` back with the results, so each result carries the settings that produced it.

**Sweeps.** Hydra's `--multirun` runs its combinations one after another in a single process, and its Slurm launcher plugin keeps a process waiting on the login node. Neither is used on the cluster. The tool expands the sweep and each task is an ordinary single Hydra run; see `--grid` under Sweeps. `--multirun` remains fine for quick tests on the laptop.

**Quoting.** Overrides can contain brackets, commas, quotes, and `$`. The command passes through the local shell, `ssh`, and `sbatch` before it reaches Python, and must arrive unchanged. The tool carries it as data (written into the generated script, not interpolated into a remote command line).

## Status-file convention

The largest saving. Each job writes a small file as it runs, `$MS_RUN_DIR/status.json`:

```json
{"step": 1200, "of": 5000, "metric": 0.431, "checkpoint": "ckpt-1200.pt", "updated": "2026-11-12T08:14:00Z"}
```

`ms status` reads that one line, so "how is it going" never means an agent interpreting a training log. The same file tells a restarted job where to resume, which the cluster's outages require anyway. A tiny helper for writing it should live in the research code.

## Sweeps

A sweep is N independent tasks, numbered 0 to N-1. The tasks never talk to each other, so no MPI or multi-node job is involved: several nodes are used by running several one-node jobs.

```
ms submit --array 100 --per-node 4 --time 02:00:00 -- python eval.py task={i} ntasks={n}
ms submit --grid "seed=0,1,2 optimizer.lr=1e-3,3e-4" --time 04:00:00 -- python train.py data=pusht
```

- `--array N` submits one Slurm job array. `{i}` is replaced by the task index and `{n}` by N, so they arrive as ordinary Hydra overrides (`task=42 ntasks=100`). The research code maps the index to its share of the work (a seed, or a range of episodes).
- `--grid` is the alternative to `--array N` for sweeping settings. The tool expands the comma lists into every combination (six in the example) and appends one combination to each task's command, for example `seed=1 optimizer.lr=3e-4`.
  - Only comma lists are expanded. Hydra's `range()` and `choice()` sweep syntax is not understood.
  - A value that itself contains a comma cannot go in `--grid`; pass it after `--`, or use `--array` and pick the value in code.
  - The tool writes the expanded list to `~/runs/<sweep>/tasks.txt`, one line per task, so the mapping from task number to settings is on record.
- `--per-node P` packs P tasks onto each node as parallel processes, because a job always holds a whole node. The array then has N/P elements, rounded up.
- The array's concurrency is capped at the per-user QoS limit (13 running jobs on `batch`, 6 on `bigbatch`); the rest wait in the queue.
- Each task writes to its own directory, `~/runs/<sweep>/task-0042/`, with its own log and status file. `MS_RUN_DIR` is set per task, not per job, so every task is a separate Hydra run with its own `.hydra/config.yaml`.
- The tool records each task's outcome in the sweep directory. A task recorded as done is skipped, so requeues and resubmissions repeat no work.
- `ms status SWEEP` prints one line: done, running, pending, failed, and the indices of failed tasks. `ms submit --retry SWEEP` resubmits only the failed ones.
- Merging per-task results into one table is done by the research code, on the laptop, after `ms fetch`.

## What the first probe jobs settled (10 November 2026)

Details are in `msClusterUserGuide.md`. The points that shape the tool:

- `module` is not defined inside a batch job. The template must `source /etc/profile.d/lmod.sh` first.
- The `pytorch/...` modules exist on compute nodes but not on the login node, and the module list differs between partitions. `submit` cannot validate a module name from the login node.
- Compute nodes have outbound internet, so a job can build its own environment and fetch its own data. Environment setup can therefore be a job, which keeps it off the login node.
- There is no `/scratch`. Node-local space is `/tmp`, which is RAM-backed (16 GB on `batch`, 62 GB on `bigbatch`). Run directories and checkpoints go on NFS.
- `batch` nodes have one RTX 3060 (12 GB), 6 cores / 12 threads, 30 GB RAM. That is the default target.
- Every node within `batch` has the same GPU, and likewise within `bigbatch`, so choosing the partition chooses the GPU. `submit` needs no node selection.
- `sinfo -t idle` includes drained nodes. `ms nodes` must count a node as free only when its state is exactly `idle`.
- The system modules are Apptainer images exposed as shell functions, which do not survive into child processes. The job template should call `apptainer exec --nv "$PYTORCH_SIF" ...` directly rather than rely on a bare `python`.
- The `pytorch` CUDA 13.3 image has Python 3.11, PyTorch 2.12.1, JAX, and `wandb` already; `gymnasium` and `lightning` are missing.
- `srun` forwards stdin to the task, so inside a piped SSH script it needs `</dev/null`.

## What testing the built tool settled

- A requeued job returns to `PENDING` in `squeue` under the same job id, so `wait` treats any job still in the queue as unfinished and only then reads its final state from `sacct`.
- A requeued job restarts in the same run directory. Slurm empties the log on restart unless told to append, so jobs are submitted with `--open-mode=append`.
- `${oc.env:MS_RUN_DIR,outputs/local}` resolves as intended, and Hydra leaves the working directory alone (Hydra 1.3.8, Python 3.12).
- Hydra does not start on Python 3.14, which is the compute nodes' system Python. The research environment needs an older Python.
- A folder deleted from the login node can stay cached for a short while on a compute node that used it, and a job touching that path then fails with `Stale file handle`.

## Still to verify

- How to add packages on top of the `pytorch` image (conda env under `~/.conda`, or `pip install --user`), tested in a job.
- How many processes can usefully share one node's GPU, measured on a real workload, to set a sensible `--per-node`.

## Open questions

- **Environment:** the `pytorch` module, the `anaconda` module, a personal venv in home, or an Apptainer image? Depends on the research stack, which does not exist yet.
- **Data:** the sideload hosts were down on 10 November 2026. Compute nodes have internet, so a download job writing to `/datasets/smohan` is a working alternative until MSS answers.
