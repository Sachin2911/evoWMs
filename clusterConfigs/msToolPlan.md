# Plan: `ms`, a command-line tool for running jobs on mscluster

**Status: unblocked, not started.** Written 10 November 2026 and parked until MSS fixed the `smohan` account. MSS added the Slurm association the same day, and `sbatch` and `srun` were both verified working, so every command below can now be built and tested. Waiting on Sachin's go-ahead and the open questions at the end.

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
- **Python, standard library only.** Shells out to the system `ssh` and `rsync`, so it uses the existing `msCluster` alias and key. (Proposed, not yet confirmed with Sachin.)
- **The guide stays** as the reference for humans and as the fallback when the tool fails. `AGENTS.md` will tell agents to use `ms` first.

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
| `ms logs JOBID [-n N] [--err]` | Tail of the job's log, default 40 lines |
| `ms sync` | Push the repo to the cluster (see open questions) |
| `ms submit [opts] -- CMD` | Generate the `sbatch` script from a template, submit, print the job id |
| `ms wait JOBID...` | Block locally, checking every few minutes, exit when the jobs finish; exit code reflects success |
| `ms cancel JOBID` | `scancel` |
| `ms fetch JOBID [PATH]` | Pull results back to the laptop |

Output rules: plain text, one line per item, no banners, no raw Slurm tables. Errors say what failed and what to do next. `--json` on the read commands for scripts.

`ms wait` is what removes polling: an agent starts it in the background and is only woken when the job ends.

## Defaults baked into `submit`

So that an agent passes only the command and any overrides:

- Partition `batch`; `--partition bigbatch|biggpu` only when asked for.
- One node, `--time` required from the caller or a short default (to decide).
- Job name from the repo and a short label; log at `~/runs/<job-name>-<jobid>/slurm.out`, with the run directory created before submission.
- Script header loads Lmod (`source /etc/profile.d/lmod.sh`), sets CUDA paths, activates the project environment.
- Records the git commit the job ran from in the run directory.

## Status-file convention

The largest saving. Each job writes a small file as it runs, for example `~/runs/<run>/status.json`:

```json
{"step": 1200, "of": 5000, "metric": 0.431, "checkpoint": "ckpt-1200.pt", "updated": "2026-11-12T08:14:00Z"}
```

`ms status` reads that one line, so "how is it going" never means an agent interpreting a training log. The same file tells a restarted job where to resume, which the cluster's outages require anyway. A tiny helper for writing it should live in the research code.

## Build order

1. `nodes`, `status`, `logs`, `sync`, plus the shared SSH wrapper (BatchMode, timeout, single session, stdin closed for remote commands).
2. `submit`, `wait`, `cancel`, `fetch`, tested with a one-minute job on `batch`.
3. The status-file helper, and the section in `AGENTS.md` that makes `ms` the default route.

## What the first probe jobs settled (10 November 2026)

Details are in `msClusterUserGuide.md`. The points that shape the tool:

- `module` is not defined inside a batch job. The template must `source /etc/profile.d/lmod.sh` first.
- The `pytorch/...` modules exist on compute nodes but not on the login node, and the module list differs between partitions. `submit` cannot validate a module name from the login node.
- Compute nodes have outbound internet, so a job can build its own environment and fetch its own data. Environment setup can therefore be a job, which keeps it off the login node.
- There is no `/scratch`. Node-local space is `/tmp`, which is RAM-backed (16 GB on `batch`, 62 GB on `bigbatch`). Run directories and checkpoints go on NFS.
- `batch` nodes have one RTX 3060 (12 GB), 6 cores / 12 threads, 30 GB RAM. That is the default target.
- GPUs differ by node, not only by partition (`biggpu` mixes 2× 48 GB and 1× 96 GB machines), and Slurm does not track them. `submit` needs a way to pick nodes by GPU (`--nodelist`), driven by the node map in the guide.
- `sinfo -t idle` includes drained nodes. `ms nodes` must count a node as free only when its state is exactly `idle`.
- The system modules are Apptainer images exposed as shell functions, which do not survive into child processes. The job template should call `apptainer exec --nv "$PYTORCH_SIF" ...` directly rather than rely on a bare `python`.
- The `pytorch` CUDA 13.3 image has Python 3.11, PyTorch 2.12.1, JAX, and `wandb` already; `gymnasium` and `lightning` are missing.
- `srun` forwards stdin to the task, so inside a piped SSH script it needs `</dev/null`.

## Still to verify

- How a requeued job looks to `squeue`/`sacct`, so `wait` does not report it as finished.
- How to add packages on top of the `pytorch` image (conda env under `~/.conda`, or `pip install --user`), tested in a job.
- Hardware on the nodes that were busy or down during probing (`stampede`, `gpuexpress`, four `biggpu` nodes), if they are ever needed.

## Open questions

- **Code sync:** `rsync` from the laptop, or `git pull` on the cluster from a remote? Rsync needs no remote and carries uncommitted work; git gives an exact commit per run.
- **Environment:** the `pytorch` module, the `anaconda` module, a personal venv in home, or an Apptainer image? Depends on the research stack, which does not exist yet.
- **Data:** the sideload hosts were down on 10 November 2026. Compute nodes have internet, so a download job writing to `/datasets/smohan` is a working alternative until MSS answers.
- **Sweeps:** job arrays (limit 1001) or many tasks packed into one whole node? Whole-node allocation favours packing.
- **Name:** `ms` is short but generic; rename if it collides with something.
