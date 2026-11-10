# Plan: `ms`, a command-line tool for running jobs on mscluster

**Status: parked.** Written 10 November 2026. Start once MSS has fixed the `smohan` account (no Slurm association, not in `mscluster-users`; ticket sent 10 November 2026). Check with:

```bash
ssh msCluster 'sacctmgr -nP show user $USER withassoc'   # empty output = still broken
```

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

| Command | Does | Needs working account |
|---|---|---|
| `ms nodes` | Partition summary: idle / busy / down per partition, one line each | no |
| `ms status [JOBID]` | My jobs, one line each: id, name, partition, state, elapsed, node, last status-file line | no (empty until jobs exist) |
| `ms logs JOBID [-n N] [--err]` | Tail of the job's log, default 40 lines | no |
| `ms sync` | Push the repo to the cluster (see open questions) | no |
| `ms submit [opts] -- CMD` | Generate the `sbatch` script from a template, submit, print the job id | **yes** |
| `ms wait JOBID...` | Block locally, checking every few minutes, exit when the jobs finish; exit code reflects success | **yes** |
| `ms cancel JOBID` | `scancel` | **yes** |
| `ms fetch JOBID [PATH]` | Pull results back to the laptop | **yes** |

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

1. **Now possible:** `nodes`, `status`, `logs`, `sync`, plus the shared SSH wrapper (BatchMode, timeout, single session, stdin closed for remote commands).
2. **After the account works:** `submit`, `wait`, `cancel`, `fetch`, tested with a one-minute job on `batch`.
3. **Then:** the status-file helper, and the section in `AGENTS.md` that makes `ms` the default route.

## To verify with the first real job

These are gaps in the guide that also affect the template. Fill them into `msClusterUserGuide.md` at the same time.

- GPU model and driver on `batch` nodes, on the small `biggpu` nodes, and on `gpuexpress`.
- Whether `/scratch` exists on compute nodes and how big it is.
- Whether `module` works inside a batch script, or needs the explicit `source`.
- Whether the `pytorch/...` modules exist on compute nodes.
- Whether compute nodes have internet access (decides how environments and data get there).
- How a requeued job looks to `squeue`/`sacct`, so `wait` does not report it as finished.

## Open questions

- **Code sync:** `rsync` from the laptop, or `git pull` on the cluster from a remote? Rsync needs no remote and carries uncommitted work; git gives an exact commit per run.
- **Environment:** the `anaconda` module, a personal conda/venv in home, or an Apptainer image? Depends on the research stack, which does not exist yet.
- **Data:** the sideload hosts were down on 10 November 2026. How datasets reach `/datasets/smohan` depends on MSS's answer.
- **Sweeps:** job arrays (limit 1001) or many tasks packed into one whole node? Whole-node allocation favours packing.
- **Name:** `ms` is short but generic; rename if it collides with something.
