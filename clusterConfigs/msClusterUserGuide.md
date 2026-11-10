# mscluster operational reference

mscluster is the Wits Mathematical Sciences HPC cluster, run by Mathematical Sciences Support (MSS). It is for supervised research or coursework HPC only: not general storage, personal computing, or non-HPC work.

**Sources.** The cluster was rebuilt from scratch in mid-2026, so the 2024 community guide PDF is out of date on partitions, hosts, storage, and software. It is no longer kept in this repo; MSS still links it at the bottom of `/etc/motd`. This file is based on a live survey of the cluster on 10 November 2026 plus the login notices in `/etc/motd`, and it wins wherever it disagrees with the PDF. Policy that could not be checked live is marked *(2024 guide)*. Things nobody has verified yet are marked *(unverified)*.

`/etc/motd` is MSS's running changelog (outages, policy changes, new modules). Read the tail of it when something behaves unexpectedly.

## Status of our account (10 November 2026)

- `smohan` can log in, and home and `/datasets/smohan` are writable.
- **Job submission does not work yet.** The user has no Slurm association and is not in the `mscluster-users` group, so every `sbatch`/`srun` fails with `Invalid account or account/partition combination specified`. A ticket was sent to MSS on 10 November 2026. Check with `sacctmgr show user $USER withassoc`; a working user shows account `general`, QoS `normal`.
- Because no job could run, nothing inside a compute node has been seen first-hand. GPU models, drivers, and local disk on the workers are from notices or the 2024 guide.

## Access

```bash
ssh msCluster        # alias in ~/.ssh/config for smohan@146.141.21.100 (also: wits-hpc)
```

| Role | Host | Address | State on 10 Nov 2026 |
|---|---|---|---|
| Login | `mscluster-login` | `146.141.21.100` | up. Lands in `/home-mscluster/$USER`. |
| Slurm controller | `mscluster0` | `10.100.14.253` | up |
| Submit host | `mscluster-submit` | `10.100.14.252` | up, SSH open. Purpose not documented. |
| Datasets sideload | `mscluster-datasets-sideload` | `10.100.14.248` | no ping, SSH closed |
| Home sideload | `mscluster-files-sideload` | `10.100.14.250` | no ping, SSH closed |
| Datasets NFS server | `mscluster-datasets` | `10.100.14.249` | up |
| Home NFS server | `mscluster-files` | `10.100.14.251` | up |

- There is no backup login node. `146.141.21.101` is now `openondemand.ms.wits.ac.za` and answers on no port; it is probably the IDE portal MSS has promised, not yet live.
- The internal `10.100.14.x` hosts are reachable only from inside the cluster.
- Old SSH keys were not migrated in the rebuild, and MSS treats any key used on the old cluster as compromised. Use a fresh key.
- The login node appears to ban an IP after a few failed authentication attempts. Offer one key only (`IdentitiesOnly yes`) and never guess passwords.

Help: one ticket per issue, from a Wits staff or student address, to `support@wits-mss.supportsystem.com`, subject `TWK HPC Query` *(2024 guide)*. Do not email staff directly.

## Login node rules

The login node has **one CPU core**, 30 GB RAM, and dozens of users. It has crashed repeatedly from misuse, and MSS now enforces the following.

- It is for simple file management, simple edits (`vim`, `nano`), and `sbatch`. Anything that is not submitted through Slurm is running on the login node.
- Each user is capped at about 10% of its CPU and RAM.
- VS Code Server, Remote-SSH, tunnels, and any IDE or AI-agent server holding a live connection are not supported. MSS kills these processes without warning.
- Code run directly on the login node is killed without warning.

For an agent this means:

- Batch several read-only commands into one short SSH session. Do not poll in a loop or leave anything running.
- Use `ssh -o BatchMode=yes` so a failed key never turns into password attempts.
- Expect a session to take a minute or more to return when the node is loaded.
- Do not run `du`, `find` over large trees, builds, installs, or tests there. Those are jobs.
- When piping a script into `ssh host 'bash -s'`, any remote command that reads stdin swallows the rest of the script. Give such commands `</dev/null`.

## Partitions

Slurm 25.11.2. Surveyed 10 November 2026; node states change daily.

| Partition | Nodes | CPUs / RAM per node (as Slurm reports) | Time limit | Per-user QoS: running jobs / queued jobs / nodes | State at survey |
|---|---|---|---|---|---|
| `batch` (default) | 100, `mscluster[120-219]` | 6 / 30 GB | 1 day | 13 / 100 / 25 | 83 idle, 1 busy, 16 down |
| `stampede` | 20, `mscluster[22-41]` | 16 / 30 GB | 3 days | 2 / 20 / 5 | all 20 down |
| `bigbatch` | 48, `mscluster[42-89]` | 14 / 124 GB | 3 days | 6 / 48 / 12 | 38 busy, 1 idle, 9 down |
| `biggpu` | 7, `mscluster[106-112]` | mixed: three at 12 / 122 GB, two at 56 / 1 TB, two at 112 / 0.9–1 TB | 3 days | 1 / 6 / 2 (3 nodes max per job) | 3 busy, 2 idle, 2 down |
| `gpuexpress` | 1, `mscluster116` | 12 / 188 GB | 1 hour | 1 / 12 / 1 | down since 30 June 2026 |

GPUs:

- `bigbatch`: one RTX 3090 (24 GB) per node, per MSS notices.
- `biggpu`: the large-memory nodes had 2× Quadro RTX 8000 (48 GB each) in 2024. The three small nodes are *(unverified)*, as is whether some of the seven are privately owned.
- `batch`: has GPUs with less than 24 GB (MSS publishes a CUDA 13.3 PyTorch module for it), model *(unverified)*.
- `stampede`: 2× GTX 1060 (6 GB each) *(2024 guide)*.
- `gpuexpress`: *(unverified)*.

Choosing a partition:

- Default to `batch`. MSS is actively steering smaller jobs there because `bigbatch` is oversubscribed. Its costs: 24-hour limit and **no UPS protection**, so a power dip kills the job.
- Use `bigbatch` only when the job needs its 124 GB RAM or 24 GB of GPU memory.
- Use `biggpu` only for mature code that can really use that hardware. MSS watches for misuse.
- September to November is the busy season; QoS caps are tightened then.

## Slurm

How this cluster is configured, and what follows from it:

- **Whole-node allocation** (`select/linear`). A job gets entire nodes, never a slice. Use all the cores and the GPU of a node you hold, and pack small tasks into one job rather than one node per task.
- **No GRES.** GPUs are not tracked by Slurm. Do not pass `--gres=gpu:N`; you get the node's GPUs by getting the node.
- **Accounting is enforced** (`associations,limits`). No association, no jobs.
- **No memory quota.** The limit is the node's physical RAM.
- `MaxArraySize` is 1001. Backfill scheduling, multifactor priority.
- Jobs interrupted by power, network, or cooling failures are usually requeued and restarted automatically. That happens often (see Reliability), so jobs must checkpoint and be safe to restart from the top.

```bash
sinfo -s                      # partitions and node counts (allocated/idle/other/total)
squeue -u $USER
scontrol show job JOBID
sbatch job.sh
scancel JOBID
sacct -u $USER -X             # past jobs
```

Batch script:

```bash
#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=/home-mscluster/smohan/logs/%x-%j.out

hostname
nvidia-smi
```

The output directory must already exist and be writable. Always set `--time`; the partitions have no default. Avoid interactive `srun`: outages interrupt it.

## Storage

| Path | Size (10 Nov 2026) | Role |
|---|---|---|
| `/home-mscluster/$USER` | 73 TB shared, 56% used | Code, environments, small data. Guideline of about 50 GB per user *(2024 guide)*; MSS emails heavy users and asks them to move data out. |
| `/datasets/$USER` | 377 TB shared, 44% used | Datasets from tens of GB up to a few TB. Long-term. |
| `/scratch/$USER` | per-node local disk | Not present on the login node. On compute nodes *(unverified)*; in 2024 it was node-local, cleared after a few weeks. |
| `/gluster` | — | **Retired.** Not offered since 2026. |

- Both shared filesystems are NFS and visible on the login node. The file server has been overloaded by heavy I/O before; do few large reads and writes rather than many small ones.
- There is no `quota` command. Measure usage inside a job, not with `du` on the login node.
- `/home-mscluster/smohan` still holds 2025 work (`alphafold3`, `ASC25`, `spack`, a self-built `gcc-14.2.0`) of unknown size.
- Large downloads belong on the sideload hosts, which exist to place files and nothing else. Both were down at survey time and MSS has been asked about them. Do not pull large data through the login node in the meantime.

## Software

Ubuntu 26.04 LTS, kernel 7.0, on the login node and the compute nodes sampled.

**CUDA.** MSS prefers the system installs:

```bash
export CUDA_HOME=/usr/local/cuda-13.3        # or /usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

The 2024 guide's `cuda-12.0` no longer exists. NVIDIA driver on the login node is 595.91.07 (RTX 2060 SUPER); compute-node drivers *(unverified)*.

**Modules (Lmod).** `module` is defined only in login shells. If a script reports `module: command not found`, start it with `#!/bin/bash -l` or `source /etc/profile.d/lmod.sh` *(unverified inside a job)*.

| Module | Seen on login node |
|---|---|
| `anaconda/anaconda-2025-12-2` | yes |
| `matlab/R2026a` | yes |
| `R/r-ubuntu-26.04-all-packages-00` (default), `R/r-credit-risk-modelling-00`, `-01` | yes |
| `pytorch/pytorch-cuda13.3-ubuntu26.04-00` (for `batch`, `bigbatch`, `biggpu`) | no; announced by MSS, likely compute nodes only |
| `pytorch/pytorch-cuda12.9-ubuntu24.04-00` | no; announced by MSS |

The Anaconda and R modules are Apptainer images under `/usr/local/containers/`.

**On the login node's PATH:** Python 3.14.4, pip 25.1, gcc/g++/gfortran 15.2, cmake 4.2.3, make, git 2.53, Apptainer 1.4.5 (`singularity` is an alias), Java 25, tmux, screen, rsync, wget, curl, htop.

**Not available:** conda/mamba outside the module, uv, MPI (`mpirun`, `mpicc`), Docker, git-lfs, Julia, node. Install what you need into home, or use an Apptainer image for stacks that are hard to build. Request missing system software through the help desk.

## Reliability

The notices record power failures, cooling failures, network drops, and node crashes roughly monthly, including a complete power failure on 6 November 2026 and overheating that keeps `biggpu` and `stampede` at reduced capacity. Plan for it:

- Checkpoint regularly and resume from the latest checkpoint on start.
- Make jobs idempotent, since Slurm requeues and reruns them.
- Write a progress line every so often so a stalled job is distinguishable from a slow one.
- Verify results from jobs that ran through a known disruption.
- Keep the only copy of anything important off `/scratch`.

## Acknowledgement

> Computations were performed using High Performance Computing infrastructure provided by the Mathematical Sciences Support unit at the University of the Witwatersrand, Johannesburg.
