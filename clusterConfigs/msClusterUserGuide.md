# mscluster operational reference

mscluster is the Wits Mathematical Sciences HPC cluster, run by Mathematical Sciences Support (MSS). It is for supervised research or coursework HPC only: not general storage, personal computing, or non-HPC work.

**Sources.** The cluster was rebuilt from scratch in mid-2026, so the 2024 community guide PDF is out of date on partitions, hosts, storage, and software. It is no longer kept in this repo; MSS still links it at the bottom of `/etc/motd`. This file is based on a live survey of the cluster on 10 November 2026 plus the login notices in `/etc/motd`, and it wins wherever it disagrees with the PDF. Policy that could not be checked live is marked *(2024 guide)*. Things nobody has verified yet are marked *(unverified)*.

`/etc/motd` is MSS's running changelog (outages, policy changes, new modules). Read the tail of it when something behaves unexpectedly.

## Status of our account (10 November 2026)

- `smohan` can log in, and home and `/datasets/smohan` are writable.
- **Job submission works.** MSS added the Slurm association (account `general`, QoS `normal`) on 10 November 2026. Verified the same day with an `sbatch` job on `batch` (`mscluster219`) and an `srun` job on `bigbatch` (`mscluster44`); both completed and could write to home.
- `smohan` is still not in the `mscluster-users` group. Nothing has been found that this blocks.
- If submission ever fails with `Invalid account or account/partition combination specified`, the association is gone: check with `sacctmgr show user $USER withassoc` and open a ticket.
- Compute-node facts below come from probe jobs run on 10 November 2026 on 11 nodes: seven in `batch`, one in `bigbatch`, three in `biggpu`. Nodes that were busy or down could not be probed; the node map says which.
- The probe scripts are in `~/runs/probe/` on the cluster (`hw.sh` prints one line of hardware facts per node; `mod.sh` inspects the modules). Rerun them to refresh this file.

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

### Node map: what each node has

Slurm does not track GPUs, so the node you land on decides the GPU you get. Request specific nodes with `--nodelist` (`-w`) or avoid them with `--exclude` (`-x`).

| Nodes | Partition | CPU | Cores / threads | RAM | GPU | Driver | Local disks | Checked |
|---|---|---|---|---|---|---|---|---|
| `mscluster[120-219]` | `batch` | Intel Core i5-14400 | 6 / 12 | 30 GB | 1× GeForce RTX 3060, 12 GB | 595.71.05 | 466 GB SATA SSD | 7 of 100 probed (130, 156, 172, 186, 201, 215, 219), all identical; Slurm reports the same shape for all 100 |
| `mscluster[42-89]` | `bigbatch` | Intel Core i9-10940X | 14 / 28 | 123 GB | 1× GeForce RTX 3090, 24 GB | 595.91.07 | not recorded | 1 of 48 probed (44); MSS notices also say RTX 3090 |
| `mscluster108` | `biggpu` | 2× Xeon Platinum 8280L | 56 / 112 | 1002 GB | 2× Quadro RTX 8000, 48 GB each | 595.91.07 | 7.3 TB HDD, 477 GB NVMe, 954 GB NVMe | probed |
| `mscluster107` | `biggpu` | *(unverified)* | Slurm: 112 CPUs | 900 GB | *(unverified)* | | | busy, not probed. Same shape as 108, so probably the same hardware |
| `mscluster[106,109]` | `biggpu` | *(unverified)* | Slurm: 56 CPUs | 1010 GB | *(unverified)*; 2× Quadro RTX 8000 in the 2024 guide | | | down |
| `mscluster[111-112]` | `biggpu` | AMD Ryzen 9 9900X3D | 12 / 24 | 121 GB | 1× RTX PRO 6000 Blackwell, 96 GB | 595.91.07 | 256 GB SATA SSD, 1 TB NVMe | both probed, identical |
| `mscluster110` | `biggpu` | *(unverified)* | Slurm: 12 CPUs | 122 GB | *(unverified)* | | | busy, not probed. Same shape as 111 and 112, so probably the same hardware |
| `mscluster[22-41]` | `stampede` | *(unverified)* | Slurm: 16 CPUs | 30 GB | 2× GTX 1060, 6 GB each *(2024 guide)* | | | all 20 down |
| `mscluster116` | `gpuexpress` | *(unverified)* | Slurm: 12 CPUs | 188 GB | *(unverified)* | | | down since 30 June 2026 |

Notes on the map:

- GPU memory per job, smallest to largest: 12 GB (`batch`), 24 GB (`bigbatch`), 2× 48 GB or 1× 96 GB (`biggpu`).
- `biggpu` is two different kinds of machine under one name. `-w mscluster111` (or 112) gets a single 96 GB GPU with a small CPU and 121 GB RAM; `-w mscluster108` gets two 48 GB GPUs with 112 threads and 1 TB RAM. Without `-w` you get whichever is free.
- Slurm counts physical cores on `batch`, `bigbatch`, and the Ryzen `biggpu` nodes (so `nproc` shows twice `SLURM_CPUS_ON_NODE`), but counts all 112 threads on `mscluster108`.
- Every probed node runs Ubuntu 26.04 with kernel 7.0, has no `/scratch`, and has a RAM-backed `/tmp` sized at half its RAM.
- When listing free nodes, filter on the state being exactly `idle`. `sinfo -t idle` also returns drained nodes, which cannot take jobs.

Choosing a partition:

- Default to `batch`. MSS is actively steering smaller jobs there because `bigbatch` is oversubscribed. Its costs: 24-hour limit and **no UPS protection**, so a power dip kills the job.
- Use `bigbatch` only when the job needs more than `batch`'s 30 GB RAM or 12 GB of GPU memory.
- Use `biggpu` only for mature code that can really use that hardware (more than 24 GB of GPU memory, or hundreds of GB of RAM). MSS watches for misuse, and the cap is one running job per user.
- September to November is the busy season; QoS caps are tightened then.

## Slurm

How this cluster is configured, and what follows from it:

- **Whole-node allocation** (`select/linear`). A job gets entire nodes, never a slice. Use all the cores and the GPU of a node you hold, and pack small tasks into one job rather than one node per task.
- **No GRES.** GPUs are not tracked by Slurm. Do not pass `--gres=gpu:N`; you get the node's GPUs by getting the node. `CUDA_VISIBLE_DEVICES` is not set inside a job.
- On most nodes Slurm's CPU count is physical cores: the job can see twice that many hyperthreads (`nproc`), while `SLURM_CPUS_ON_NODE` reports the core count. See the node map for the exception.
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

source /etc/profile.d/lmod.sh        # `module` is not defined in a job otherwise
hostname
nvidia-smi
```

The output directory must already exist and be writable. Always set `--time`; the partitions have no default. Avoid interactive `srun`: outages interrupt it.

## Storage

| Path | Size (10 Nov 2026) | Role |
|---|---|---|
| `/home-mscluster/$USER` | 73 TB shared, 56% used | Code, environments, small data. Guideline of about 50 GB per user *(2024 guide)*; MSS emails heavy users and asks them to move data out. |
| `/datasets/$USER` | 377 TB shared, 44% used | Datasets from tens of GB up to a few TB. Long-term. |
| `/scratch` | — | **Does not exist**, on the login node or on any of the 11 compute nodes probed. |
| `/tmp` on a compute node | half the node's RAM: 16 GB on `batch`, 62 GB on `bigbatch`, 61 GB or 502 GB on `biggpu` | RAM-backed (`tmpfs`), so files there use the job's memory and vanish with the node. |
| Local disk on a compute node | 466 GB SSD on `batch`; SSD, NVMe, and HDD on `biggpu` (see node map) | The disks exist and hold the node's own system. No user-writable path on them has been identified; `/var/tmp` and similar were not tested. Ask MSS before relying on one. |
| `/gluster` | — | **Retired.** Not offered since 2026. |

- Both shared filesystems are NFS, visible on the login node and mounted read-write on compute nodes. The file server has been overloaded by heavy I/O before; do few large reads and writes rather than many small ones.
- There is no `quota` command. Measure usage inside a job, not with `du` on the login node.
- `/home-mscluster/smohan` still holds 2025 work (`alphafold3`, `ASC25`, `spack`, a self-built `gcc-14.2.0`) of unknown size.
- Large downloads belong on the sideload hosts, which exist to place files and nothing else. Both were still down on 10 November 2026 after the account was fixed, and MSS has been asked about them. Do not pull large data through the login node.
- Compute nodes have outbound internet (PyPI and GitHub both answered from `batch` and `bigbatch`), so a job can download its own data and install its own environment.

## Software

Ubuntu 26.04 LTS, kernel 7.0, on the login node and the compute nodes sampled.

**CUDA.** MSS prefers the system installs:

```bash
export CUDA_HOME=/usr/local/cuda-13.3        # or /usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

The 2024 guide's `cuda-12.0` no longer exists. Both CUDA directories are present on the compute nodes probed. NVIDIA driver is 595.91.07 on the login node (RTX 2060 SUPER), `bigbatch`, and `biggpu`, and 595.71.05 on `batch`.

**Modules (Lmod).** `module` is **not defined inside a batch job** (confirmed on `batch` and `bigbatch`), nor in any non-login shell. Put this at the top of every job script before the first `module load`:

```bash
source /etc/profile.d/lmod.sh
```

| Module | Login node | `batch` node | `bigbatch` node |
|---|---|---|---|
| `anaconda/anaconda-2025-12-2` | yes | yes | yes |
| `matlab/R2026a` | yes | yes | yes |
| `R/r-ubuntu-26.04-all-packages-00` | yes | yes | yes |
| `R/r-credit-risk-modelling-00`, `-01` | yes | no | yes |
| `pytorch/pytorch-cuda13.3-ubuntu26.04-00` | no | yes | yes |
| `pytorch/pytorch-cuda12.9-ubuntu24.04-00` | no | yes | yes |
| `comfyui/comfyui-0.39.0-cu128-00` | no | yes | yes |
| `gadget/gadget4-generic-network-00` | no | no | yes |

The module list differs per node, so check with a job on the partition you intend to use.

**How the modules work.** Each one is an Apptainer image under `/usr/local/containers/`. Loading it does not change `PATH`; it defines shell functions named `python`, `python3`, `pip`, `pip3`, `conda`, and `jupyter` that run inside the image (`apptainer exec`, with `--nv` for GPU access in the PyTorch ones), and points conda's environments and package cache at `~/.conda`. Two consequences:

- Shell functions exist only in the shell that loaded the module. They are not seen by a script you call, by `srun`, `xargs`, `env`, or `nohup`. In those places plain `python3` is the node's own Python 3.14 with no PyTorch.
- The form that works everywhere is to call the image directly. The module sets `PYTORCH_SIF` (or `ANACONDA_SIF`) to its path:

```bash
source /etc/profile.d/lmod.sh
module load pytorch/pytorch-cuda13.3-ubuntu26.04-00
apptainer exec --nv "$PYTORCH_SIF" python3 train.py
```

The direct `apptainer exec --nv` form was run on a `batch` node and saw the RTX 3060. The shell-function route is described from the module file and has not been exercised.

**What `pytorch/pytorch-cuda13.3-ubuntu26.04-00` contains** (7.2 GB image, checked in a job on `batch`): Python 3.11.15, PyTorch 2.12.1 built for CUDA 13.2, torchvision 0.27.1, NumPy 2.4.4, `wandb` 0.28.0, JAX 0.10.2. Not included: `gymnasium`, `lightning`. Extra packages need a conda environment or `pip install --user` on top of it. The CUDA 12.9 image (11 GB) and the Anaconda image (1.7 GB) have not been opened.

**On compute nodes:** Python 3.14.4, gcc 15, git, and Apptainer were all present on both nodes probed.

**On the login node's PATH:** Python 3.14.4, pip 25.1, gcc/g++/gfortran 15.2, cmake 4.2.3, make, git 2.53, Apptainer 1.4.5 (`singularity` is an alias), Java 25, tmux, screen, rsync, wget, curl, htop.

**Not available:** conda/mamba outside the module, uv, MPI (`mpirun`, `mpicc`), Docker, git-lfs, Julia, node. Install what you need into home, or use an Apptainer image for stacks that are hard to build. Request missing system software through the help desk.

## Reliability

The notices record power failures, cooling failures, network drops, and node crashes roughly monthly, including a complete power failure on 6 November 2026 and overheating that keeps `biggpu` and `stampede` at reduced capacity. Plan for it:

- Checkpoint regularly and resume from the latest checkpoint on start.
- Make jobs idempotent, since Slurm requeues and reruns them.
- Write a progress line every so often so a stalled job is distinguishable from a slow one.
- Verify results from jobs that ran through a known disruption.
- Never keep the only copy of anything in a node's `/tmp`. Checkpoints go to home or `/datasets`.

## Acknowledgement

> Computations were performed using High Performance Computing infrastructure provided by the Mathematical Sciences Support unit at the University of the Witwatersrand, Johannesburg.
