# mscluster operational reference

Compacted from the Mathematical Sciences HPC Cluster community guidelines (University of the Witwatersrand, 8 February 2024). Facts an agent needs to use the cluster.

mscluster is the Wits Mathematical Sciences HPC cluster, run by Mathematical Sciences Support (MSS). It is for supervised research or coursework HPC only: not general storage, personal computing, or non-HPC work. Jobs this cluster cannot handle belong on the national CHPC system.

## Constraints

- Load shedding in Braamfontein affects access, availability, and performance. The UPS and generator cope with about one event a day; more than that causes outages and hardware damage. Prefer `sbatch`. Interactive `srun` is a poor fit because load shedding interrupts it.
- Do not run CPU-, memory-, disk-, or network-heavy work on a login node. Work that is not `sbatch` or `srun` is running on the login node.
- Do not flood the queue. QoS (reservations, quotas, `MaxJobs`, `MaxSubmitJobs`, per user, partition, or research group) is applied when needed. Default posture is no QoS.
- Scale only as far as the code needs: `stampede`, then `bigbatch`, then `biggpu`. `biggpu` is for mature, debugged code (scarce and expensive to run).
- Jobs must exit cleanly. Leftover processes on worker nodes are the user's to clean up.
- The scheduler does not parallelise serial code. Use MPI, TensorFlow, PyTorch, or another parallel library.
- Correctness-test on a desktop first. Login nodes are not a development machine.
- Write a short progress line every few hours. Check CPU, RAM, GPU util, GPU RAM, disk IO, and network IO on the online monitoring page. Heavy utilisation is not the same as a useful result.
- September–November is historically full. Start work earlier in the year.
- Non-trivial pre-processing is a submitted job, including while sideloading data.
- Jobs are not given a memory quota. The limit is the node's physical RAM.

## Access

| Role | Host | Notes |
|---|---|---|
| Primary login | `146.141.21.100` | Lands in `/home-mscluster/$USER`. Use this. |
| Backup login | `146.141.21.101` | Only if the primary is unusable. Extra restrictions. |
| Sideload `/datasets` | `10.100.14.250` | Place files only. |
| Sideload `/gluster` | `10.100.14.2` | Place files only. |
| Sideload home | `10.100.14.252` | Rare. Ask MSS first. |

Help: one ticket per issue, from a Wits staff or student address, to `support@wits-mss.supportsystem.com`, subject `TWK HPC Query`. Do not email staff directly. Multi-terabyte USB transfer is arranged through that desk.

## Partitions

| Partition | Nodes | CPU | GPU | RAM | Limits | Use |
|---|---|---|---|---|---|---|
| `stampede` | 40 | 2× Xeon E5-2680 | 2× GTX 1060 (6 GB each, 12 GB/node) | 32 GB | MaxTime=4320, MaxNodes=20 | Default. General jobs and InfiniBand/MPI. |
| `bigbatch` | 48 | 1× Core i9-10940X (14 cores) | 1× RTX 3090 (24 GB) | 128 GB | MaxTime=4320, MaxNodes=48 | Larger GPU and RAM. 10 GbE is planned, not installed. |
| `biggpu` | 4 | 2× Xeon Platinum 8280L (28 cores each, 56/node) | 2× Quadro RTX 8000 (48 GB each, 96 GB/node) | 1 TB | MaxTime=4320, MaxNodes=4 | Mature code that needs very large GPU and system RAM. 10 GbE is planned, not installed. |

`MaxTime=4320` is the Slurm limit in minutes (3 days).

stampede InfiniBand is split: `mscluster[2-21]` on one switch, `mscluster[22-42]` on another. Pin MPI jobs to one island so traffic stays on a single switch.

## Slurm

```bash
sinfo
squeue
scontrol show job JOBID
sbatch test.sh
scancel JOBID
scancel --user=$USER
```

Batch script. Replace the username. The output path must be writable.

```bash
#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=/home-mscluster/$USER/result.txt
#SBATCH --partition=bigbatch
sleep 60
/bin/hostname
```

Placement. The code has to use the nodes it asks for.

```bash
#SBATCH -w mscluster61
#SBATCH --nodelist=mscluster[2-4]
#SBATCH --nodes=2
```

`srun` works but should be avoided:

```bash
srun -N6 -p bigbatch -l /bin/hostname
srun -N2 -p biggpu -l cat /proc/cpuinfo | grep model
srun -N4 -p stampede -l /usr/bin/uptime
```

## Storage

| Path | Scope | Role | Retention |
|---|---|---|---|
| `/home-mscluster/$USER` | cluster-wide | Code, environment, and up to 50 GB of data. | Persistent working space. |
| `/scratch/$USER` | local disk on each node | High disk IO without flooding the network. | A few weeks. Cleared with little warning if it affects others. Delete it when the job is done. |
| `/gluster/$USER` | cluster-wide GlusterFS | Network scratch when per-node scratch is awkward to manage. | A few months. Same clearing rule. |
| `/datasets/$USER` | cluster-wide | Datasets from about 50 GB up to a few TB. | Long-term dataset store. |

`cssh` may stage or delete `/scratch` across nodes. Do not launch compute through it.

Large copies use the sideload hosts so other users are not affected. Do nothing there except put the files in place:

```bash
ssh $USER@10.100.14.250
cd /datasets/$USER
wget http://URL
```

Same pattern for Gluster: `10.100.14.2` and `/gluster/$USER`. Home sideload (`10.100.14.252`) only after MSS agrees. Multi-TB sets can also move over dedicated NICs and disks, or USB by arrangement. Moving large data is best done inside TW Kambule Laboratories.

## Software

Install into the home directory when you can. A system stack exists; personal environments are the expected path.

- **Anaconda:** install your own under home. The system install will not match every project.
- **TensorFlow, PyTorch, other ML libraries:** install into that environment.
- **Apptainer** (formerly Singularity): use it for shared group workflows and hard-to-build stacks.
- **CUDA:** prefer 12.0. Other versions are under `/usr/local/`. If unsure, use the newest installed. Open a help-desk ticket for version conflicts.

```bash
export PATH=/usr/local/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.0
```

- **R:** compile it and install packages into home.
- **MATLAB, Mathematica, and other occasional tools:** look in `/usr/local/` and add chosen binaries to `PATH` in `.bashrc`. Request missing software via the help desk.
- Documented client is Ubuntu. Windows and macOS are untested.

## Acknowledgement

> Computations were performed using High Performance Computing infrastructure provided by the Mathematical Sciences Support unit at the University of the Witwatersrand, Johannesburg.
