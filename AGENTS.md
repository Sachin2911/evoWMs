# Agent notes

Running things on the cluster: use `msCli/ms`. It submits, watches, and collects jobs in one short command each and already follows the cluster's rules. `msCli/ms --help` lists the commands and `msCli/README.md` explains them. Do not write `sbatch` scripts or open your own SSH sessions unless `ms` cannot do the job.

- Start a wait with `msCli/ms wait JOBID` in the background and act when it returns; do not poll `status` in a loop.
- `ms rm` has no undo. Only delete what the task calls for.

Cluster background: read `clusterConfigs/msClusterUserGuide.md` when `ms` is not enough, or before choosing a partition or placing data. It is the operational reference for mscluster (login hosts, login-node rules, Slurm, storage, software), based on a live survey of the rebuilt cluster on 10 November 2026. It replaces the 2024 community guide PDF, which is out of date and no longer kept in this repo.
