# Agent notes

Running things on the cluster: use `msCli/ms`. It submits, watches, and collects jobs in one short command each and already follows the cluster's rules. `msCli/ms --help` lists the commands and `msCli/README.md` explains them. Do not write `sbatch` scripts or open your own SSH sessions unless `ms` cannot do the job.

- Do all research compute on the cluster: image builds, package installs, simulator and model runs, experiments. Do not run them on the laptop, which is only for editing code and calling `ms`.
- Start a wait with `msCli/ms wait JOBID` in the background and act when it returns; do not poll `status` in a loop.
- `ms rm` has no undo. Only delete what the task calls for.

Cluster background: read `clusterConfigs/msClusterUserGuide.md` when `ms` is not enough, or before choosing a partition or placing data. It is the operational reference for mscluster (login hosts, login-node rules, Slurm, storage, software), based on a live survey of the rebuilt cluster on 10 November 2026. It replaces the 2024 community guide PDF, which is out of date and no longer kept in this repo.

Research notes live in Notion under [GECCO Research](https://app.notion.com/p/414d313322188095b422d0d86fef077f), not in this repo. Use the Notion connector; skip this if you do not have one.

- Read [Central Questions and Themes](https://app.notion.com/p/41bd3133221880218551f06bf811a87f) for the research question and the six themes.
- Write session and experiment notes to the [Lab NoteBook](https://app.notion.com/p/41bd3133221880208e28dfa64aed9db5): one sub-page per day, titled like `17th November 2026`. Add to that day's page if it already exists.
- For an experiment, record the job ID, the git commit hash, and the result.
- Mark what you write with 🤖. Do not edit or delete anything else in Notion without asking.
