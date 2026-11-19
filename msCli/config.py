"""Fixed facts about mscluster. These are hardcoded on purpose, not configurable."""

HOST = "msCluster"  # alias in ~/.ssh/config

# The only two folders the tool may touch on the cluster.
HOME = "/home-mscluster/smohan"
DATASETS = "/datasets/smohan"

# Every run gets a directory here: <RUNS>/<job-name>-<jobid>/
RUNS = HOME + "/runs"

# The only partitions the tool submits to, with the per-user limits MSS enforces.
PARTITIONS = {
    "batch": {"max_seconds": 1 * 86400, "max_running": 13, "max_submit": 100},
    "bigbatch": {"max_seconds": 3 * 86400, "max_running": 6, "max_submit": 48},
}
DEFAULT_PARTITION = "batch"
DEFAULT_TIME = "01:00:00"

# A directory with more entries than this is deleted by a job, not on the login node.
BIG_TREE_ENTRIES = 2000

LS_LIMIT = 200
FETCH_MAX_MB = 50

SSH_TIMEOUT = 180  # seconds; the login node can take over a minute when loaded
