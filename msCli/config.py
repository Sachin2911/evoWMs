"""Fixed facts about mscluster. These are hardcoded on purpose, not configurable."""

HOST = "msCluster"  # alias in ~/.ssh/config

# The only two folders the tool may touch on the cluster.
HOME = "/home-mscluster/smohan"
DATASETS = "/datasets/smohan"

# Every run gets a directory here: <RUNS>/<job-name>-<jobid>/
RUNS = HOME + "/runs"

# Hugging Face token, readable only by the account. Jobs are pointed at it; it is never copied.
HF_TOKEN_FILE = HOME + "/credentials/hf_token"

# The only partitions the tool submits to, with the per-user limits MSS enforces.
PARTITIONS = {
    "batch": {"max_seconds": 1 * 86400, "max_running": 13, "max_submit": 100},
    "bigbatch": {"max_seconds": 3 * 86400, "max_running": 6, "max_submit": 48},
}
DEFAULT_PARTITION = "batch"

# Nodes no job is sent to. Their GPUs cannot draw the simulator's cameras ("Cannot initialize
# a EGL device display"), and because a failed job frees its node at once, the next job
# lands on the same node and fails too.
EXCLUDED_NODES = {
    "batch": ["mscluster129"],
    "bigbatch": ["mscluster44", "mscluster48", "mscluster65", "mscluster71", "mscluster75", "mscluster76", "mscluster82"],
}
DEFAULT_TIME = "01:00:00"

# A directory with more entries than this is deleted by a job, not on the login node.
BIG_TREE_ENTRIES = 2000

LS_LIMIT = 200
FETCH_MAX_MB = 50

SSH_TIMEOUT = 180  # seconds; the login node can take over a minute when loaded
