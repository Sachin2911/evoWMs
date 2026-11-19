"""Status file for a run. Copy or import this into the research code.

A job calls write_status() as it goes. `ms status` shows the latest values in one
line, and a restarted job calls read_status() to find where to resume.

    from ms_status import run_dir, read_status, write_status

    last = read_status()                      # None on a fresh start
    start = last["step"] if last else 0
    for step in range(start, total):
        ...
        write_status(step=step, of=total, metric=loss, checkpoint="ckpt.pt")
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def run_dir():
    """The run's own folder: MS_RUN_DIR on the cluster, outputs/local on the laptop."""
    path = Path(os.environ.get("MS_RUN_DIR", "outputs/local"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_status(**fields):
    """Replace status.json with these fields plus the time of writing."""
    fields["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    target = run_dir() / "status.json"
    scratch = target.with_suffix(".json.tmp")
    scratch.write_text(json.dumps(fields) + "\n")
    os.replace(scratch, target)  # a reader never sees a half-written file


def read_status():
    """The last status written in this run folder, or None."""
    try:
        return json.loads((run_dir() / "status.json").read_text())
    except (OSError, ValueError):
        return None
