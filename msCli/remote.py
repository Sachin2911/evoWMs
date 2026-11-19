"""One short SSH session per command.

Every remote action is a bash script piped to `bash -s` on the login node. The
script is wrapped in a function that is called with stdin closed, so no remote
command can swallow the rest of the script. Values are embedded with
shlex.quote and are never interpreted by a shell on the way.
"""

import shlex
import subprocess
import time

import config

BEGIN = "@@MS-BEGIN@@"

# Tests replace this with ["bash", "-s"] to run the "remote" side locally.
TRANSPORT = None


class MsError(Exception):
    """A failure to report to the user as one line."""


def q(value):
    return shlex.quote(str(value))


_PRELUDE = r"""
MS_USER=${USER:-$(id -un)}
MS_R1=$(realpath -m -- "$MS_HOME")
MS_R2=$(realpath -m -- "$MS_DATA")

ms_die() { echo "ms: $*" >&2; exit 1; }

# Expand ~ and relative paths against the cluster home.
ms_abs() {
  local p=$1
  case "$p" in
    "~") p=$MS_HOME ;;
    "~/"*) p=$MS_HOME/${p#"~/"} ;;
    /*) ;;
    *) p=$MS_HOME/$p ;;
  esac
  printf '%s' "$p"
}

# True if the already-resolved path is one of the allowed folders or inside one.
ms_inside() {
  case "$1/" in "$MS_R1"/*|"$MS_R2"/*) return 0 ;; esac
  return 1
}

# Fully resolve a path (.. and symlinks) and require it to be allowed.
ms_resolve() {
  local r
  r=$(realpath -m -- "$(ms_abs "$1")") || ms_die "cannot resolve $1"
  ms_inside "$r" || ms_die "refused: $1 resolves to $r, which is outside the allowed folders"
  printf '%s' "$r"
}

# Resolve the parent and keep the last component, so that a symlink is acted on
# itself and never followed. Used for things about to be deleted or moved.
ms_resolve_entry() {
  local a b d r
  a=$(ms_abs "$1")
  while [ "${a%/}" != "$a" ] && [ -n "${a%/}" ]; do a=${a%/}; done
  b=${a##*/}
  d=${a%/*}
  case "$b" in
    ""|.|..) r=$(realpath -m -- "$a") || ms_die "cannot resolve $1" ;;
    *) r=$(realpath -m -- "${d:-/}") || ms_die "cannot resolve $1"; r=${r%/}/$b ;;
  esac
  ms_inside "$r" || ms_die "refused: $1 resolves to $r, which is outside the allowed folders"
  if [ "$r" = "$MS_R1" ] || [ "$r" = "$MS_R2" ]; then
    ms_die "refused: $r is an allowed folder itself; only things inside it can be deleted or moved"
  fi
  ms_not_keys "$r" || exit 1
  printf '%s' "$r"
}

# The login keys live inside the home folder. Losing them locks the account out.
ms_not_keys() {
  case "$1/" in
    "$MS_R1"/.ssh/*) echo "ms: refused: $1 is part of ~/.ssh, which holds the login keys" >&2; return 1 ;;
  esac
  return 0
}

ms_same_root() {
  case "$1/" in
    "$MS_R1"/*) case "$2/" in "$MS_R1"/*) return 0 ;; esac ;;
    "$MS_R2"/*) case "$2/" in "$MS_R2"/*) return 0 ;; esac ;;
  esac
  return 1
}

# Run directory of a job id: <runs>/<name>-<id>
ms_rundir() {
  local d
  for d in "$MS_RUNS"/*-"$1"; do
    [ -d "$d" ] && { printf '%s' "$d"; return 0; }
  done
  return 1
}

# Refuse to queue past the per-user cap. $1=partition $2=new jobs $3=cap
ms_queue_check() {
  local cur
  cur=$(squeue -h -u "$MS_USER" -p "$1" -r -o %i 2>/dev/null | wc -l)
  if [ $((cur + $2)) -gt "$3" ]; then
    ms_die "refused: $cur jobs already queued on $1; $2 more would pass the per-user cap of $3"
  fi
}

# Submit a script held, so its run directory can be created before it starts.
# $1=name $2=script file, rest=extra sbatch options. Sets MS_JID and MS_DIR.
ms_held_submit() {
  local name=$1 file=$2 out
  shift 2
  out=$(sbatch --hold --parsable "$@" "$file" 2>"$file.err") || {
    echo "ms: sbatch failed: $(tr '\n' ' ' < "$file.err")" >&2
    rm -f "$file" "$file.err"
    return 1
  }
  rm -f "$file.err"
  MS_JID=${out%%;*}
  case "$MS_JID" in
    ''|*[!0-9]*) echo "ms: unexpected sbatch output: $out" >&2; return 1 ;;
  esac
  MS_DIR=$MS_RUNS/$name-$MS_JID
}

ms_release() {
  scontrol release "$1" >/dev/null 2>&1 || {
    echo "ms: job $1 was submitted but could not be released; run: scontrol release $1" >&2
    return 1
  }
}

# Run one fixed command as a one-node job. $1=label, rest=command. Prints the job id.
ms_file_job() {
  local label=$1 tmp
  shift
  ms_queue_check batch 1 "$MS_BATCH_CAP"
  mkdir -p "$MS_RUNS" || return 1
  tmp=$(mktemp "$MS_RUNS/.ms-tmp.XXXXXX") || return 1
  {
    echo '#!/bin/bash'
    echo "#SBATCH --job-name=ms-$label"
    echo '#SBATCH --partition=batch'
    echo '#SBATCH --nodes=1'
    echo '#SBATCH --time=04:00:00'
    echo "#SBATCH --output=$MS_RUNS/ms-$label-%j/slurm.out"
    echo 'echo "ms: start $(date -Is) host=$(hostname)"'
    printf '%q ' "$@"
    echo
    echo 'rc=$?'
    echo 'echo "ms: exit $rc $(date -Is)"'
    echo 'exit $rc'
  } > "$tmp"
  ms_held_submit "ms-$label" "$tmp" || return 1
  mkdir -p "$MS_DIR" && mv "$tmp" "$MS_DIR/job.sbatch"
  ms_release "$MS_JID" || return 1
  echo "$MS_JID"
}
"""


def prelude():
    head = "\n".join(
        [
            f"MS_HOME={q(config.HOME)}",
            f"MS_DATA={q(config.DATASETS)}",
            f"MS_RUNS={q(config.RUNS)}",
            f"MS_BATCH_CAP={config.PARTITIONS['batch']['max_submit']}",
        ]
    )
    return head + _PRELUDE


def _command():
    if TRANSPORT is not None:
        return list(TRANSPORT)
    return [
        "ssh",
        "-T",
        "-o", "BatchMode=yes",  # a failed key must never become password attempts
        "-o", "ConnectTimeout=30",
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=4",
        config.HOST,
        "bash -s",
    ]


def run(body, timeout=None):
    """Run `body` (bash) on the login node in one session. Returns its stdout."""
    timeout = timeout or config.SSH_TIMEOUT
    script = "\n".join(
        [prelude(), "ms_main() {", body, "}", f"echo {BEGIN}", "ms_main </dev/null", ""]
    )
    for attempt in (1, 2):
        try:
            proc = subprocess.run(
                _command(), input=script, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            raise MsError(
                f"no answer from {config.HOST} after {timeout}s; the login node may be loaded. "
                "If this was a submit, check `ms status` before repeating it."
            )
        started = BEGIN in proc.stdout
        if proc.returncode == 255 and not started:
            err = proc.stderr.strip().splitlines()[-1:] or ["no error text"]
            if "ermission denied" in proc.stderr:
                raise MsError(
                    f"authentication to {config.HOST} failed. Not retrying: the login node "
                    "bans an address after a few failed attempts."
                )
            if attempt == 1:
                time.sleep(10)
                continue
            raise MsError(f"cannot reach {config.HOST}: {err[0]}")
        break
    out = proc.stdout.split(BEGIN + "\n", 1)[1] if started else ""
    if proc.returncode != 0:
        lines = [ln for ln in proc.stderr.strip().splitlines() if ln.strip()]
        ours = [ln[4:] for ln in lines if ln.startswith("ms: ")]
        message = ours[-1] if ours else (lines[-1] if lines else f"remote exit {proc.returncode}")
        raise MsError(message)
    return out
