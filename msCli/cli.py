"""The `ms` commands. Each one is a single short SSH session, except `wait`
(one session per check) and `fetch` (one to resolve the path, then rsync)."""

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time

import config
import jobs
import remote
from remote import MsError, q


def remote_path(path):
    """Undo the local shell's expansion of ~, which points at the laptop's home."""
    home = os.path.expanduser("~")
    if path == home:
        return "~"
    if path.startswith(home + "/"):
        return "~/" + path[len(home) + 1:]
    return path


def absolute_remote(path):
    """A cluster path as an absolute one; relative and ~ paths start at the cluster home."""
    path = remote_path(path)
    if path == "~":
        return config.HOME
    if path.startswith("~/"):
        return f"{config.HOME}/{path[2:]}"
    return path if path.startswith("/") else f"{config.HOME}/{path}"


def job_ids(values):
    for value in values:
        if not re.fullmatch(r"[0-9]+", value):
            raise MsError(f"'{value}' is not a job id (digits only)")
    return list(values)


def tagged(out):
    """Lines the remote side printed as TAG|a|b, as {TAG: [a, b]}."""
    return {f[0]: f[1:] for f in (line.split("|") for line in out.splitlines()) if len(f) > 1}


def human_size(n):
    n = float(n)
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024 or unit == "T":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024


def print_table(rows):
    if not rows:
        return
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]) - 1)]
    for row in rows:
        cells = [str(c).ljust(w) for c, w in zip(row[:-1], widths)]
        print("  ".join(cells + [str(row[-1])]).rstrip())


def git(*args):
    proc = subprocess.run(["git", *args], capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else None


def repo_info():
    """The git repository the command is run from, or None."""
    top = git("rev-parse", "--show-toplevel")
    if not top:
        return None
    url = git("remote", "get-url", "origin") or ""
    # The cluster clones over HTTPS, and never receives credentials embedded in a URL.
    ssh_form = re.fullmatch(r"git@([^:]+):(.+)", url)
    if ssh_form:
        url = f"https://{ssh_form.group(1)}/{ssh_form.group(2)}"
    url = re.sub(r"^(https?://)[^/@]+@", r"\1", url)
    name = os.path.basename(url)
    name = name[:-4] if name.endswith(".git") else name
    name = jobs.clean_name(name or os.path.basename(top))
    return {
        "url": url,
        "name": name,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "head": git("rev-parse", "HEAD"),
        "subdir": os.path.relpath(os.getcwd(), top),
        "remote_dir": f"{config.HOME}/{name}",
    }


def cmd_nodes(args):
    parts = ",".join(config.PARTITIONS)
    out = remote.run(
        f"sinfo -h -p {parts} -o 'N|%R|%T|%D'\n"
        f"squeue -h -u \"$MS_USER\" -p {parts} -r -o 'Q|%P|%T' 2>/dev/null\n"
        "true"
    )
    stats = {
        p: {"idle": 0, "busy": 0, "down": 0, "running": 0, "queued": 0}
        for p in config.PARTITIONS
    }
    for line in out.splitlines():
        f = line.split("|")
        if f[0] == "N" and f[1] in stats:
            state, count = f[2], int(f[3])
            # Only an exact "idle" can take a job; "idle*", "drained" and the rest cannot.
            if state == "idle":
                stats[f[1]]["idle"] += count
            elif re.sub(r"[^a-z]", "", state) in ("allocated", "mixed", "completing"):
                stats[f[1]]["busy"] += count
            else:
                stats[f[1]]["down"] += count
        elif f[0] == "Q" and f[1] in stats:
            stats[f[1]]["running" if f[2] == "RUNNING" else "queued"] += 1
    if args.json:
        print(json.dumps(stats))
        return
    rows = []
    for name, s in stats.items():
        limits = config.PARTITIONS[name]
        rows.append((
            name,
            f"{s['idle']} idle",
            f"{s['busy']} busy",
            f"{s['down']} down",
            f"limit {limits['max_seconds'] // 86400}d",
            f"mine: {s['running']} running, {s['queued']} waiting "
            f"(caps {limits['max_running']} running, {limits['max_submit']} in queue)",
        ))
    print_table(rows)


_STATUS_BODY = r"""
fmt='%i|%j|%P|%T|%M|%N|%r'
cols=JobID,JobName,Partition,State,Elapsed,NodeList,ExitCode
sqo=$(squeue -h -u "$MS_USER" -o "$fmt" 2>/dev/null)
if [ -n "$ids" ]; then
  sao=$(sacct -X -n -P -j "$ids" -o "$cols" 2>/dev/null)
else
  sao=$(sacct -X -n -P -S now-1days -o "$cols" 2>/dev/null)
fi
[ -n "$sqo" ] && printf '%s\n' "$sqo" | sed 's/^/SQ|/'
[ -n "$sao" ] && printf '%s\n' "$sao" | sed 's/^/SA|/'
masters=$( { printf '%s\n' "$sqo"; printf '%s\n' "$sao"; printf '%s\n' "$ids" | tr ',' '\n'; } \
  | awk -F'|' 'NF { split($1, a, /[_.]/); print a[1] }' | sort -un )
for id in $masters; do
  d=$(ms_rundir "$id") || continue
  echo "RD|$id|$(realpath -m -- "$d")"
  [ -f "$d/status.json" ] && echo "ST|$id|$(head -c 600 "$d/status.json" | head -n 1)"
  if [ -d "$d/.ms" ]; then
    echo "SW|$id|$(. "$d/.ms/meta"; echo "$n")|$(ls "$d/.ms/done" | tr '\n' ' ')|$(ls "$d/.ms/failed" | tr '\n' ' ')|$(ls "$d/.ms/started" | tr '\n' ' ')"
  fi
done
true
"""


def _status_text(raw):
    try:
        return " ".join(f"{k}={v}" for k, v in json.loads(raw).items())[:200]
    except (ValueError, AttributeError):
        return raw.strip()[:200]


def collect(ids=None):
    """One session: Slurm's view of my jobs plus each run directory's status file.

    Returns one row per job (a sweep is one row), sorted by job id.
    """
    out = remote.run(f"ids={q(','.join(ids or []))}\n" + _STATUS_BODY)
    queue, acct, dirs, status, sweeps = {}, {}, {}, {}, {}
    for line in out.splitlines():
        kind, _, rest = line.partition("|")
        if kind in ("SQ", "SA"):
            f = rest.split("|")
            if len(f) < 7:
                continue
            master = re.split(r"[_.]", f[0])[0]
            (queue if kind == "SQ" else acct).setdefault(master, []).append(f)
        elif kind == "RD":
            jid, path = rest.split("|", 1)
            dirs[jid] = path
        elif kind == "ST":
            jid, raw = rest.split("|", 1)
            status[jid] = raw
        elif kind == "SW":
            jid, n, done, failed, started = rest.split("|")
            sweeps[jid] = (int(n or 0), set(done.split()), set(failed.split()), set(started.split()))

    wanted = set(ids) if ids else set(queue) | set(acct)
    rows = []
    for jid in sorted(wanted, key=int):
        q_rows, a_rows = queue.get(jid, []), acct.get(jid, [])
        active = bool(q_rows)
        first = (q_rows or a_rows or [[jid, "?", "?", "UNKNOWN", "-", "-", ""]])[0]
        row = {
            "id": jid, "name": first[1], "partition": first[2], "state": first[3].split()[0],
            "elapsed": first[4] or "-", "node": first[5] or "-",
            "info": _status_text(status[jid]) if jid in status else "",
            "dir": dirs.get(jid), "active": active, "ok": None, "known": bool(q_rows or a_rows),
        }
        if jid in sweeps:
            n, done, failed, started = sweeps[jid]
            running = started - done - failed
            lost = set()
            if not active:
                lost, running = running, set()  # started, never finished, and nothing is running
            pending = n - len(done) - len(failed) - len(running) - len(lost)
            bad = sorted(failed | lost, key=int)
            row["state"] = "RUNNING" if active else ("COMPLETED" if len(done) == n else "FAILED")
            if active and not any(r[3] == "RUNNING" for r in q_rows):
                row["state"] = "PENDING"
            row["elapsed"] = row["node"] = "-"
            row["ok"] = None if active else len(done) == n
            info = f"sweep {len(done)}/{n} done, {len(running)} running, {pending} waiting, {len(bad)} failed"
            if bad:
                shown = " ".join(str(int(b)) for b in bad[:20])
                info += f" (tasks {shown}{' ...' if len(bad) > 20 else ''})"
            row["info"] = info
            row["sweep"] = {"tasks": n, "done": len(done), "running": len(running),
                            "waiting": pending, "failed": [int(b) for b in bad]}
        elif len(q_rows or a_rows) > 1:
            # A job array that ms did not submit: summarise its elements.
            counts = {}
            for r in q_rows or a_rows:
                state = r[3].split()[0]
                counts[state] = counts.get(state, 0) + 1
            all_good = set(counts) == {"COMPLETED"}
            bad = {k: v for k, v in counts.items() if k != "COMPLETED"}
            row["state"] = ("RUNNING" if active else "COMPLETED" if all_good
                            else max(bad, key=bad.get))
            row["elapsed"] = row["node"] = "-"
            row["info"] = "array: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
            row["ok"] = None if active else all_good
        elif not active and row["known"]:
            row["ok"] = row["state"] == "COMPLETED"
            if not row["ok"] and len(first) > 6 and first[6] not in ("", "0:0"):
                row["info"] = (row["info"] + f" exit={first[6]}").strip()
        rows.append(row)

    # A retried sweep has two job ids pointing at one directory; keep the newer.
    newest = {}
    for row in rows:
        if row.get("sweep") and row["dir"]:
            newest[row["dir"]] = row["id"]
    return [r for r in rows if not (r.get("sweep") and r["dir"] and newest[r["dir"]] != r["id"])
            or (ids and r["id"] in ids)]


def print_rows(rows, as_json=False):
    if as_json:
        print(json.dumps(rows))
        return
    print_table([
        (r["id"], r["name"], r["partition"], r["state"], r["elapsed"], r["node"], r["info"])
        for r in rows
    ])


def cmd_status(args):
    rows = collect(job_ids(args.ids))
    if not rows and not args.json:
        print("no jobs queued, running, or finished in the last day")
        return
    print_rows(rows, args.json)


def cmd_wait(args):
    ids = job_ids(args.ids)
    interval = max(args.interval, 60)
    deadline = time.time() + args.timeout if args.timeout else None
    failures = unknown = 0
    while True:
        rows = None
        try:
            rows = collect(ids)
            failures = 0
        except MsError as exc:
            failures += 1
            if failures >= 5:
                raise MsError(f"gave up after 5 failed checks in a row: {exc}")
        if rows is not None:
            waiting = [r for r in rows if r["active"]]
            missing = [r for r in rows if not r["known"] and not r.get("sweep")]
            # A job can be gone from the queue before accounting has its record.
            unknown = unknown + 1 if missing and not waiting else 0
            if not waiting and (not missing or unknown >= 3):
                break
        if deadline and time.time() >= deadline:
            raise MsError(f"still not finished after {args.timeout}s: {' '.join(ids)}")
        time.sleep(interval)
    print_rows(rows, args.json)
    return 0 if all(r["ok"] for r in rows) else 1


def cmd_logs(args):
    (jid,) = job_ids([args.id])
    lines = max(1, args.lines)
    body = [f"d=$(ms_rundir {jid}) || ms_die \"no run directory for job {jid} under $MS_RUNS\""]
    if args.task is not None:
        body.append(f'f=$d/task-{args.task:04d}/task.log')
    else:
        body += [
            'if [ -d "$d/.ms" ]; then',
            f'  ms_die "job {jid} is a sweep; pass --task N. Failed tasks: '
            '$(ls "$d/.ms/failed" 2>/dev/null | tr \'\\n\' \' \')"',
            "fi",
            "f=$d/slurm.out",
        ]
    body += [
        '[ -f "$f" ] || ms_die "no log yet at $f (the job may not have started)"',
        f'tail -n {lines} "$f"',
    ]
    sys.stdout.write(remote.run("\n".join(body)))


def cmd_ls(args):
    path = remote_path(args.path)
    fmt = r"%y\t%s\t%TY-%Tm-%Td %TH:%TM\t%f\n"
    out = remote.run(f"""
p=$(ms_resolve {q(path)}) || exit 1
[ -e "$p" ] || ms_die "not found: $p"
if [ -d "$p" ]; then
  find "$p" -mindepth 1 -maxdepth 1 -printf '{fmt}' | LC_ALL=C sort -t "$(printf '\\t')" -k4 \\
    | awk -v m={config.LS_LIMIT} 'NR <= m {{ print }} END {{ print "TOTAL\\t" NR }}'
else
  find "$p" -maxdepth 0 -printf '{fmt}'
  printf 'TOTAL\\t1\\n'
fi
""")
    entries, total = [], 0
    for line in out.splitlines():
        f = line.split("\t")
        if f[0] == "TOTAL":
            total = int(f[1])
        elif len(f) == 4:
            kind = {"d": "dir", "l": "link"}.get(f[0], "file")
            entries.append({"name": f[3], "type": kind, "bytes": int(f[1]), "modified": f[2]})
    if args.json:
        print(json.dumps({"total": total, "entries": entries}))
        return
    print_table([
        ("-" if e["type"] == "dir" else human_size(e["bytes"]), e["modified"],
         e["name"] + {"dir": "/", "link": "@"}.get(e["type"], ""))
        for e in entries
    ])
    if total > len(entries):
        print(f"... {total - len(entries)} more ({total} entries in all)")
    elif total == 0:
        print("(empty)")


def cmd_rm(args):
    paths = " ".join(q(remote_path(p)) for p in args.paths)
    out = remote.run(f"""
paths=()
for a in {paths}; do
  p=$(ms_resolve_entry "$a") || exit 1
  [ -e "$p" ] || [ -L "$p" ] || ms_die "not found: $p (nothing was deleted)"
  paths+=("$p")
done
for p in "${{paths[@]}}"; do
  n=0
  if [ -d "$p" ] && [ ! -L "$p" ]; then
    n=$(find "$p" -mindepth 1 2>/dev/null | head -n {config.BIG_TREE_ENTRIES + 1} | wc -l)
  fi
  if [ "$n" -gt {config.BIG_TREE_ENTRIES} ]; then
    jid=$(ms_file_job rm rm -rf -- "$p") || exit 1
    echo "JOB|$jid|$p"
  else
    rm -rf -- "$p" || ms_die "could not remove $p"
    echo "RM|$p"
  fi
done
""")
    for line in out.splitlines():
        f = line.split("|")
        if f[0] == "RM":
            print(f"removed {f[1]}")
        elif f[0] == "JOB":
            print(f"removing {f[2]} in job {f[1]} (large tree; check with: ms status {f[1]})")


# A move between the two folders is a copy and then a delete. Written so that a
# job requeued halfway can simply run again: the copy overwrites, and a source
# that is already gone means the move had finished.
_MOVE_JOB = """
s=$1; t=$2
if [ ! -e "$s" ] && [ ! -L "$s" ]; then
  echo "ms: $s is already gone; nothing left to move"
  exit 0
fi
if [ -d "$s" ] && [ ! -L "$s" ]; then
  mkdir -p -- "$t" && cp -a -- "$s/." "$t/" && rm -rf -- "$s"
else
  cp -a -- "$s" "$t" && rm -f -- "$s"
fi
"""


def cmd_mv(args):
    src, dst = remote_path(args.src), remote_path(args.dst)
    out = remote.run(f"""
s=$(ms_resolve_entry {q(src)}) || exit 1
[ -e "$s" ] || [ -L "$s" ] || ms_die "not found: $s"
d=$(ms_resolve {q(dst)}) || exit 1
if [ -d "$d" ]; then
  t=$d/${{s##*/}}
else
  t=$d
  [ -d "${{t%/*}}" ] || ms_die "destination folder does not exist: ${{t%/*}}"
fi
ms_not_keys "$t" || exit 1
if [ -e "$t" ] || [ -L "$t" ]; then ms_die "destination already exists: $t"; fi
if ms_same_root "$s" "$t"; then
  mv -- "$s" "$t" || ms_die "could not move $s"
  echo "MV|$s|$t"
else
  jid=$(ms_file_job mv bash -c {q(_MOVE_JOB)} ms-move "$s" "$t") || exit 1
  echo "JOB|$jid|$s|$t"
fi
""")
    for line in out.splitlines():
        f = line.split("|")
        if f[0] == "MV":
            print(f"moved {f[1]} -> {f[2]}")
        elif f[0] == "JOB":
            print(f"moving {f[2]} -> {f[3]} in job {f[1]} "
                  f"(between the two folders; check with: ms status {f[1]})")


def cmd_sync(args):
    repo = repo_info()
    if not repo:
        raise MsError("not inside a git repository")
    if not repo["url"]:
        raise MsError("this repository has no 'origin' remote to pull from on the cluster")
    branch = repo["branch"]
    if not branch or branch == "HEAD":
        raise MsError("HEAD is detached; check out a branch before syncing")
    pushed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", "HEAD", f"origin/{branch}"], capture_output=True
    ).returncode == 0
    if not pushed:
        raise MsError(
            f"local commit {repo['head'][:7]} is not on origin/{branch}. Run `git push` first; "
            "the cluster pulls from GitHub."
        )
    dirty = git("status", "--porcelain")
    out = remote.run(f"""
dir=$(ms_resolve {q(repo['remote_dir'])}) || exit 1
export GIT_TERMINAL_PROMPT=0
if [ -d "$dir/.git" ]; then
  git -C "$dir" fetch --quiet origin || ms_die "git fetch failed in $dir"
  git -C "$dir" checkout --quiet {q(branch)} || ms_die "git checkout {branch} failed in $dir"
  git -C "$dir" merge --quiet --ff-only {q('origin/' + branch)} \\
    || ms_die "cannot fast-forward $dir; the cluster copy has changes of its own"
elif [ -e "$dir" ]; then
  ms_die "$dir exists and is not a git clone"
else
  git clone --quiet --branch {q(branch)} {q(repo['url'])} "$dir" || ms_die "git clone failed"
fi
echo "HEAD|$(git -C "$dir" rev-parse HEAD)|$dir"
echo "WAITING|$(squeue -h -u "$MS_USER" -t PD -r -o %i 2>/dev/null | wc -l)"
""", timeout=300)
    info = tagged(out)
    head, path = info["HEAD"]
    print(f"synced {repo['name']} {branch} @ {head[:7]} -> {path}")
    if head != repo["head"]:
        print(f"note: the cluster is at {head[:7]} but local HEAD is {repo['head'][:7]}")
    if dirty:
        print("note: uncommitted local changes were not synced")
    if info["WAITING"] != ["0"]:
        print(f"note: {info['WAITING'][0]} waiting jobs will run this new code when they start")


def cmd_submit(args):
    if args.retry:
        return _retry(args)
    argv = args.command_argv
    if not argv:
        raise MsError("nothing to run; put the command after --, e.g. ms submit -- python train.py")
    partition = args.partition
    limits = config.PARTITIONS[partition]
    time_limit = jobs.check_time(args.time, partition)

    repo = repo_info()
    if args.workdir:
        workdir = absolute_remote(args.workdir)
    elif repo:
        workdir = os.path.normpath(os.path.join(repo["remote_dir"], repo["subdir"]))
    else:
        workdir = config.HOME
    prefix = repo["name"] if repo else "job"
    name = jobs.clean_name(f"{prefix}-{args.name or jobs.default_label(argv)}")
    venv = absolute_remote(args.env) if args.env else None
    common = dict(
        name=name, partition=partition, time=time_limit, workdir=workdir, argv=argv,
        modules=args.module or [], venv=venv,
    )

    tasks = None
    if args.grid and args.array:
        raise MsError("use --array or --grid, not both")
    if args.grid:
        tasks = jobs.expand_grid(args.grid)
        ntasks = len(tasks)
    else:
        ntasks = args.array
    if ntasks is None:
        if args.per_node != 1:
            raise MsError("--per-node only applies to a sweep (--array or --grid)")
        script, new_jobs = jobs.job_script(**common), 1
    else:
        if ntasks < 1 or args.per_node < 1:
            raise MsError("--array and --per-node must be at least 1")
        new_jobs = -(-ntasks // args.per_node)
        if new_jobs > limits["max_submit"]:
            raise MsError(
                f"{ntasks} tasks at {args.per_node} per node is {new_jobs} nodes' worth of jobs; "
                f"the {partition} cap is {limits['max_submit']} in the queue. Raise --per-node."
            )
        script = jobs.sweep_script(ntasks=ntasks, per_node=args.per_node, **common)

    def b64(text):
        return base64.b64encode(text.encode()).decode()

    body = [
        f"ms_queue_check {partition} {new_jobs} {limits['max_submit']}",
        f"w=$(ms_resolve {q(workdir)}) || exit 1",
        '[ -d "$w" ] || ms_die "the working directory $w does not exist on the cluster (run: ms sync)"',
    ]
    if venv:
        body.append(f"v=$(ms_resolve {q(venv)}) || exit 1")
    body += [
        'mkdir -p "$MS_RUNS" || ms_die "cannot create $MS_RUNS"',
        'tmp=$(mktemp "$MS_RUNS/.ms-tmp.XXXXXX") || ms_die "cannot write in $MS_RUNS"',
        f'echo {b64(script)} | base64 -d > "$tmp"',
        f'ms_held_submit {q(name)} "$tmp" || exit 1',
        'mkdir -p "$MS_DIR" || ms_die "cannot create $MS_DIR"',
        'mv "$tmp" "$MS_DIR/job.sbatch"',
        f'echo {b64(" ".join(map(q, argv)))} | base64 -d > "$MS_DIR/command.txt"',
    ]
    if ntasks is not None:
        # What --retry and status need to know about the sweep, as shell assignments.
        meta = (f"n={ntasks} p={args.per_node} cap={limits['max_running']} "
                f"limit={limits['max_submit']} part={partition} name={name}")
        body += [
            'mkdir -p "$MS_DIR/.ms/done" "$MS_DIR/.ms/failed" "$MS_DIR/.ms/started"',
            f'echo {q(meta)} > "$MS_DIR/.ms/meta"',
        ]
    if tasks:
        body.append(f'echo {b64(chr(10).join(tasks) + chr(10))} | base64 -d > "$MS_DIR/tasks.txt"')
    body.append('ms_release "$MS_JID" || exit 1')
    if repo and not args.workdir:
        body.append(f'echo "HEAD|$(git -C {q(repo["remote_dir"])} rev-parse HEAD 2>/dev/null)"')
    body.append('echo "OK|$MS_JID|$MS_DIR"')

    out = remote.run("\n".join(body))
    info = tagged(out)
    jid, path = info["OK"]
    what = f"{partition}  {time_limit}"
    if ntasks is not None:
        what += (f"  sweep of {ntasks} tasks, {args.per_node} per node, {new_jobs} nodes' worth, "
                 f"at most {limits['max_running']} at a time")
    print(f"submitted {jid}  {name}  {what}  {path}")
    cluster_head = info.get("HEAD", [""])[0]
    if repo and cluster_head and cluster_head != repo["head"]:
        print(f"note: the cluster copy is at {cluster_head[:7]}, local HEAD is "
              f"{repo['head'][:7]} (run: ms sync)", file=sys.stderr)


def _retry(args):
    (jid,) = job_ids([args.retry])
    if args.command_argv or args.array or args.grid:
        raise MsError("--retry reruns the original command; give it only the sweep's job id")
    out = remote.run(f"""
d=$(ms_rundir {jid}) || ms_die "no run directory for job {jid}"
d=$(realpath -m -- "$d")
[ -d "$d/.ms" ] || ms_die "job {jid} is not a sweep; submit it again instead"
. "$d/.ms/meta"
for other in $(squeue -h -u "$MS_USER" -o %F 2>/dev/null | sort -u); do
  o=$(ms_rundir "$other") || continue
  [ "$(realpath -m -- "$o")" = "$d" ] && ms_die "sweep {jid} still has jobs in the queue (job $other); wait or cancel first"
done
elements=""; count=0; last=-1; todo=0
i=0
while [ "$i" -lt "$n" ]; do
  printf -v pad '%04d' "$i"
  if [ ! -e "$d/.ms/done/$pad" ]; then
    todo=$((todo + 1))
    rm -f "$d/.ms/failed/$pad" "$d/.ms/started/$pad"
    e=$((i / p))
    if [ "$e" -ne "$last" ]; then elements="$elements,$e"; count=$((count + 1)); last=$e; fi
  fi
  i=$((i + 1))
done
[ "$todo" -gt 0 ] || ms_die "nothing to retry: all $n tasks of sweep {jid} are done"
ms_queue_check "$part" "$count" "$limit"
cp "$d/job.sbatch" "$d/.ms/retry.sbatch"
ms_held_submit "$name" "$d/.ms/retry.sbatch" --array="${{elements#,}}%$cap" || exit 1
ln -s "$d" "$MS_DIR" || ms_die "cannot link $MS_DIR"
ms_release "$MS_JID" || exit 1
echo "OK|$MS_JID|$todo|$count|$d"
""")
    new_id, todo, count, path = tagged(out)["OK"]
    print(f"submitted {new_id}  retry of sweep {jid}: {todo} unfinished tasks on {count} nodes' worth  {path}")


def cmd_cancel(args):
    ids = job_ids(args.ids)
    remote.run(f"scancel {' '.join(ids)} || ms_die \"scancel failed\"")
    print(f"cancelled {' '.join(ids)}")


def cmd_fetch(args):
    target = args.target
    limit = None if args.all else config.FETCH_MAX_MB
    if re.fullmatch(r"[0-9]+", target):
        find = f'd=$(ms_rundir {target}) || ms_die "no run directory for job {target}"\np=$(ms_resolve "$d") || exit 1'
    else:
        find = f'p=$(ms_resolve {q(remote_path(target))}) || exit 1\n[ -e "$p" ] || ms_die "not found: $p"'
    skip = ""
    if limit:
        skip = (f'[ -d "$p" ] && find "$p" -type f -size +{limit}M -printf \'SKIP|%P|%s\\n\' '
                "2>/dev/null | head -n 20")
    out = remote.run(f"""
{find}
if [ -d "$p" ]; then echo "P|d|$p"; else echo "P|f|$p"; fi
{skip}
true
""")
    kind = path = None
    skipped = []
    for line in out.splitlines():
        f = line.split("|")
        if f[0] == "P":
            kind, path = f[1], f[2]
        elif f[0] == "SKIP":
            skipped.append((f[1], int(f[2])))

    dest = os.path.join(args.dest, os.path.basename(path)) if kind == "d" else args.dest
    os.makedirs(dest, exist_ok=True)
    source = path + "/" if kind == "d" else path
    if remote.TRANSPORT is None:
        source = f"{config.HOST}:{source}"
    cmd = ["rsync", "-a", "-z", "-s", "--stats", "--timeout=120",
           "-e", "ssh -o BatchMode=yes -o ConnectTimeout=30"]
    if limit:
        cmd.append(f"--max-size={limit}m")
    proc = subprocess.run(cmd + [source, dest + "/"], capture_output=True, text=True)
    if proc.returncode != 0:
        raise MsError("rsync failed: " + (proc.stderr.strip().splitlines() or ["no error text"])[-1])
    count = re.search(r"Number of regular files transferred: ([\d,.]+)", proc.stdout)
    size = re.search(r"Total transferred file size: ([\d,.]+)", proc.stdout)
    count = count.group(1) if count else "?"
    size = human_size(int(re.sub(r"\D", "", size.group(1)))) if size else "?"
    print(f"fetched {count} files ({size}) from {path} to {dest}")
    for rel, nbytes in skipped:
        print(f"skipped {rel} ({human_size(nbytes)}; over {limit}M, use --all to include)")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="ms",
        description="Run jobs on mscluster. Each command is one short SSH session.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    def add(name, func, help_text):
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.set_defaults(func=func)
        return p

    p = add("nodes", cmd_nodes, "free and busy nodes per partition, and my usage against the caps")
    p.add_argument("--json", action="store_true")

    p = add("status", cmd_status, "my jobs, one line each; a sweep is one line")
    p.add_argument("ids", nargs="*", metavar="JOBID")
    p.add_argument("--json", action="store_true")

    p = add("logs", cmd_logs, "tail of a job's log")
    p.add_argument("id", metavar="JOBID")
    p.add_argument("-n", "--lines", type=int, default=40)
    p.add_argument("--task", type=int, metavar="N", help="task number, for a sweep")

    p = add("ls", cmd_ls, "list a folder on the cluster")
    p.add_argument("path", nargs="?", default="~")
    p.add_argument("--json", action="store_true")

    p = add("rm", cmd_rm, "delete files or folders on the cluster (no undo)")
    p.add_argument("paths", nargs="+", metavar="PATH")

    p = add("mv", cmd_mv, "move or rename on the cluster")
    p.add_argument("src")
    p.add_argument("dst")

    add("sync", cmd_sync, "bring the cluster's copy of this git repo up to date with origin")

    p = add("submit", cmd_submit, "submit a job: ms submit [options] -- COMMAND")
    p.add_argument("-p", "--partition", choices=list(config.PARTITIONS),
                   default=config.DEFAULT_PARTITION)
    p.add_argument("-t", "--time", default=config.DEFAULT_TIME,
                   help=f"time limit, e.g. 02:00:00, 90m, 1d (default {config.DEFAULT_TIME})")
    p.add_argument("-n", "--name", help="short label for the job name")
    p.add_argument("--workdir", help="folder on the cluster to run in (default: this repo's copy)")
    p.add_argument("--env", metavar="DIR", help="virtual environment to activate (default: .venv)")
    p.add_argument("--module", action="append", metavar="NAME", help="module to load; repeatable")
    p.add_argument("--array", type=int, metavar="N", help="sweep of N tasks; {i} and {n} are filled in")
    p.add_argument("--grid", metavar="SPEC", help='sweep over settings, e.g. "seed=0,1,2 lr=1e-3,3e-4"')
    p.add_argument("--per-node", type=int, default=1, metavar="P", help="tasks to run on each node")
    p.add_argument("--retry", metavar="JOBID", help="rerun the unfinished tasks of a sweep")

    p = add("wait", cmd_wait, "block until jobs finish; exit code 0 only if all succeeded")
    p.add_argument("ids", nargs="+", metavar="JOBID")
    p.add_argument("--interval", type=int, default=180, help="seconds between checks (min 60)")
    p.add_argument("--timeout", type=int, default=0, help="give up after this many seconds")
    p.add_argument("--json", action="store_true")

    p = add("cancel", cmd_cancel, "cancel jobs")
    p.add_argument("ids", nargs="+", metavar="JOBID")

    p = add("fetch", cmd_fetch, "copy a run's folder (or any allowed path) to the laptop")
    p.add_argument("target", metavar="JOBID_OR_PATH")
    p.add_argument("dest", nargs="?", default="results")
    p.add_argument("--all", action="store_true",
                   help=f"include files over {config.FETCH_MAX_MB} MB")
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    command_argv = []
    if "--" in argv:
        split = argv.index("--")
        argv, command_argv = argv[:split], argv[split + 1:]
    args = build_parser().parse_args(argv)
    args.command_argv = command_argv
    try:
        return args.func(args) or 0
    except MsError as exc:
        print(f"ms: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
