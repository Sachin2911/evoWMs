"""Tests that need no cluster.

The bash that normally runs on the login node runs locally instead, the two
allowed folders are temporary directories, and Slurm is replaced by small fake
commands. Generated job scripts are then executed for real.

    python3 -m unittest discover -s msCli
"""

import contextlib
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cli  # noqa: E402
import config  # noqa: E402
import jobs  # noqa: E402
import remote  # noqa: E402
from remote import MsError  # noqa: E402

FAKE_SLURM = {
    "sbatch": r"""
arr=""; file=""
for a in "$@"; do
  case "$a" in --array=*) arr=${a#--array=} ;; --hold|--parsable) ;; *) file=$a ;; esac
done
# Like Slurm, an option on the command line wins over the one in the script.
[ -n "$arr" ] || arr=$(sed -n 's/^#SBATCH --array=//p' "$file")
n=$(( $(cat "$FAKE/next" 2>/dev/null || echo 100) + 1 ))
echo "$n" > "$FAKE/next"
cp "$file" "$FAKE/job-$n.sh"
echo "$arr" > "$FAKE/job-$n.array"
echo "$n"
""",
    "scontrol": 'echo "$@" >> "$FAKE/scontrol.log"',
    "scancel": 'echo "$@" >> "$FAKE/scancel.log"',
    "sinfo": 'cat "$FAKE/sinfo.out" 2>/dev/null; true',
    "sacct": 'cat "$FAKE/sacct.out" 2>/dev/null; true',
    # The status query asks for job names (%j); the others only count or list ids.
    "squeue": r"""
case "$*" in
  *%j*) cat "$FAKE/squeue.out" 2>/dev/null ;;
  *) cat "$FAKE/squeue.ids" 2>/dev/null ;;
esac
true
""",
}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ms-test-")
        self.home = os.path.join(self.tmp, "home")
        self.data = os.path.join(self.tmp, "data")
        self.outside = os.path.join(self.tmp, "outside")
        self.fake = os.path.join(self.tmp, "fake")
        for d in (self.home, self.data, self.outside, self.fake, os.path.join(self.fake, "bin")):
            os.makedirs(d)
        for name, body in FAKE_SLURM.items():
            path = os.path.join(self.fake, "bin", name)
            with open(path, "w") as f:
                f.write("#!/bin/bash\n" + body + "\n")
            os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)

        self.saved = (config.HOME, config.DATASETS, config.RUNS, remote.TRANSPORT,
                      os.environ["PATH"], os.getcwd(), config.BIG_TREE_ENTRIES,
                      config.FETCH_MAX_MB)
        config.HOME, config.DATASETS = self.home, self.data
        config.RUNS = os.path.join(self.home, "runs")
        remote.TRANSPORT = ["bash", "-s"]
        os.environ["PATH"] = os.path.join(self.fake, "bin") + os.pathsep + os.environ["PATH"]
        os.environ["FAKE"] = self.fake
        os.chdir(self.tmp)  # not a git repository

    def tearDown(self):
        (config.HOME, config.DATASETS, config.RUNS, remote.TRANSPORT,
         os.environ["PATH"], cwd, config.BIG_TREE_ENTRIES, config.FETCH_MAX_MB) = self.saved
        os.chdir(cwd)
        shutil.rmtree(self.tmp)

    def ms(self, *argv):
        """Run a command; return (exit code, stdout, stderr)."""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def ok(self, *argv):
        code, out, err = self.ms(*argv)
        self.assertEqual(code, 0, err)
        return out

    def refused(self, *argv):
        code, out, err = self.ms(*argv)
        self.assertEqual(code, 1, out)
        return err

    def write(self, path, text="x"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
        return path

    def read(self, path):
        with open(path) as f:
            return f.read()

    def run_job(self, jid, element=None, name=None):
        """Execute a submitted script the way Slurm would."""
        env = dict(os.environ, SLURM_JOB_ID=str(jid))
        log = None
        if element is not None:
            env.update(SLURM_ARRAY_JOB_ID=str(jid), SLURM_ARRAY_TASK_ID=str(element))
        elif name:
            log = open(os.path.join(config.RUNS, f"{name}-{jid}", "slurm.out"), "w")
        proc = subprocess.run(["bash", os.path.join(self.fake, f"job-{jid}.sh")], env=env,
                              stdout=log or subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if log:
            log.close()
        return proc


class PureHelpers(unittest.TestCase):
    def test_time_formats(self):
        for text, seconds in [("02:00:00", 7200), ("2-01:02:03", 176523), ("90m", 5400),
                              ("2h", 7200), ("1d", 86400), ("1h30m", 5400)]:
            self.assertEqual(jobs.parse_time(text), seconds, text)
        for bad in ("", "abc", "90", "10:30", "1:2:3:4", "h"):
            with self.assertRaises(MsError):
                jobs.parse_time(bad)

    def test_time_must_fit_the_partition(self):
        self.assertEqual(jobs.check_time("1d", "batch"), "1-00:00:00")
        self.assertEqual(jobs.check_time("3d", "bigbatch"), "3-00:00:00")
        with self.assertRaises(MsError):
            jobs.check_time("25:00:00", "batch")
        with self.assertRaises(MsError):
            jobs.check_time("4d", "bigbatch")

    def test_grid(self):
        self.assertEqual(
            jobs.expand_grid("seed=0,1 optimizer.lr=1e-3,3e-4"),
            ["seed=0 optimizer.lr=1e-3", "seed=0 optimizer.lr=3e-4",
             "seed=1 optimizer.lr=1e-3", "seed=1 optimizer.lr=3e-4"],
        )
        for bad in ("seed", "seed=", "seed=1,,2", "dims=[1,2]", ""):
            with self.assertRaises(MsError):
                jobs.expand_grid(bad)

    def test_names(self):
        self.assertEqual(jobs.clean_name("evoWMs-train one/two"), "evoWMs-train-one-two")
        self.assertEqual(jobs.default_label(["python", "src/train.py", "a=1"]), "train")
        self.assertEqual(jobs.default_label(["nvidia-smi"]), "nvidia-smi")

    def test_tilde_expanded_by_the_local_shell_is_undone(self):
        home = os.path.expanduser("~")
        self.assertEqual(cli.remote_path(home + "/runs"), "~/runs")
        self.assertEqual(cli.remote_path("/datasets/smohan/x"), "/datasets/smohan/x")


class AllowedFolders(Base):
    def test_ls(self):
        self.write(os.path.join(self.home, "a.txt"), "hello")
        os.makedirs(os.path.join(self.home, "sub"))
        out = self.ok("ls")
        self.assertIn("a.txt", out)
        self.assertIn("sub/", out)
        listing = json.loads(self.ok("ls", self.home, "--json"))
        self.assertEqual(listing["total"], 2)
        self.assertIn("(empty)", self.ok("ls", self.data))

    def test_ls_is_capped(self):
        for i in range(config.LS_LIMIT + 5):
            self.write(os.path.join(self.data, f"f{i:04d}"))
        out = self.ok("ls", self.data)
        self.assertEqual(len(out.splitlines()), config.LS_LIMIT + 1)
        self.assertIn("5 more", out)

    def test_outside_is_refused(self):
        self.write(os.path.join(self.outside, "secret"))
        for path in (self.outside, "/etc", self.home + "/../outside", self.home + "x"):
            self.assertIn("outside the allowed folders", self.refused("ls", path))
        self.assertIn("outside", self.refused("rm", os.path.join(self.outside, "secret")))
        self.assertTrue(os.path.exists(os.path.join(self.outside, "secret")))

    def test_symlink_cannot_lead_outside(self):
        self.write(os.path.join(self.outside, "secret"))
        link = os.path.join(self.home, "link")
        os.symlink(self.outside, link)
        self.assertIn("outside", self.refused("ls", link))
        self.assertIn("outside", self.refused("rm", os.path.join(link, "secret")))
        self.assertIn("outside", self.refused("mv", self.write(os.path.join(self.home, "f")), link))
        # Deleting the link removes the link, never what it points at.
        self.ok("rm", link)
        self.assertFalse(os.path.lexists(link))
        self.assertTrue(os.path.exists(os.path.join(self.outside, "secret")))

    def test_the_two_folders_themselves_are_protected(self):
        for path in (self.home, self.data, self.home + "/", self.home + "/sub/.."):
            self.assertIn("allowed folder itself", self.refused("rm", path))
        self.assertIn("allowed folder itself", self.refused("mv", self.home, self.data))
        self.assertTrue(os.path.isdir(self.home))

    def test_login_keys_are_protected(self):
        key = self.write(os.path.join(self.home, ".ssh", "authorized_keys"))
        for path in (os.path.dirname(key), key):
            self.assertIn("login keys", self.refused("rm", path))
        self.assertIn("login keys", self.refused("mv", key, self.data))
        other = self.write(os.path.join(self.home, "f"))
        self.assertIn("login keys", self.refused("mv", other, os.path.dirname(key)))
        self.assertTrue(os.path.exists(key))

    def test_rm(self):
        a = self.write(os.path.join(self.home, "d", "a"))
        b = self.write(os.path.join(self.data, "b"))
        out = self.ok("rm", os.path.dirname(a), b)
        self.assertEqual(len(out.splitlines()), 2)
        self.assertFalse(os.path.exists(os.path.dirname(a)))
        self.assertFalse(os.path.exists(b))

    def test_rm_deletes_nothing_if_any_path_is_bad(self):
        a = self.write(os.path.join(self.home, "a"))
        self.assertIn("not found", self.refused("rm", a, os.path.join(self.home, "missing")))
        self.assertTrue(os.path.exists(a))

    def test_rm_of_a_large_tree_becomes_a_job(self):
        config.BIG_TREE_ENTRIES = 3
        big = os.path.join(self.home, "big")
        for i in range(6):
            self.write(os.path.join(big, f"f{i}"))
        out = self.ok("rm", big)
        self.assertIn("in job 101", out)
        self.assertTrue(os.path.isdir(big))  # not deleted on the login node
        script = self.read(os.path.join(self.fake, "job-101.sh"))
        self.assertIn("rm -rf -- " + big, script)
        self.assertEqual(self.run_job(101).returncode, 0)
        self.assertFalse(os.path.exists(big))

    def test_mv(self):
        src = self.write(os.path.join(self.home, "a.txt"))
        self.ok("mv", src, os.path.join(self.home, "b.txt"))
        self.assertTrue(os.path.exists(os.path.join(self.home, "b.txt")))
        os.makedirs(os.path.join(self.home, "dir"))
        self.ok("mv", os.path.join(self.home, "b.txt"), os.path.join(self.home, "dir"))
        self.assertTrue(os.path.exists(os.path.join(self.home, "dir", "b.txt")))
        other = self.write(os.path.join(self.home, "c.txt"))
        self.assertIn("already exists", self.refused("mv", other, other))
        self.assertIn("outside", self.refused("mv", other, self.outside))

    def test_mv_between_the_folders(self):
        tree = os.path.join(self.home, "tree")
        self.write(os.path.join(tree, "f"))
        self.assertIn("in job 101", self.ok("mv", tree, self.data))
        self.assertTrue(os.path.isdir(tree))
        # Interrupted halfway: the destination exists and holds part of the tree.
        self.write(os.path.join(self.data, "tree", "f"), "partial")
        self.assertEqual(self.run_job(101).returncode, 0)
        self.assertEqual(self.read(os.path.join(self.data, "tree", "f")), "x")
        self.assertFalse(os.path.exists(tree))
        self.assertFalse(os.path.exists(os.path.join(self.data, "tree", "tree")))
        # Running the same job again after it finished is harmless.
        proc = self.run_job(101)
        self.assertEqual(proc.returncode, 0)
        self.assertIn("already gone", proc.stdout)


AWKWARD = ["a b", "x=[1,2]", "$HOME", "it's", 'q="z"', "*", "a;b", "{i}", "back\\slash", "ü"]
PRINT_ARGS = "import sys, os, json; print(json.dumps(sys.argv[1:])); print(os.environ['MS_RUN_DIR'])"


class Submit(Base):
    def test_single_job_gets_its_arguments_unchanged(self):
        out = self.ok("submit", "--time", "10m", "--name", "probe", "--",
                      sys.executable, "-c", PRINT_ARGS, *AWKWARD)
        self.assertTrue(out.startswith("submitted 101  job-probe  batch  0-00:10:00"), out)
        run = os.path.join(config.RUNS, "job-probe-101")
        self.assertTrue(os.path.isfile(os.path.join(run, "job.sbatch")))
        self.assertIn("release 101", self.read(os.path.join(self.fake, "scontrol.log")))
        self.assertEqual(self.run_job(101, name="job-probe").returncode, 0)
        log = self.read(os.path.join(run, "slurm.out")).splitlines()
        self.assertEqual(json.loads(log[1]), AWKWARD)
        self.assertEqual(log[2], run)
        self.assertIn("ms: exit 0", log[3])
        self.assertIn("ms: start", self.ok("logs", "101"))

    def test_job_exit_code_is_the_commands(self):
        self.ok("submit", "-n", "bad", "--", sys.executable, "-c", "raise SystemExit(7)")
        self.assertEqual(self.run_job(101, name="job-bad").returncode, 7)

    def test_limits(self):
        self.assertIn("over the batch limit", self.refused("submit", "-t", "2d", "--", "true"))
        self.ok("submit", "-p", "bigbatch", "-t", "2d", "--", "true")
        self.assertIn("nothing to run", self.refused("submit"))
        self.assertIn("cap is 100", self.refused("submit", "--array", "101", "--", "true"))
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            cli.main(["submit", "-p", "biggpu", "--", "true"])

    def test_queue_cap(self):
        self.write(os.path.join(self.fake, "squeue.ids"), "1\n" * 100)
        self.assertIn("per-user cap of 100", self.refused("submit", "--", "true"))
        self.assertFalse(os.path.exists(os.path.join(self.fake, "job-101.sh")))

    def test_missing_workdir(self):
        err = self.refused("submit", "--workdir", os.path.join(self.home, "nope"), "--", "true")
        self.assertIn("ms sync", err)
        err = self.refused("submit", "--workdir", self.outside, "--", "true")
        self.assertIn("outside", err)

    def test_sweep_packs_tasks_and_records_results(self):
        code = "import sys; print(sys.argv[1:]); sys.exit(3 if sys.argv[1] == 't=3' else 0)"
        out = self.ok("submit", "--array", "5", "--per-node", "2", "-n", "sw", "--",
                      sys.executable, "-c", code, "t={i}", "n={n}")
        self.assertIn("sweep of 5 tasks, 2 per node, 3 nodes' worth", out)
        self.assertEqual(self.read(os.path.join(self.fake, "job-101.array")).strip(), "0-2%13")
        sweep = os.path.join(config.RUNS, "job-sw-101")
        codes = [self.run_job(101, element=e).returncode for e in range(3)]
        self.assertEqual(codes, [0, 1, 0])
        self.assertEqual(sorted(os.listdir(os.path.join(sweep, ".ms", "done"))),
                         ["0000", "0001", "0002", "0004"])
        self.assertEqual(self.read(os.path.join(sweep, ".ms", "failed", "0003")).strip(), "3")
        self.assertIn("['t=4', 'n=5']", self.read(os.path.join(sweep, "task-0004", "task.log")))
        self.assertIn("ms: exit 3", self.ok("logs", "101", "--task", "3"))
        self.assertIn("pass --task", self.refused("logs", "101"))

        row = json.loads(self.ok("status", "101", "--json"))[0]
        self.assertEqual(row["state"], "FAILED")
        self.assertEqual(row["sweep"], {"tasks": 5, "done": 4, "running": 0, "waiting": 0,
                                        "failed": [3]})
        self.assertIn("4/5 done", self.ok("status", "101"))
        self.assertEqual(self.ms("wait", "101")[0], 1)

        # A finished task is not run again.
        before = self.read(os.path.join(sweep, "task-0000", "task.log"))
        self.run_job(101, element=0)
        self.assertEqual(self.read(os.path.join(sweep, "task-0000", "task.log")), before)

        # Retry resubmits only the node holding the unfinished task, into the same folder.
        out = self.ok("submit", "--retry", "101")
        self.assertIn("retry of sweep 101: 1 unfinished tasks on 1 nodes", out)
        self.assertEqual(self.read(os.path.join(self.fake, "job-102.array")).strip(), "1%13")
        self.assertEqual(os.path.realpath(os.path.join(config.RUNS, "job-sw-102")), sweep)
        self.assertFalse(os.path.exists(os.path.join(sweep, ".ms", "failed", "0003")))
        self.assertIn("only the sweep's job id", self.refused("submit", "--retry", "101", "--", "x"))

    def test_retry_when_all_done(self):
        self.ok("submit", "--array", "2", "-n", "ok", "--", "true")
        for e in range(2):
            self.assertEqual(self.run_job(101, element=e).returncode, 0)
        self.assertIn("nothing to retry", self.refused("submit", "--retry", "101"))
        row = json.loads(self.ok("status", "101", "--json"))[0]
        self.assertEqual((row["state"], row["ok"]), ("COMPLETED", True))
        self.assertEqual(self.ms("wait", "101")[0], 0)

    def test_grid_appends_one_combination_per_task(self):
        self.ok("submit", "--grid", "seed=0,1 lr=1e-3,3e-4", "-n", "g", "--",
                sys.executable, "-c", "import sys; print(sys.argv[1:])", "data=pusht")
        sweep = os.path.join(config.RUNS, "job-g-101")
        self.assertEqual(self.read(os.path.join(sweep, "tasks.txt")).splitlines()[3],
                         "seed=1 lr=3e-4")
        self.assertEqual(self.read(os.path.join(self.fake, "job-101.array")).strip(), "0-3%13")
        self.assertEqual(self.run_job(101, element=3).returncode, 0)
        self.assertIn("['data=pusht', 'seed=1', 'lr=3e-4']",
                      self.read(os.path.join(sweep, "task-0003", "task.log")))

    def test_generated_scripts_are_valid_bash(self):
        for script in (
            jobs.job_script(name="n", partition="batch", time="0-01:00:00", workdir="/w",
                            argv=["python", "a.py"], modules=["pytorch/x"], venv="/v"),
            jobs.sweep_script(name="n", partition="bigbatch", time="0-01:00:00", workdir="/w",
                              argv=["python", "a.py"], ntasks=7, per_node=3),
        ):
            proc = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertNotIn("--gres", script)
        self.assertIn("--array=0-2%6", script)


class StatusAndFriends(Base):
    def test_status_merges_queue_accounting_and_status_file(self):
        self.write(os.path.join(self.fake, "squeue.out"),
                   "205|evoWMs-train|batch|RUNNING|0:12:44|mscluster130|None\n"
                   "206|evoWMs-eval|bigbatch|PENDING|0:00|(Priority)|Priority\n")
        self.write(os.path.join(self.fake, "sacct.out"),
                   "205|evoWMs-train|batch|RUNNING|00:12:44|mscluster130|0:0\n"
                   "190|evoWMs-old|batch|COMPLETED|00:03:00|mscluster131|0:0\n"
                   "191|evoWMs-oops|batch|FAILED|00:00:05|mscluster131|1:0\n"
                   "192|evoWMs-gone|batch|CANCELLED by 1234|00:00:05|mscluster131|0:15\n")
        self.write(os.path.join(config.RUNS, "evoWMs-train-205", "status.json"),
                   '{"step": 1200, "of": 5000, "metric": 0.431}\n')
        rows = {r["id"]: r for r in json.loads(self.ok("status", "--json"))}
        self.assertEqual(list(rows), ["190", "191", "192", "205", "206"])
        self.assertEqual(rows["205"]["info"], "step=1200 of=5000 metric=0.431")
        self.assertTrue(rows["205"]["active"])
        self.assertEqual((rows["190"]["ok"], rows["191"]["ok"]), (True, False))
        self.assertEqual(rows["192"]["state"], "CANCELLED")
        self.assertIn("exit=1:0", rows["191"]["info"])
        text = self.ok("status")
        self.assertEqual(len(text.splitlines()), 5)
        self.assertIn("step=1200", text)

    def test_an_array_with_one_failed_element_is_failed(self):
        self.write(os.path.join(self.fake, "sacct.out"),
                   "300_0|a|batch|COMPLETED|00:00:05|n1|0:0\n300_1|a|batch|FAILED|00:00:05|n2|3:0\n")
        (row,) = json.loads(self.ok("status", "300", "--json"))
        self.assertEqual((row["state"], row["ok"]), ("FAILED", False))

    def test_wait_reports_failure_through_the_exit_code(self):
        self.write(os.path.join(self.fake, "sacct.out"),
                   "190|a|batch|COMPLETED|00:03:00|n1|0:0\n191|b|batch|TIMEOUT|01:00:00|n1|0:0\n")
        self.assertEqual(self.ms("wait", "190")[0], 0)
        code, out, _ = self.ms("wait", "190", "191")
        self.assertEqual(code, 1)
        self.assertIn("TIMEOUT", out)

    def test_nodes(self):
        self.write(os.path.join(self.fake, "sinfo.out"),
                   "N|batch|idle|83\nN|batch|allocated|1\nN|batch|down*|10\nN|batch|drained|6\n"
                   "N|bigbatch|mixed|38\nN|bigbatch|idle|1\nN|bigbatch|idle*|9\n")
        stats = json.loads(self.ok("nodes", "--json"))
        self.assertEqual(stats["batch"]["idle"], 83)
        self.assertEqual(stats["batch"]["down"], 16)
        self.assertEqual((stats["bigbatch"]["idle"], stats["bigbatch"]["down"]), (1, 9))
        self.assertEqual(len(self.ok("nodes").splitlines()), 2)

    def test_cancel(self):
        self.ok("cancel", "101", "102")
        self.assertEqual(self.read(os.path.join(self.fake, "scancel.log")).strip(), "101 102")
        self.assertIn("not a job id", self.refused("cancel", "101; rm -rf /"))

    def test_fetch_skips_large_files_unless_asked(self):
        config.FETCH_MAX_MB = 1
        run = os.path.join(config.RUNS, "job-x-300")
        self.write(os.path.join(run, "result.json"), "{}")
        self.write(os.path.join(run, ".hydra", "config.yaml"), "a: 1")
        self.write(os.path.join(run, "ckpt.pt"), "0" * (2 * 1024 * 1024))
        dest = os.path.join(self.tmp, "results")
        out = self.ok("fetch", "300", dest)
        self.assertIn("skipped ckpt.pt", out)
        got = os.path.join(dest, "job-x-300")
        self.assertTrue(os.path.exists(os.path.join(got, ".hydra", "config.yaml")))
        self.assertFalse(os.path.exists(os.path.join(got, "ckpt.pt")))
        self.ok("fetch", "300", dest, "--all")
        self.assertTrue(os.path.exists(os.path.join(got, "ckpt.pt")))
        self.assertIn("outside", self.refused("fetch", self.outside, dest))


class Sync(Base):
    def git(self, *args, cwd=None):
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.com",
                   GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.com")
        proc = subprocess.run(["git", "-c", "init.defaultBranch=main", *args], cwd=cwd or self.work,
                              env=env, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc.stdout.strip()

    def setUp(self):
        super().setUp()
        self.origin = os.path.join(self.tmp, "proj.git")
        self.work = os.path.join(self.tmp, "work")
        self.git("init", "--bare", self.origin, cwd=self.tmp)
        self.git("clone", self.origin, self.work, cwd=self.tmp)
        self.write(os.path.join(self.work, "train.py"), "print('v1')\n")
        self.git("add", "-A")
        self.git("commit", "-m", "one")
        self.git("push", "origin", "HEAD:main")
        self.git("branch", "--set-upstream-to=origin/main")
        os.chdir(self.work)

    def test_clone_then_update(self):
        copy = os.path.join(self.home, "proj")
        self.assertIn(f"synced proj main @ {self.git('rev-parse', 'HEAD')[:7]} -> {copy}",
                      self.ok("sync"))
        self.assertEqual(self.read(os.path.join(copy, "train.py")), "print('v1')\n")

        self.write(os.path.join(self.work, "train.py"), "print('v2')\n")
        self.assertIn("uncommitted local changes", self.ok("sync"))
        self.git("commit", "-am", "two")
        self.assertIn("git push", self.refused("sync"))
        self.assertEqual(self.read(os.path.join(copy, "train.py")), "print('v1')\n")
        self.git("push", "origin", "HEAD:main")
        self.ok("sync")
        self.assertEqual(self.read(os.path.join(copy, "train.py")), "print('v2')\n")

    def test_submit_runs_in_the_synced_copy_and_records_the_commit(self):
        self.assertIn("ms sync", self.refused("submit", "--", sys.executable, "train.py"))
        self.ok("sync")
        out = self.ok("submit", "--", sys.executable, "train.py")
        self.assertTrue(out.startswith("submitted 101  proj-train  batch"), out)
        self.assertEqual(self.run_job(101, name="proj-train").returncode, 0)
        run = os.path.join(config.RUNS, "proj-train-101")
        self.assertIn("v1", self.read(os.path.join(run, "slurm.out")))
        self.assertEqual(self.read(os.path.join(run, "commit.txt")).strip(),
                         self.git("rev-parse", "HEAD"))

    def test_submit_warns_when_the_cluster_copy_is_behind(self):
        self.ok("sync")
        self.write(os.path.join(self.work, "train.py"), "print('v2')\n")
        self.git("commit", "-am", "two")
        code, _, err = self.ms("submit", "--", sys.executable, "train.py")
        self.assertEqual(code, 0)
        self.assertIn("run: ms sync", err)


class StatusHelper(unittest.TestCase):
    def test_round_trip(self):
        import ms_status
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["MS_RUN_DIR"] = tmp
            try:
                self.assertIsNone(ms_status.read_status())
                ms_status.write_status(step=3, of=10)
                status = ms_status.read_status()
                self.assertEqual((status["step"], status["of"]), (3, 10))
                self.assertIn("updated", status)
            finally:
                del os.environ["MS_RUN_DIR"]


if __name__ == "__main__":
    unittest.main()
