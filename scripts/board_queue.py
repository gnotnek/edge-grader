#!/usr/bin/env python3
"""edge-grader's own job queue on a lab board: runs scripts/board_grade.sh jobs one after another.

    python3 scripts/board_queue.py add BENCH=gsm INSTANCES=1 -- e2b symbolic   # one job
    python3 scripts/board_queue.py add --preset phase0      # phase 0's 10 jobs for this board
    python3 scripts/board_queue.py add --preset rag         # stage 4's 12 RAG-en jobs
    python3 scripts/board_queue.py status
    python3 scripts/board_queue.py start                    # the worker, detached
    python3 scripts/board_queue.py stop                     # finish the current job, then stop
    python3 scripts/board_queue.py cancel 3 | retry 3

From the Mac: scripts/queue.sh pi|jetson <the same arguments>.

It is not agentic-edge's queue and never runs through it (CLAUDE.md): every job is an ordinary
board_grade.sh call, so the serving flags, telemetry and separation checks are the script's own.
Sharing the board with the lab: board_grade.sh refuses while an agentic-edge run has the board;
the job then goes back to pending and the worker tries again after a wait. agentic-edge's queue
in turn waits while one of these runs is alive (its run_measured.sh). One worker per board.

State lives in queue/ (git-ignored, never deployed over): jobs.json, logs/job-<id>.log per job,
worker.log. Standard library only: the boards have no pip.
"""

import argparse
import contextlib
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# The settings a job may carry: board_grade.sh's own switches (later phases add the last ones).
JOB_ENV = ("BENCH", "INSTANCES", "MEASURE", "THINKING", "VARIANT", "RAG", "IMAGES", "MMPROJ",
           "LG_ENGINE")
# Never inherited by a real run: the worker's shell may have any of these set.
STRIP_ENV = (*JOB_ENV, "DRY_RUN", "EG_BOARD", "EG_RUN_DIR", "EG_INNER")
BUSY = "refusing: the board is busy"
RUN_DIR_RE = re.compile(r"everything for this run is in (\S+)")


def queue_dir() -> Path:
    return Path(os.environ.get("EG_QUEUE_DIR", REPO / "queue"))


def board_grade() -> str:
    return os.environ.get("EG_BOARD_GRADE", str(REPO / "scripts" / "board_grade.sh"))


def board() -> str:
    if os.environ.get("EG_BOARD"):
        return os.environ["EG_BOARD"]
    if Path("/etc/nv_tegra_release").exists():
        return "jetson"
    with contextlib.suppress(OSError):
        if "raspberry" in Path("/proc/device-tree/model").read_text().lower():
            return "pi5"
    sys.exit("not a Pi 5 or Jetson (set EG_BOARD to pretend)")


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@contextlib.contextmanager
def jobs_locked(qdir: Path):
    """jobs.json, read and written under a lock so add/cancel and the worker don't collide."""
    qdir.mkdir(parents=True, exist_ok=True)
    with open(qdir / "jobs.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = qdir / "jobs.json"
        jobs = json.loads(path.read_text()) if path.exists() else []
        yield jobs
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(jobs, indent=1) + "\n")
        tmp.replace(path)


# Stage 4's RAG-en runs: each benchmark's frozen retrieval file, the same ones the Kaggle RAG runs used.
RAG_FILES = {bench: f"results/rag/retrieval-{bench}-en.bge-small-en-k5.json"
             for bench in ("scientsbank", "gsm", "news")}


def preset(name: str, where: str) -> list[tuple[dict, list[str]]]:
    if name not in ("phase0", "rag"):
        sys.exit(f"unknown preset: {name} (phase0, rag)")
    # Phase 0 (docs/superpowers/plans/2026-10-09-phase0-edge-baseline.md): the Jetson runs all
    # 1,000 GSM-Symbolic variants, the Pi the first variant set only.
    instances = "10" if where == "jetson" else "1"
    out = []
    if name == "rag":
        # Stage 4, RAG-en next to phase 0's baseline on the same board: every SciEntsBank split
        # (test_ua too: its board baseline is stage 1's), both GSM sets and news. Shortest first.
        for bench, split in (("scientsbank", "test_ua"), ("scientsbank", "test_uq"),
                             ("scientsbank", "test_ud"), ("news", "test"),
                             ("gsm", "gsm8k"), ("gsm", "symbolic")):
            for model in ("e2b", "e4b"):
                out.append(({"BENCH": bench, "INSTANCES": instances, "RAG": RAG_FILES[bench]},
                            [model, split]))
        return out
    for model in ("e2b", "e4b"):
        for bench, split in (("scientsbank", "test_uq"), ("scientsbank", "test_ud"),
                             ("gsm", "gsm8k"), ("gsm", "symbolic"), ("news", "test")):
            out.append(({"BENCH": bench, "INSTANCES": instances}, [model, split]))
    return out


def job_environ(job_env: dict, dry: bool) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in STRIP_ENV}
    if dry:
        env["DRY_RUN"] = "1"
        if os.environ.get("EG_BOARD"):
            env["EG_BOARD"] = os.environ["EG_BOARD"]
    return env | job_env


def check(job_env: dict, args: list[str]) -> None:
    """board_grade.sh's own dry run: a job it would refuse is refused now, not hours later."""
    p = subprocess.run(["bash", board_grade(), *args], env=job_environ(job_env, dry=True),
                       capture_output=True, text=True, check=False)
    if p.returncode != 0:
        sys.exit(f"board_grade.sh refuses {' '.join(args)}: {p.stderr.strip()}")


def cmd_add(a) -> None:
    if a.preset:
        todo = preset(a.preset, board())
    else:
        env, args = {}, list(a.args)
        while args and "=" in args[0]:
            key, value = args.pop(0).split("=", 1)
            if key not in JOB_ENV:
                sys.exit(f"{key}: not a job setting ({', '.join(JOB_ENV)})")
            env[key] = value
        if args and args[0] == "--":
            args.pop(0)
        if not args:
            sys.exit("usage: add [KEY=value ...] -- <model> [split|set] [n]")
        todo = [(env, args)]
    for env, args in todo:
        check(env, args)
    with jobs_locked(queue_dir()) as jobs:
        for env, args in todo:
            jobs.append({"id": max((j["id"] for j in jobs), default=0) + 1, "env": env,
                         "args": args, "state": "pending", "added": now(), "busy_waits": 0})
            print(f"added {jobs[-1]['id']}: {describe(jobs[-1])}")


def describe(job: dict) -> str:
    env = " ".join(f"{k}={v}" for k, v in job["env"].items())
    return f"{env + '  ' if env else ''}{' '.join(job['args'])}"


def cmd_status(_a) -> None:
    with jobs_locked(queue_dir()) as jobs:
        rows = [dict(j) for j in jobs]
    if not rows:
        print("queue is empty")
    for j in rows:
        print(f"{j['id']:>3}  {j['state']:<11} {describe(j):<44} rc={j.get('rc', '-')}"
              f"  {j.get('run_dir') or ''}")
    alive = worker_alive(queue_dir())
    print(f"worker: {'running' if alive else 'not running'}")


def worker_alive(qdir: Path) -> bool:
    qdir.mkdir(parents=True, exist_ok=True)
    with open(qdir / "worker.lock", "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(f, fcntl.LOCK_UN)
        return False


def set_state(job_id: int, frm: tuple[str, ...], to: str) -> None:
    with jobs_locked(queue_dir()) as jobs:
        job = next((j for j in jobs if j["id"] == job_id), None)
        if job is None:
            sys.exit(f"no job {job_id}")
        if job["state"] not in frm:
            sys.exit(f"job {job_id} is {job['state']}; only {'/'.join(frm)} jobs can become {to}")
        job["state"] = to
        print(f"job {job_id}: {to}")


def run_job(job: dict, qdir: Path) -> tuple[int, str]:
    """One board_grade.sh call; the log keeps every attempt. Returns (rc, this attempt's output)."""
    (qdir / "logs").mkdir(parents=True, exist_ok=True)
    out = []
    with open(qdir / "logs" / f"job-{job['id']}.log", "a") as log:
        log.write(f"== {now()} attempt: {describe(job)}\n")
        log.flush()
        p = subprocess.Popen(["bash", board_grade(), *job["args"]], cwd=REPO, text=True,
                             env=job_environ(job["env"], dry=False), stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in p.stdout:
            log.write(line)
            log.flush()
            out.append(line)
        rc = p.wait()
        log.write(f"== {now()} rc={rc}\n")
    return rc, "".join(out)


def cmd_run(_a) -> None:
    qdir = queue_dir()
    qdir.mkdir(parents=True, exist_ok=True)
    lock = open(qdir / "worker.lock", "a")  # noqa: SIM115 -- held, and locked, for the worker's whole life
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("a worker is already running on this board")
    gap = float(os.environ.get("EG_QUEUE_GAP", "20"))  # let the last server exit first
    busy_wait = float(os.environ.get("EG_QUEUE_BUSY_WAIT", "300"))
    (qdir / "stop").unlink(missing_ok=True)
    with jobs_locked(qdir) as jobs:
        for j in jobs:
            if j["state"] == "running":  # its worker died mid-run: not knowable whether it finished
                j["state"] = "interrupted"
                print(f"{now()} job {j['id']} was left running by a dead worker: interrupted")
    print(f"{now()} worker started (pid {os.getpid()})", flush=True)
    while True:
        if (qdir / "stop").exists():
            (qdir / "stop").unlink()
            print(f"{now()} stopped on request", flush=True)
            return
        with jobs_locked(qdir) as jobs:
            job = next((j for j in jobs if j["state"] == "pending"), None)
            if job is None:
                print(f"{now()} nothing pending; worker exits", flush=True)
                return
            job |= {"state": "running", "started": now()}
            job = dict(job)
        print(f"{now()} job {job['id']}: {describe(job)}", flush=True)
        rc, out = run_job(job, qdir)
        busy = rc != 0 and BUSY in out
        m = RUN_DIR_RE.search(out)
        with jobs_locked(qdir) as jobs:
            j = next(x for x in jobs if x["id"] == job["id"])
            if busy:
                j |= {"state": "pending", "busy_waits": j.get("busy_waits", 0) + 1}
            else:
                j |= {"state": "done" if rc == 0 else "failed", "rc": rc, "ended": now(),
                      "run_dir": m.group(1) if m else None}
        print(f"{now()} job {job['id']}: {'board busy, will retry' if busy else f'rc={rc}'}", flush=True)
        time.sleep(busy_wait if busy else gap)


def cmd_start(_a) -> None:
    qdir = queue_dir()
    if worker_alive(qdir):
        sys.exit("a worker is already running on this board")
    with open(qdir / "worker.log", "a") as log:  # the child keeps its own copy of the handle
        p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "run"], cwd=REPO,
                             stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)
    print(f"worker started, pid {p.pid}; log: {qdir / 'worker.log'}")


def cmd_stop(_a) -> None:
    qdir = queue_dir()
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / "stop").touch()
    print("the worker stops after the current job")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    add = sub.add_parser("add", help="add a job, or a preset's jobs")
    add.add_argument("--preset", help="phase0 | rag")
    add.add_argument("args", nargs=argparse.REMAINDER, help="[KEY=value ...] -- <model> [split|set] [n]")
    for name in ("status", "run", "start", "stop"):
        sub.add_parser(name)
    for name in ("cancel", "retry"):
        sub.add_parser(name).add_argument("id", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "cancel":
        set_state(a.id, ("pending",), "cancelled")
    elif a.cmd == "retry":
        set_state(a.id, ("failed", "interrupted", "cancelled"), "pending")
    else:
        {"add": cmd_add, "status": cmd_status, "run": cmd_run, "start": cmd_start,
         "stop": cmd_stop}[a.cmd](a)


if __name__ == "__main__":
    main()
