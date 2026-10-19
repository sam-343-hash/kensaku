#!/usr/bin/env python3
"""
kaggle_agent.py - Submit training scripts to Kaggle automatically.

Commands
--------
  python kaggle_agent.py run  train.py --gpu            # push, wait, download results
  python kaggle_agent.py push train.py --gpu            # push only
  python kaggle_agent.py status my-job-name             # check status
  python kaggle_agent.py output my-job-name             # download results
  python kaggle_agent.py watch                          # watch ./jobs/ folder forever

Watch mode
----------
Drop  jobs/<name>.py  (and optionally  jobs/<name>.json  for settings)
into the jobs/ folder. The agent will submit it to Kaggle, wait, download
results into results/<name>/ and move the job to jobs/done/ (or jobs/failed/).

Optional jobs/<name>.json example:
  {"gpu": true, "internet": true,
   "datasets": ["username/my-dataset"],
   "competitions": ["titanic"]}

Setup
-----
  pip install kaggle
  Put your API token in ~/.kaggle/kaggle.json  (Kaggle > Settings > Create New Token)
  or set env vars KAGGLE_USERNAME and KAGGLE_KEY.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

POLL_SECONDS = 20
TERMINAL_OK = ("complete",)
TERMINAL_BAD = ("error", "cancel", "failed")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def sh(cmd, check=True):
    """Run a shell command and return stdout."""
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{out}")
    return out.strip()


def get_username():
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if cfg.exists():
        return json.loads(cfg.read_text())["username"]
    sys.exit("No Kaggle credentials found. See setup instructions at top of file.")


def slugify(name):
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "agent-job"


# --------------------------------------------------------------------------- #
# Core operations
# --------------------------------------------------------------------------- #
def push(script, name=None, gpu=False, internet=True, datasets=None,
         competitions=None, private=True):
    """Push a python script to Kaggle as a kernel. Returns 'user/slug'."""
    script = Path(script)
    if not script.exists():
        sys.exit(f"Script not found: {script}")

    user = get_username()
    slug = slugify(name or script.stem)
    ref = f"{user}/{slug}"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copy(script, tmp / script.name)
        meta = {
            "id": ref,
            "title": slug,  # title must slugify to the id slug
            "code_file": script.name,
            "language": "python",
            "kernel_type": "script",
            "is_private": private,
            "enable_gpu": gpu,
            "enable_internet": internet,
            "dataset_sources": datasets or [],
            "competition_sources": competitions or [],
            "kernel_sources": [],
        }
        (tmp / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        out = sh(["kaggle", "kernels", "push", "-p", str(tmp)])
        print(out)

    print(f"[push] submitted -> https://www.kaggle.com/code/{ref}")
    return ref


def status(ref):
    """Return a normalized status string: queued/running/complete/error/cancel."""
    out = sh(["kaggle", "kernels", "status", ref]).lower()
    for word in ("complete", "error", "cancel", "running", "queued"):
        if word in out:
            return word
    return out


def wait(ref, timeout_hours=12):
    """Block until the kernel finishes. Returns final status."""
    start = time.time()
    last = None
    while True:
        try:
            s = status(ref)
        except RuntimeError as e:
            # Kernel may not be visible for a few seconds right after push
            s = "queued"
            print(f"[wait] status not ready yet ({str(e)[:80]})")
        if s != last:
            print(f"[wait] {ref}: {s}")
            last = s
        if any(t in s for t in TERMINAL_OK + TERMINAL_BAD):
            return s
        if time.time() - start > timeout_hours * 3600:
            return "timeout"
        time.sleep(POLL_SECONDS)


def output(ref, dest=None):
    """Download kernel output files (models, logs, csvs)."""
    dest = Path(dest or f"results/{ref.split('/')[-1]}")
    dest.mkdir(parents=True, exist_ok=True)
    print(sh(["kaggle", "kernels", "output", ref, "-p", str(dest)], check=False))
    print(f"[output] saved to {dest}/")
    return dest


def run(script, **kw):
    """Push -> wait -> download. Returns (status, results_dir)."""
    ref = push(script, **kw)
    time.sleep(10)
    final = wait(ref)
    dest = output(ref)
    print(f"[done] {ref} finished with status: {final}")
    return final, dest


# --------------------------------------------------------------------------- #
# Watch mode (fully automatic)
# --------------------------------------------------------------------------- #
def watch(jobs_dir="jobs"):
    jobs = Path(jobs_dir)
    for d in (jobs, jobs / "done", jobs / "failed"):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[watch] drop .py files into {jobs}/  (Ctrl+C to stop)")
    while True:
        for script in sorted(jobs.glob("*.py")):
            cfg_path = script.with_suffix(".json")
            cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
            print(f"\n[watch] new job: {script.name} {cfg}")
            try:
                final, _ = run(
                    script,
                    gpu=cfg.get("gpu", False),
                    internet=cfg.get("internet", True),
                    datasets=cfg.get("datasets"),
                    competitions=cfg.get("competitions"),
                )
                target = jobs / ("done" if final in TERMINAL_OK else "failed")
            except Exception as e:  # keep the agent alive no matter what
                print(f"[watch] job crashed: {e}")
                target = jobs / "failed"
            shutil.move(str(script), target / script.name)
            if cfg_path.exists():
                shutil.move(str(cfg_path), target / cfg_path.name)
        time.sleep(5)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Run training scripts on Kaggle.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("push", "run"):
        p = sub.add_parser(name)
        p.add_argument("script")
        p.add_argument("--name")
        p.add_argument("--gpu", action="store_true")
        p.add_argument("--no-internet", action="store_true")
        p.add_argument("--dataset", action="append", default=[])
        p.add_argument("--competition", action="append", default=[])

    p = sub.add_parser("status"); p.add_argument("name")
    p = sub.add_parser("output"); p.add_argument("name")
    sub.add_parser("watch")

    a = ap.parse_args()

    def full(n):  # allow short names
        return n if "/" in n else f"{get_username()}/{slugify(n)}"

    if a.cmd in ("push", "run"):
        kw = dict(name=a.name, gpu=a.gpu, internet=not a.no_internet,
                  datasets=a.dataset, competitions=a.competition)
        push(a.script, **kw) if a.cmd == "push" else run(a.script, **kw)
    elif a.cmd == "status":
        print(status(full(a.name)))
    elif a.cmd == "output":
        output(full(a.name))
    elif a.cmd == "watch":
        watch()


if __name__ == "__main__":
    main()
