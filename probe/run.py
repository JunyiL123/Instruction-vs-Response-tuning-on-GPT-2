"""Sequential unattended compute; human review remains a separate explicit stage."""
import fcntl
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone

from .common import write_json


def run(config, root, device, allow_unreviewed):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "study.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("An unattended study is already running in this directory")
        prefix = [sys.executable, "-u", "-m", "probe", "--config", str(config.resolve()),
                  "--root", str(root), "--device", device]
        extra = ["--allow-unreviewed"] if allow_unreviewed else []
        stages = [["audit"], ["train", "--mode", "response"], ["train", "--mode", "instruction"],
                  ["evaluate", "--condition", "base"] + extra, ["report"],
                  ["evaluate", "--condition", "response"] + extra, ["report"],
                  ["evaluate", "--condition", "instruction"] + extra,
                  ["blind"], ["report"]]
        state = {"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(), "status": "running",
                 "human_review_required": True, "allow_unreviewed": allow_unreviewed}
        child = None

        def stop(signum, frame):
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait()
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            for stage in stages:
                state["stage"] = " ".join(stage)
                write_json(root / "study_status.json", state)
                print(f"\n[{datetime.now(timezone.utc).isoformat()}] {state['stage']}", flush=True)
                child = subprocess.Popen(prefix + stage)
                state["child_pid"] = child.pid
                write_json(root / "study_status.json", state)
                code = child.wait()
                if code:
                    raise subprocess.CalledProcessError(code, prefix + stage)
            state.update(status="compute_complete", finished_at=datetime.now(timezone.utc).isoformat())
        except (subprocess.CalledProcessError, KeyboardInterrupt) as error:
            state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
            raise
        finally:
            write_json(root / "study_status.json", state)
