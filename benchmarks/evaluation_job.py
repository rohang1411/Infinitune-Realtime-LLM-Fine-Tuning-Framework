"""Durable launcher: save the complete console transcript and process exit code."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from benchmarks.evaluation_dashboard import ROOT, runner_command


def save(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def main():
    config = Path(sys.argv[1]).resolve()
    directory = config.parent
    state_path = directory / "dashboard_state.json"
    log_path = directory / "console.log"
    state = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
             "launcher_pid": os.getpid(), "log_path": str(log_path)}
    save(state_path, state)
    exit_code = 1
    try:
        settings = json.loads(config.read_text(encoding="utf-8"))
        command = runner_command(settings, directory)
        state["command"] = command
        save(state_path, state)
        # Append on resume. It preserves prior failed attempts and exact commands.
        with log_path.open("a", encoding="utf-8", buffering=1) as output:
            output.write("\n=== EVALUATION START " + state["started_utc"] + " ===\n")
            output.write("Command: " + json.dumps(command) + "\n")
            output.flush()
            child = subprocess.Popen(command, cwd=ROOT, stdout=output,
                                     stderr=subprocess.STDOUT,
                                     env={**os.environ, "PYTHONIOENCODING": "utf-8",
                                          "PYTHONUNBUFFERED": "1"})
            state["runner_pid"] = child.pid
            save(state_path, state)
            exit_code = child.wait()
            output.write(f"\n=== EVALUATION EXIT {exit_code} ===\n")
    except BaseException:
        detail = traceback.format_exc()
        (directory / "launch_error.txt").write_text(detail, encoding="utf-8")
        with log_path.open("a", encoding="utf-8") as output:
            output.write("\n=== LAUNCH FAILURE ===\n" + detail)
    finally:
        state["status"] = "complete" if exit_code == 0 else "failed"
        state["exit_code"] = exit_code
        state["ended_utc"] = datetime.now(timezone.utc).isoformat()
        save(state_path, state)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
