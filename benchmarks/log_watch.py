"""Read an evaluation's saved log in a visible terminal until its job ends."""

import json
from pathlib import Path
import sys
import time


def main():
    log = Path(sys.argv[1])
    state = Path(sys.argv[2])
    print(f"Watching {log}\n", flush=True)
    position = 0
    idle_after_finish = 0
    while True:
        if log.is_file():
            with log.open("rb") as stream:
                stream.seek(position)
                data = stream.read()
                position = stream.tell()
            if data:
                print(data.decode("utf-8", errors="replace").replace("\r", "\n"),
                      end="", flush=True)
        try:
            status = json.loads(state.read_text(encoding="utf-8")).get("status")
        except (OSError, ValueError):
            status = None
        if status in ("complete", "failed"):
            idle_after_finish += 1
            if idle_after_finish >= 3:
                print(f"\nRun {status}. Results and files are available in the dashboard.", flush=True)
                if sys.stdin.isatty():
                    try:
                        input("Press Enter to close this log terminal. ")
                    except (EOFError, KeyboardInterrupt):
                        pass
                return 0
        else:
            idle_after_finish = 0
        time.sleep(0.4)


if __name__ == "__main__":
    sys.exit(main())
