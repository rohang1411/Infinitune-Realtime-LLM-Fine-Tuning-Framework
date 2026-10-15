#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv-eval/bin/python ]]; then
  for candidate in python3.10 python3.11 python3.12; do
    if command -v "$candidate" >/dev/null 2>&1; then
      "$candidate" -m venv .venv-eval
      break
    fi
  done
fi
if [[ ! -x .venv-eval/bin/python ]]; then
  echo 'Install Python 3.10-3.12, then retry.' >&2
  exit 1
fi
required_hash="$(.venv-eval/bin/python -c 'import hashlib; print(hashlib.sha256(open("requirements-eval.txt", "rb").read()).hexdigest())')"
installed_hash="$(cat .venv-eval/requirements-eval.sha256 2>/dev/null || true)"
if [[ "$required_hash" != "$installed_hash" ]]; then
  .venv-eval/bin/python -m pip install --quiet --disable-pip-version-check -r requirements-eval.txt
  printf '%s' "$required_hash" > .venv-eval/requirements-eval.sha256
fi
if [[ $# -eq 0 ]]; then
  set -- run --suites integrity --profile smoke
fi
if [[ "${1:-}" == dashboard ]]; then
  shift
  exec .venv-eval/bin/python -m benchmarks.evaluation_dashboard "$@"
fi
exec .venv-eval/bin/python -m benchmarks.run "$@"
