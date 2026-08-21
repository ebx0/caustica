#!/usr/bin/env bash
# caustica phantom launcher — ./phantoms.sh [gui|build|dataset|catalog|tissues|info|fetch]
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for candidate in .venv/Scripts/python.exe .venv/bin/python python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 || [ -x "$candidate" ]; then
        PY="$candidate"
        break
    fi
done
if [ -z "$PY" ]; then
    echo "no python found; create the venv with: python -m venv .venv" >&2
    exit 1
fi

exec "$PY" -m apps.phantom_launcher "$@"
