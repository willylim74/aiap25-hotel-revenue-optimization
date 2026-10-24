#!/usr/bin/env bash
# Runs the end-to-end AIAP training pipeline.
#
# Per the assignment spec, this script does NOT create a virtual environment or install
# dependencies itself — the grading environment installs everything listed in
# requirements.txt before invoking this script. This just runs the pipeline in whatever
# Python environment it's called from.
#
# Usage:
#   ./run.sh                       # train with src/config.yaml
#   ./run.sh --config other.yaml   # any extra args are passed straight through to main.py
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python src/main.py "$@"
