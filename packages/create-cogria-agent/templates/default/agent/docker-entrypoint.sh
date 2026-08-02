#!/usr/bin/env bash
# Pre-publish entrypoint: install the CogriaAgent framework from the local
# checkout mounted at /srv/framework, then launch the kernel.
#
# Once cogria-* are published to PyPI (C6), replace this with a real Dockerfile
# that does `pip install cogria-agentserv cogria-backend` at build time — see the
# README. Passing all three local packages in one `pip install` lets pip satisfy
# their inter-dependencies (agentserv→backend→contract) without hitting the index.
set -euo pipefail

FW=/srv/framework
echo "Installing CogriaAgent framework from $FW ..."
pip install --quiet --no-cache-dir \
  "$FW/packages/contract" \
  "$FW/packages/backend-py" \
  "$FW/packages/agentserv"

echo "Starting agentserv on :8001 ..."
exec uvicorn app.app:app --host 0.0.0.0 --port 8001
