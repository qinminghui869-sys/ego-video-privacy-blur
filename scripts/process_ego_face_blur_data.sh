#!/usr/bin/env bash
set -euo pipefail
# Forward arguments to the resumable batch entry point.
exec privacy-blur-batch "$@"
