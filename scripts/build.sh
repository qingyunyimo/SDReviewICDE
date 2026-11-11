#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR=${1:-${TMPDIR:-/tmp}/sdreviewicde-build}
cmake -S "$ROOT/SkipDisk" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DUSE_AIO=ON -DBLA_VENDOR=OpenBLAS
cmake --build "$BUILD_DIR" --target search_pcamixed_disk_index test_mixed_pivot --parallel "${JOBS:-2}"
